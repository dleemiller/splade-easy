from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np

from .index import Index
from .inverted_index import InvertedIndexWriter
from .scoring import ensure_sorted_splade_vector
from .shard import ShardReader, ShardWriter
from .utils import extract_model_id, extract_splade_vectors, hash_file

logger = logging.getLogger(__name__)


class IndexWriter:
    def __init__(self, index: Index, *, shard_size_mb: Optional[float] = None):
        self.index = index
        self.shard_size_mb = shard_size_mb or 32.0
        self.shard_size_bytes = int(self.shard_size_mb * 1024 * 1024)
        self._opened = False
        self._writer: Optional[ShardWriter] = None
        self._temp_path: Optional[Path] = None
        self._model = None

    def open(self) -> "IndexWriter":
        if self._opened:
            return self
        self._opened = True
        return self

    def close(self) -> None:
        try:
            self.commit()
        finally:
            self._opened = False

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def set_model(self, model) -> None:
        self._model = model
        manifest = self.index.manifest
        manifest.model_id = extract_model_id(model)
        manifest.save(self.index.root)
        self.index.refresh()

    def insert(
        self,
        doc_id: str,
        text: str | bytes,
        metadata: dict | None = None,
        token_ids: np.ndarray | None = None,
        weights: np.ndarray | None = None,
    ):
        if not self._opened:
            raise ValueError("Writer not opened")

        if metadata is None:
            metadata = {}

        if token_ids is None and weights is None:
            model = self._get_model()
            encoding = model.encode(text, show_progress_bar=False)
            token_ids, weights = extract_splade_vectors(encoding)

        token_ids, weights = ensure_sorted_splade_vector(token_ids, weights, deduplicate=True)

        shard = self._begin_shard()
        shard.append(doc_id=doc_id, text=text, metadata=metadata, token_ids=token_ids, weights=weights)

        self.index.manifest.num_docs += 1

        if shard.size() >= self.shard_size_bytes:
            self._finalize_shard()

    def insert_batch(self, documents: List[Dict[str, Any]]):
        for doc in documents:
            self.insert(
                doc_id=doc["doc_id"],
                text=doc["text"],
                metadata=doc.get("metadata", {}),
                token_ids=doc.get("token_ids"),
                weights=doc.get("weights"),
            )

    def commit(self) -> None:
        """Finalize any open shard and update manifest."""
        if self._writer is not None:
            self._finalize_shard()
        self.index.manifest.save(self.index.root)

    def delete(self, doc_id: str) -> bool:
        if not self._opened:
            raise ValueError("Writer not opened")

        if self.index.get(doc_id) is None:
            return False

        deleted_path = self.index.root / "deleted_ids.txt"
        deleted_ids = self.index._load_deleted_ids()
        deleted_ids.add(doc_id)

        with open(deleted_path, "w") as f:
            for doc_id_del in sorted(deleted_ids):
                f.write(f"{doc_id_del}\n")

        self.index.manifest.num_docs -= 1
        self.index.manifest.save(self.index.root)
        self.index.refresh()
        logger.info(f"Deleted {doc_id}")
        return True

    def compact(self) -> None:
        if not self._opened:
            raise ValueError("Writer not opened")

        logger.info("Compacting index...")
        self._finalize_shard()

        deleted_ids = self.index._load_deleted_ids()
        all_docs = []
        for shard_path in self.index.iter_shards():
            reader = ShardReader(str(shard_path))
            for doc in reader.scan(load_text=True):
                if doc["doc_id"] not in deleted_ids:
                    all_docs.append(
                        {
                            "doc_id": doc["doc_id"],
                            "text": doc["text"],
                            "metadata": doc["metadata"],
                            "token_ids": doc["token_ids"],
                            "weights": doc["weights"],
                        }
                    )

        for shard_path in self.index.iter_shards():
            shard_path.unlink()
            # Also delete inverted index files
            inv_path = shard_path.with_suffix('.inv')
            if inv_path.exists():
                inv_path.unlink()

        manifest = self.index.manifest
        manifest.shard_hashes = []
        manifest.num_docs = 0
        manifest.bump_commit()

        self.insert_batch(all_docs)
        self._finalize_shard()

        deleted_path = self.index.root / "deleted_ids.txt"
        if deleted_path.exists():
            deleted_path.unlink()

        logger.info("Compaction complete")

    def reshard(self, target_shard_size_mb: int = 32, keep_originals: bool = False) -> dict:
        if not self._opened:
            raise ValueError("Writer not opened")
        from .reshard import IndexResharder
        self._finalize_shard()
        with IndexResharder(str(self.index.root), target_shard_size_mb, keep_originals) as resharder:
            stats = resharder.reshard()
        self.index.refresh()
        return stats

    def _get_model(self):
        if self._model is not None:
            return self._model
        model_id = self.index.manifest.model_id
        if model_id:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(model_id)
                return self._model
            except Exception:
                pass
        raise ValueError("No model available. Call set_model() or provide token_ids/weights")

    def _begin_shard(self) -> ShardWriter:
        if self._writer is None:
            self._temp_path = self.index.root / f"_temp_shard_{uuid4().hex}.fb"
            self._writer = ShardWriter(str(self._temp_path))
            self._writer.enable_inverted_index()
        return self._writer

    def _finalize_shard(self) -> None:
        """Finalize current shard if it exists."""
        if self._writer is None:
            return
        
        # Close shard writer
        self._writer.close()
        
        # Check if temp file exists
        if not self._temp_path.exists():
            logger.warning(f"Temp shard file missing: {self._temp_path}")
            self._writer = None
            self._temp_path = None
            return
        
        # Hash and rename temp shard to final location
        shard_hash = hash_file(self._temp_path)
        final_path = self.index.root / f"{shard_hash}.fb"
        
        # Atomic rename using os.replace
        os.replace(self._temp_path, final_path)
        
        # Write inverted index if available
        inv_index = self._writer.get_inverted_index()
        if inv_index is not None:
            inv_path = final_path.with_suffix('.inv')
            inv_writer = InvertedIndexWriter(str(inv_path))
            inv_writer.write(inv_index)
            inv_writer.close()
            logger.debug(f"Written inverted index: {inv_path.name}")
        
        # Update manifest
        self.index.manifest.shard_hashes.append(shard_hash)
        self.index.manifest.save(self.index.root)
        
        logger.debug(f"Finalized shard: {shard_hash[:16]}...")
        
        # Clean up
        self._writer = None
        self._temp_path = None