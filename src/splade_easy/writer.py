from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np

from .index import Index
from .scoring import ensure_sorted_splade_vector
from .shard import ShardReader, ShardWriter
from .utils import (
    extract_model_id,
    extract_splade_vectors,
    hash_file,
)

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

    # Lifecycle
    def open(self) -> IndexWriter:
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
        """Set model and update manifest"""
        self._model = model
        manifest = self.index.manifest
        manifest.model_id = extract_model_id(model)
        manifest.save(self.index.root)
        self.index.refresh()

    # Write operations
    def insert(
        self,
        doc_id: str,
        text: str,
        metadata: dict | None = None,
        token_ids: np.ndarray | None = None,
        weights: np.ndarray | None = None,
    ):
        """Insert a single document"""
        if not self._opened:
            raise ValueError("Writer not opened")

        if metadata is None:
            metadata = {}

        # Handle encoding
        if token_ids is None and weights is None:
            model = self._get_model()
            encoding = model.encode(text, show_progress_bar=False)
            token_ids, weights = extract_splade_vectors(encoding)

        token_ids, weights = ensure_sorted_splade_vector(token_ids, weights, deduplicate=True)

        # Write to shard
        shard = self._begin_shard()
        shard.append(
            doc_id=doc_id, text=text, metadata=metadata, token_ids=token_ids, weights=weights
        )

        # Update manifest
        self.index.manifest.num_docs += 1

        # Auto-rotate if full
        if shard.size() >= self.shard_size_bytes:
            self._finalize_shard()

    def insert_batch(self, documents: List[Dict[str, Any]]):
        """Insert multiple documents"""
        for doc in documents:
            self.insert(
                doc_id=doc["doc_id"],
                text=doc["text"],
                metadata=doc.get("metadata", {}),
                token_ids=doc.get("token_ids"),
                weights=doc.get("weights"),
            )

    def commit(self):
        """Commit pending changes"""
        if self._writer:
            self._writer._flush()
            self._writer.f.flush()
            os.fsync(self._writer.f.fileno())
        self.index.manifest.save(self.index.root)
        self.index.refresh()

    # Maintenance operations
    def delete(self, doc_id: str) -> bool:
        """Mark document as deleted"""
        if not self._opened:
            raise ValueError("Writer not opened")

        # Check if doc exists
        if self.index.get(doc_id) is None:
            return False

        # Add to deleted IDs
        deleted_path = self.index.root / "deleted_ids.txt"
        deleted_ids = self.index._load_deleted_ids()
        deleted_ids.add(doc_id)

        with open(deleted_path, "w") as f:
            for doc_id_del in sorted(deleted_ids):
                f.write(f"{doc_id_del}\n")

        # Update manifest
        self.index.manifest.num_docs -= 1
        self.index.manifest.save(self.index.root)
        self.index.refresh()

        logger.info(f"Deleted {doc_id}")
        return True

    def compact(self) -> None:
        """Remove deleted documents by rebuilding shards"""
        if not self._opened:
            raise ValueError("Writer not opened")

        logger.info("Compacting index...")

        # Finalize current shard
        self._finalize_shard()

        # Collect non-deleted docs
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

        old_shards = len(self.index.iter_shards())

        # Delete old shards
        for shard_path in self.index.iter_shards():
            shard_path.unlink()

        # Reset manifest
        manifest = self.index.manifest
        manifest.shard_hashes = []
        manifest.num_docs = 0
        manifest.bump_commit()

        # Re-add docs
        self.insert_batch(all_docs)
        self._finalize_shard()

        # Clear deleted IDs
        deleted_path = self.index.root / "deleted_ids.txt"
        if deleted_path.exists():
            deleted_path.unlink()

        logger.info(f"Compacted {old_shards} shards -> {len(self.index.iter_shards())} shards")

    def reshard(self, target_shard_size_mb: int = 32, keep_originals: bool = False) -> dict:
        """Reshard index with new target size"""
        if not self._opened:
            raise ValueError("Writer not opened")

        from .reshard import IndexResharder

        # Finalize current shard
        self._finalize_shard()

        # Perform resharding
        with IndexResharder(
            str(self.index.root), target_shard_size_mb, keep_originals
        ) as resharder:
            stats = resharder.reshard()

        # Reload index state
        self.index.refresh()
        return stats

    # Internal helpers
    def _get_model(self):
        """Get model for encoding"""
        if self._model is not None:
            return self._model

        # Try loading from manifest
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
        return self._writer

    def _finalize_shard(self):
        """Finalize current shard"""
        if not self._writer:
            return

        self._writer.close()
        shard_hash = hash_file(self._temp_path)
        final_path = self.index.root / f"{shard_hash}.fb"
        os.replace(self._temp_path, final_path)

        # Update manifest
        manifest = self.index.manifest
        if shard_hash not in manifest.shard_hashes:
            manifest.shard_hashes.append(shard_hash)
        manifest.bump_commit()
        manifest.save(self.index.root)

        self._writer = None
        self._temp_path = None
        self.index.refresh()
