from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .index import Index
from .inverted_index import InvertedIndexWriter
from .scoring import ensure_sorted_splade_vector
from .shard import ShardReader, ShardWriter
from .utils import extract_model_id, extract_splade_vectors, hash_file

logger = logging.getLogger(__name__)


class IndexWriter:
    """
    Writer for adding, modifying, and managing documents in a SPLADE index.

    Handles document insertion, deletion, compaction, and resharding.
    Automatically manages shard rotation and inverted index generation.

    Args:
        index: Index instance to write to
        shard_size_mb: Target shard size in MB (default: 32MB)

    Example:
        >>> index = Index("my_index")
        >>> with index.writer() as writer:
        ...     writer.set_model(model)
        ...     writer.insert(doc_id="1", text="Hello world", metadata={"source": "test"})
        ...     writer.delete("old_doc")
    """

    def __init__(self, index: Index, *, shard_size_mb: float | None = None):
        self.index = index
        self.shard_size_mb = shard_size_mb or 32.0
        self.shard_size_bytes = int(self.shard_size_mb * 1024 * 1024)
        self._opened = False
        self._writer: ShardWriter | None = None
        self._temp_path: Path | None = None
        self._model = None

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
        """
        Set the model for encoding documents.

        Updates the manifest with the model ID and makes it available
        for future insert operations.

        Args:
            model: SentenceTransformer model for encoding

        Example:
            >>> writer.set_model(model)
            >>> writer.insert(doc_id="1", text="Hello")  # Will use model to encode
        """
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
        """
        Insert a document into the index.

        Args:
            doc_id: Unique document identifier
            text: Document text content
            metadata: Optional metadata dictionary
            token_ids: Pre-encoded SPLADE token IDs (optional, will encode if not provided)
            weights: Pre-encoded SPLADE weights (optional, will encode if not provided)

        Raises:
            ValueError: If writer is not opened or no model is available for encoding

        Example:
            >>> writer.insert(
            ...     doc_id="1",
            ...     text="Machine learning paper",
            ...     metadata={"year": "2024"}
            ... )
        """
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
        shard.append(
            doc_id=doc_id, text=text, metadata=metadata, token_ids=token_ids, weights=weights
        )

        self.index.manifest.num_docs += 1

        if shard.size() >= self.shard_size_bytes:
            self._finalize_shard()

    def insert_batch(self, documents: list[dict[str, Any]]):
        """
        Insert multiple documents in batch.

        Args:
            documents: List of document dictionaries with keys: doc_id, text, metadata (optional),
                      token_ids (optional), weights (optional)

        Example:
            >>> writer.insert_batch([
            ...     {"doc_id": "1", "text": "Doc 1", "metadata": {"type": "A"}},
            ...     {"doc_id": "2", "text": "Doc 2", "metadata": {"type": "B"}},
            ... ])
        """
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
        """
        Mark a document as deleted.

        Adds the document ID to the deleted list without physically removing it.
        Use compact() to physically remove deleted documents and reclaim space.

        Args:
            doc_id: Document identifier to delete

        Returns:
            True if document was deleted, False if not found

        Raises:
            ValueError: If writer is not opened

        Example:
            >>> writer.delete("obsolete_doc")
            True
        """
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
        self.index.manifest.bump_commit()  # Document deletion - bump commit_id
        self.index.manifest.save(self.index.root)
        self.index.refresh()
        logger.info(f"Deleted {doc_id}")
        return True

    def compact(self) -> None:
        """
        Remove deleted documents and rebuild shards.

        Physically removes all deleted documents by rebuilding the index
        from scratch with only non-deleted documents. This reclaims disk space
        and clears the deleted_ids.txt file.

        Raises:
            ValueError: If writer is not opened

        Example:
            >>> writer.delete("doc1")
            >>> writer.delete("doc2")
            >>> writer.compact()  # Physically remove doc1 and doc2
        """
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
            inv_path = shard_path.with_suffix(".inv")
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
        with IndexResharder(
            str(self.index.root), target_shard_size_mb, keep_originals
        ) as resharder:
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

        self._writer.close()

        if not self._temp_path.exists():
            logger.warning(f"Temp shard file missing: {self._temp_path}")
            self._writer = None
            self._temp_path = None
            return

        shard_hash = hash_file(self._temp_path)
        final_path = self.index.root / f"{shard_hash}.fb"

        # Atomic rename using os.replace
        os.replace(self._temp_path, final_path)

        inv_index = self._writer.get_inverted_index()
        if inv_index is not None:
            inv_path = final_path.with_suffix(".inv")
            inv_writer = InvertedIndexWriter(str(inv_path))
            inv_writer.write(inv_index)
            inv_writer.close()
            logger.debug(f"Written inverted index: {inv_path.name}")

        self.index.manifest.shard_hashes.append(shard_hash)
        self.index.manifest.bump_commit()  # Structural change - bump commit_id
        self.index.manifest.save(self.index.root)

        logger.debug(f"Finalized shard: {shard_hash[:16]}...")

        self._writer = None
        self._temp_path = None
