# src/splade_easy/index.py

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .retriever import SpladeRetriever
from .shard import ShardReader, ShardWriter


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict[str, str]
    token_ids: np.ndarray
    weights: np.ndarray


class SpladeIndex:
    """Index for writing and maintaining SPLADE documents."""

    def __init__(self, index_dir: str, shard_size_mb: int = 32):
        """
        Create or load index.

        Args:
            index_dir: Directory to store index
            shard_size_mb: Target size for each shard in MB
        """
        self.index_dir = Path(index_dir)
        self.shard_size_mb = shard_size_mb
        self.shard_size_bytes = shard_size_mb * 1024 * 1024

        # Create directory
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.meta_path = self.index_dir / "metadata.json"
        self.deleted_path = self.index_dir / "deleted_ids.txt"

        # Load or create metadata
        if self.meta_path.exists():
            self._load_metadata()
        else:
            self._init_metadata()

        # Load deleted IDs
        self.deleted_ids = self._load_deleted_ids()

        # Current writer state
        self.current_writer = None
        self.current_shard_idx = self._get_next_shard_idx()

    @classmethod
    def retriever(cls, index_dir: str, mode: str = "disk") -> SpladeRetriever:
        """
        Create a retriever for searching the index.

        Args:
            index_dir: Index directory
            mode: 'disk' or 'memory'

        Returns:
            SpladeRetriever instance
        """
        return SpladeRetriever(index_dir, mode)

    # === PRIVATE: Metadata Management ===

    def _init_metadata(self):
        """Initialize new index metadata."""
        self.metadata = {
            "version": "0.1.0",
            "num_docs": 0,
            "num_shards": 0,
            "shard_size_mb": self.shard_size_mb,
        }
        self._save_metadata()

    def _load_metadata(self):
        """Load existing metadata."""
        with open(self.meta_path) as f:
            self.metadata = json.load(f)

    def _save_metadata(self):
        """Save metadata to disk."""
        with open(self.meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _load_deleted_ids(self) -> set:
        """Load deleted document IDs."""
        if not self.deleted_path.exists():
            return set()
        with open(self.deleted_path) as f:
            return set(line.strip() for line in f if line.strip())

    def _save_deleted_ids(self):
        """Save deleted document IDs."""
        with open(self.deleted_path, "w") as f:
            for doc_id in sorted(self.deleted_ids):
                f.write(f"{doc_id}\n")

    # === PRIVATE: Shard Management ===

    def _get_shard_paths(self) -> list[Path]:
        """Get all shard file paths."""
        return sorted(self.index_dir.glob("shard_*.fb"))

    def _get_next_shard_idx(self) -> int:
        """Get next shard index."""
        paths = self._get_shard_paths()
        if not paths:
            return 0
        last = paths[-1].stem.split("_")[1]
        return int(last) + 1

    def _get_current_writer(self) -> ShardWriter:
        """Get or create writer for current shard."""
        if self.current_writer is None:
            shard_path = self.index_dir / f"shard_{self.current_shard_idx:04d}.fb"
            self.current_writer = ShardWriter(str(shard_path))
        return self.current_writer

    def _rotate_shard(self):
        """Close current shard and start new one."""
        if self.current_writer:
            self.current_writer.close()
            self.metadata["num_shards"] += 1
            self._save_metadata()

        self.current_shard_idx += 1
        self.current_writer = None

    def _ensure_flushed(self):
        """Ensure all writes are flushed to disk."""
        if self.current_writer:
            self.current_writer.f.flush()

    # === PUBLIC: Write Operations ===

    def add(self, doc: Document) -> None:
        """
        Add a single document.

        Args:
            doc: Document to add
        """
        writer = self._get_current_writer()

        writer.append(
            doc_id=doc.doc_id,
            text=doc.text,
            metadata=doc.metadata,
            token_ids=doc.token_ids,
            weights=doc.weights,
        )

        self.metadata["num_docs"] += 1

        # Rotate if shard is too large
        if writer.size() >= self.shard_size_bytes:
            self._rotate_shard()

    def add_batch(self, docs: list[Document]) -> None:
        """
        Add multiple documents.

        Args:
            docs: List of documents to add
        """
        for doc in docs:
            self.add(doc)

        # Flush and save metadata
        self._ensure_flushed()
        self._save_metadata()

    def add_text(self, doc_id: str, text: str, metadata: dict, model) -> None:
        """
        Convenience method: encode text and add document.

        Args:
            doc_id: Document ID
            text: Document text
            metadata: Document metadata
            model: sentence-transformers model with encode() method
        """
        from .utils import extract_splade_vectors

        encoding = model.encode(text)
        token_ids, weights = extract_splade_vectors(encoding)

        doc = Document(
            doc_id=doc_id, text=text, metadata=metadata, token_ids=token_ids, weights=weights
        )
        self.add(doc)

    def add_texts(self, doc_ids: list, texts: list, metadatas: list, model) -> None:
        """
        Convenience method: encode and add multiple documents.

        Args:
            doc_ids: List of document IDs
            texts: List of document texts
            metadatas: List of metadata dicts
            model: sentence-transformers model
        """
        from .utils import extract_splade_vectors

        docs = []
        for doc_id, text, metadata in zip(doc_ids, texts, metadatas):
            encoding = model.encode(text)
            token_ids, weights = extract_splade_vectors(encoding)

            docs.append(
                Document(
                    doc_id=doc_id,
                    text=text,
                    metadata=metadata,
                    token_ids=token_ids,
                    weights=weights,
                )
            )

        self.add_batch(docs)

    # === PUBLIC: Maintenance Operations ===

    def delete(self, doc_id: str) -> bool:
        """
        Soft-delete a document.

        Args:
            doc_id: Document ID to delete

        Returns:
            True if document existed and was deleted, False otherwise
        """
        # Ensure data is flushed before checking
        self._ensure_flushed()

        # Check if document exists
        retriever = self.retriever(str(self.index_dir))
        if retriever.get(doc_id) is None:
            return False

        # Mark as deleted
        self.deleted_ids.add(doc_id)
        self._save_deleted_ids()
        self.metadata["num_docs"] -= 1
        self._save_metadata()
        return True

    def compact(self) -> None:
        """
        Remove deleted documents and rebuild shards.
        This physically removes deleted documents from disk.
        """
        # Close current writer
        if self.current_writer:
            self.current_writer.close()
            self.current_writer = None

        # Collect all non-deleted documents
        all_docs = []
        for shard_path in self._get_shard_paths():
            reader = ShardReader(str(shard_path))
            for doc in reader.scan(load_text=True):
                if doc["doc_id"] not in self.deleted_ids:
                    all_docs.append(
                        Document(
                            doc_id=doc["doc_id"],
                            text=doc["text"],
                            metadata=doc["metadata"],
                            token_ids=doc["token_ids"],
                            weights=doc["weights"],
                        )
                    )

        # Remove old shards
        for shard_path in self._get_shard_paths():
            shard_path.unlink()

        # Reset state
        self.current_shard_idx = 0
        self.metadata["num_shards"] = 0
        self.metadata["num_docs"] = 0
        self.deleted_ids.clear()

        # Rewrite all documents
        self.add_batch(all_docs)
        self._save_deleted_ids()

    def stats(self) -> dict:
        """
        Get index statistics.

        Returns:
            Dictionary with num_docs, num_shards, deleted_docs, total_size_mb
        """
        self._ensure_flushed()

        total_size = sum(p.stat().st_size for p in self._get_shard_paths())

        return {
            "num_docs": self.metadata["num_docs"],
            "num_shards": len(self._get_shard_paths()),
            "deleted_docs": len(self.deleted_ids),
            "total_size_mb": total_size / (1024 * 1024),
        }

    def __len__(self) -> int:
        """Return number of active documents."""
        return self.metadata["num_docs"]

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure writer is closed."""
        if self.current_writer:
            self.current_writer.close()
            self._save_metadata()
