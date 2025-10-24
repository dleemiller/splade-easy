# src/splade_easy/index.py

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .retriever import SpladeRetriever
from .scoring import ensure_sorted_splade_vector
from .shard import ShardReader, ShardWriter
from .utils import extract_model_id

logger = logging.getLogger(__name__)


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
        self.index_dir = Path(index_dir)
        self.shard_size_mb = shard_size_mb
        self.shard_size_bytes = shard_size_mb * 1024 * 1024

        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.index_dir / "metadata.json"
        self.deleted_path = self.index_dir / "deleted_ids.txt"

        if self.meta_path.exists():
            self._load_metadata()
        else:
            self._init_metadata()

        self.deleted_ids = self._load_deleted_ids()
        self.current_writer = None
        self.current_shard_idx = self._get_next_shard_idx()

    @classmethod
    def retriever(cls, index_dir: str, mode: str = "disk") -> SpladeRetriever:
        return SpladeRetriever(index_dir, mode)

    def _init_metadata(self):
        self.metadata = {
            "version": "0.1.0",
            "num_docs": 0,
            "num_shards": 0,
            "shard_size_mb": self.shard_size_mb,
            "model_id": None,
        }
        self._save_metadata()

    def _load_metadata(self):
        with open(self.meta_path) as f:
            self.metadata = json.load(f)

    def _save_metadata(self):
        with open(self.meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _load_deleted_ids(self) -> set:
        if not self.deleted_path.exists():
            return set()
        with open(self.deleted_path) as f:
            return set(line.strip() for line in f if line.strip())

    def _save_deleted_ids(self):
        with open(self.deleted_path, "w") as f:
            for doc_id in sorted(self.deleted_ids):
                f.write(f"{doc_id}\n")

    def _get_shard_paths(self) -> list[Path]:
        return sorted(self.index_dir.glob("shard_*.fb"))

    def _get_next_shard_idx(self) -> int:
        paths = self._get_shard_paths()
        if not paths:
            return 0
        last = paths[-1].stem.split("_")[1]
        return int(last) + 1

    def _get_current_writer(self) -> ShardWriter:
        if self.current_writer is None:
            shard_path = self.index_dir / f"shard_{self.current_shard_idx:04d}.fb"
            self.current_writer = ShardWriter(str(shard_path))
        return self.current_writer

    def _rotate_shard(self):
        if self.current_writer:
            self.current_writer.close()
            self.metadata["num_shards"] += 1
            self._save_metadata()

        self.current_shard_idx += 1
        self.current_writer = None

    def _ensure_flushed(self):
        if self.current_writer:
            self.current_writer.f.flush()

    def add(self, doc: Document) -> None:
        """Add a single document. Assumes vectors are already sorted/deduplicated."""
        writer = self._get_current_writer()

        writer.append(
            doc_id=doc.doc_id,
            text=doc.text,
            metadata=doc.metadata,
            token_ids=doc.token_ids,
            weights=doc.weights,
        )

        self.metadata["num_docs"] += 1

        if writer.size() >= self.shard_size_bytes:
            self._rotate_shard()

    def add_batch(self, docs: list[Document]) -> None:
        """Add multiple documents. Assumes vectors are already sorted/deduplicated."""
        for doc in docs:
            self.add(doc)

        self._ensure_flushed()
        self._save_metadata()

    def add_text(self, doc_id: str, text: str, metadata: dict, model) -> None:
        """Encode text and add document."""
        from .utils import extract_splade_vectors

        if self.metadata.get("model_id") is None:
            model_id = extract_model_id(model)
            self.metadata["model_id"] = model_id
            self._save_metadata()
            logger.info(f"Index created with model: {model_id}")

        encoding = model.encode(text)
        token_ids, weights = extract_splade_vectors(encoding)

        # Sort and deduplicate (only place this happens for single adds)
        token_ids, weights = ensure_sorted_splade_vector(token_ids, weights, deduplicate=True)

        doc = Document(
            doc_id=doc_id, text=text, metadata=metadata, token_ids=token_ids, weights=weights
        )
        self.add(doc)

    def add_texts(self, doc_ids: list, texts: list, metadatas: list, model) -> None:
        """Encode and add multiple documents."""
        from .utils import extract_splade_vectors

        if self.metadata.get("model_id") is None:
            model_id = extract_model_id(model)
            self.metadata["model_id"] = model_id
            self._save_metadata()
            logger.info(f"Index created with model: {model_id}")

        docs = []
        for doc_id, text, metadata in zip(doc_ids, texts, metadatas):
            encoding = model.encode(text)
            token_ids, weights = extract_splade_vectors(encoding)

            # Sort and deduplicate (only place this happens for batch adds)
            token_ids, weights = ensure_sorted_splade_vector(token_ids, weights, deduplicate=True)

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

    def delete(self, doc_id: str) -> bool:
        self._ensure_flushed()

        retriever = self.retriever(str(self.index_dir))
        if retriever.get(doc_id) is None:
            return False

        self.deleted_ids.add(doc_id)
        self._save_deleted_ids()
        self.metadata["num_docs"] -= 1
        self._save_metadata()
        return True

    def compact(self) -> None:
        if self.current_writer:
            self.current_writer.close()
            self.current_writer = None

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

        for shard_path in self._get_shard_paths():
            shard_path.unlink()

        self.current_shard_idx = 0
        self.metadata["num_shards"] = 0
        self.metadata["num_docs"] = 0
        self.deleted_ids.clear()

        self.add_batch(all_docs)
        self._save_deleted_ids()

    def stats(self) -> dict:
        self._ensure_flushed()
        total_size = sum(p.stat().st_size for p in self._get_shard_paths())
        return {
            "num_docs": self.metadata["num_docs"],
            "num_shards": len(self._get_shard_paths()),
            "deleted_docs": len(self.deleted_ids),
            "total_size_mb": total_size / (1024 * 1024),
        }

    def __len__(self) -> int:
        return self.metadata["num_docs"]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.current_writer:
            self.current_writer.close()
            self._save_metadata()
