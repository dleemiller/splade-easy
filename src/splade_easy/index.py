from dataclasses import dataclass

import numpy as np


@dataclass
class Document:
    doc_id: str
    text: str | bytes
    metadata: dict[str, str]
    token_ids: np.ndarray
    weights: np.ndarray


import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

from .inverted_index import InvertedIndexReader
from .manifest import Manifest
from .shard import ShardReader

logger = logging.getLogger(__name__)


class Index:
    def __init__(self, root: str | Path, *, memory: bool = False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = Manifest.load(self.root)
        self.memory = memory
        self._cache: dict[Path, list[dict]] = dict()
        self._inv_index_cache: dict[Path, InvertedIndexReader] = (
            dict()
        )  # NEW: Cache inverted indices
        self._opened = False
        self._pos_index: dict[str, tuple[Path, int, int]] = {}
        self._doc_id_to_idx: dict[str, int] = {}
        self._idx_to_doc_id: list[str] = []
        logger.debug(f"Created Index: root={self.root}, memory={self.memory}")

    def load(self):
        logger.debug(f"load() called: memory={self.memory}, opened={self._opened}")
        if not self.memory:
            return self
        if not self._opened:
            self.open()
        return self

    def open(self):
        if self._opened:
            return self
        self._opened = True
        if self.memory:
            self._load_cache()
        return self

    def close(self):
        self._cache.clear()
        # NEW: Close inverted index cache
        for inv_reader in self._inv_index_cache.values():
            inv_reader.close()
        self._inv_index_cache.clear()
        self._opened = False

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self) -> int:
        return self.manifest.num_docs

    @property
    def stats(self) -> dict:
        deleted_ids = self._load_deleted_ids()
        total_size = sum(p.stat().st_size for p in self.iter_shards())

        return {
            "num_docs": len(self),
            "num_shards": len(self.iter_shards()),
            "num_cached_inverted_indices": len(self._inv_index_cache),  # NEW
            "deleted_docs": len(deleted_ids),
            "total_size_mb": total_size / (1024 * 1024),
            "model_id": self.manifest.model_id,
        }

    def get_model(self) -> Optional[str]:
        return self.manifest.model_id

    def refresh(self) -> None:
        new_manifest = Manifest.load(self.root)
        changed = new_manifest.commit_id != self.manifest.commit_id
        if changed:
            self.manifest = new_manifest
            if self.memory:
                self._load_cache()

    def iter_shards(self) -> list[Path]:
        return self.manifest.shard_paths(self.root)

    def iter_docs(self, shard_path: Path, *, load_text: bool = False) -> Iterable[dict]:
        if self.memory and shard_path in self._cache:
            yield from self._cache[shard_path]
        else:
            reader = ShardReader(str(shard_path))
            yield from reader.scan(load_text=load_text)

    def get(self, doc_id: str) -> Optional[dict]:
        deleted_ids = self._load_deleted_ids()
        if doc_id in deleted_ids:
            return None

        # If in memory mode, search cache first
        if self.memory and self._cache:
            for docs in self._cache.values():
                for doc in docs:
                    if doc["doc_id"] == doc_id:
                        return doc

        # Fallback to disk search
        for shard_path in self.iter_shards():
            for doc in self.iter_docs(shard_path, load_text=True):
                if doc["doc_id"] == doc_id:
                    return doc
        return None

    def reader(self, *, memory: bool = None):
        from .reader import IndexReader

        # If memory mode is explicitly requested or index is already in memory mode
        if memory is True or (memory is None and self.memory):
            # Ensure we're loaded into memory
            if not self._opened:
                self.load()
        return IndexReader(self, memory_mode=memory)

    def writer(self, shard_size_mb: float = 32.0):
        from .writer import IndexWriter

        return IndexWriter(self, shard_size_mb=shard_size_mb)

    def _load_cache(self):
        logger.debug("_load_cache() called")
        self._cache.clear()
        self._pos_index.clear()
        self._inv_index_cache.clear()
        self._offset_to_doc_id: Dict[tuple[str, int], str] = {}  # NEW
        self._doc_id_to_idx.clear()
        self._idx_to_doc_id.clear()

        # First pass: collect all doc_ids to create a stable mapping
        all_doc_ids = []
        for shard_path in self.iter_shards():
            reader = ShardReader(str(shard_path))
            for rec in reader.scan(load_text=False, want_positions=False, light=True):
                all_doc_ids.append(rec["doc_id"])
        
        self._idx_to_doc_id = sorted(list(set(all_doc_ids)))
        self._doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(self._idx_to_doc_id)}

        for shard_path in self.iter_shards():
            reader = ShardReader(str(shard_path))
            docs = []
            for rec in reader.scan(load_text=False, want_positions=True, light=False):
                docs.append(rec)
                did = rec["doc_id"]
                off, mlen = rec["_pos"]
                self._pos_index[did] = (shard_path, off, mlen)
                self._offset_to_doc_id[(str(shard_path), off)] = did
            self._cache[shard_path] = docs
            
            inv_path = shard_path.with_suffix('.inv')
            if inv_path.exists():
                self._inv_index_cache[shard_path] = InvertedIndexReader(str(inv_path), shard_path)

    def _load_deleted_ids(self) -> set[str]:
        deleted_path = self.root / "deleted_ids.txt"
        if deleted_path.exists():
            with open(deleted_path) as f:
                return set(line.strip() for line in f if line.strip())
        return set()
