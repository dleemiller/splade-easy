from collections.abc import Iterable
from pathlib import Path
from typing import Optional

from .manifest import Manifest
from .shard import ShardReader


class Index:
    def __init__(self, root: str | Path, *, memory: bool = False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = Manifest.load(self.root)
        self.memory = memory
        self._cache: dict[Path, list[dict]] = dict()
        self._open = False

    # Lifecycle
    def open(self):
        if self._open:
            return self
        if self.memory:
            self._load_cache()
        self._open = True
        return self

    def close(self):
        self._cache.clear()
        self._open = False

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # Properties and stats
    def __len__(self) -> int:
        return self.manifest.num_docs

    @property
    def stats(self) -> dict:
        """Get index statistics"""
        deleted_ids = self._load_deleted_ids()
        total_size = sum(p.stat().st_size for p in self.iter_shards())

        return {
            "num_docs": len(self),
            "num_shards": len(self.iter_shards()),
            "deleted_docs": len(deleted_ids),
            "total_size_mb": total_size / (1024 * 1024),
            "model_id": self.manifest.model_id,
        }

    def get_model(self) -> Optional[str]:
        """Returns the model_id for the index."""
        return self.manifest.model_id

    # Manifest/visibility
    def refresh(self) -> None:
        """Refresh manifest and cache if needed"""
        new_manifest = Manifest.load(self.root)
        changed = new_manifest.commit_id != self.manifest.commit_id
        if changed:
            self.manifest = new_manifest
            if self.memory:
                self._load_cache()

    # Read helpers - delegate to shards
    def iter_shards(self) -> list[Path]:
        return self.manifest.shard_paths(self.root)

    def iter_docs(self, shard_path: Path, *, load_text: bool = False) -> Iterable[dict]:
        if self.memory and shard_path in self._cache:
            yield from self._cache[shard_path]
        else:
            reader = ShardReader(str(shard_path))
            yield from reader.scan(load_text=load_text)

    def get(self, doc_id: str) -> Optional[dict]:
        """Get document by ID"""
        deleted_ids = self._load_deleted_ids()
        if doc_id in deleted_ids:
            return None

        for shard_path in self.iter_shards():
            for doc in self.iter_docs(shard_path, load_text=True):
                if doc["doc_id"] == doc_id:
                    return doc
        return None

    # Factory methods for reader/writer
    def reader(self):
        """Create an IndexReader for this index"""
        from .reader import IndexReader

        return IndexReader(self)

    def writer(self, shard_size_mb: float = 32.0):
        """Create an IndexWriter for this index"""
        from .writer import IndexWriter

        return IndexWriter(self, shard_size_mb=shard_size_mb)

    # Internal helpers
    def _load_cache(self):
        """Load all shards into memory cache"""
        self._cache.clear()
        for shard_path in self.iter_shards():
            reader = ShardReader(str(shard_path))
            self._cache[shard_path] = list(reader.scan(load_text=True))

    def _load_deleted_ids(self) -> set[str]:
        """Load deleted document IDs"""
        deleted_path = self.root / "deleted_ids.txt"
        if deleted_path.exists():
            with open(deleted_path) as f:
                return set(line.strip() for line in f if line.strip())
        return set()
