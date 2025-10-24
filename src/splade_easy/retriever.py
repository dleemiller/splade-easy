# src/splade_easy/retriever.py

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .scoring import compute_splade_score
from .shard import ShardReader


@dataclass
class SearchResult:
    doc_id: str
    score: float
    metadata: dict
    text: Optional[str] = None


class SpladeRetriever:
    """Read-only retriever for searching SPLADE index."""

    def __init__(self, index_dir: str, mode: str = "disk"):
        """
        Create retriever.

        Args:
            index_dir: Index directory
            mode: 'disk' (scan from disk) or 'memory' (load in RAM)
        """
        self.index_dir = Path(index_dir)
        self.mode = mode

        # Load metadata and deleted IDs
        meta_path = self.index_dir / "metadata.json"
        with open(meta_path) as f:
            self.metadata = json.load(f)

        deleted_path = self.index_dir / "deleted_ids.txt"
        if deleted_path.exists():
            with open(deleted_path) as f:
                self.deleted_ids = set(line.strip() for line in f if line.strip())
        else:
            self.deleted_ids = set()

        # Memory mode: preload shards
        self.shard_cache = {}
        if mode == "memory":
            self._load_shards_to_memory()

    def _get_shard_paths(self) -> list[Path]:
        return sorted(self.index_dir.glob("shard_*.fb"))

    def _load_shards_to_memory(self):
        """Load all shards into memory."""
        for shard_path in self._get_shard_paths():
            reader = ShardReader(str(shard_path))
            self.shard_cache[shard_path] = list(reader.scan(load_text=True))

    def search(
        self,
        query_tokens: np.ndarray,
        query_weights: np.ndarray,
        top_k: int = 10,
        return_text: bool = False,
        num_workers: int = 1,
    ) -> list[SearchResult]:
        """
        Search with SPLADE vectors.

        Args:
            query_tokens: Query token IDs
            query_weights: Query token weights
            top_k: Number of results
            return_text: Whether to load full text
            num_workers: Number of parallel workers
        """
        shard_paths = self._get_shard_paths()

        if not shard_paths:
            return []

        if num_workers == 1 or len(shard_paths) == 1:
            results = []
            for shard_path in shard_paths:
                results.extend(
                    self._search_shard(shard_path, query_tokens, query_weights, top_k, return_text)
                )
        else:
            results = []
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        self._search_shard,
                        shard_path,
                        query_tokens,
                        query_weights,
                        top_k,
                        return_text,
                    ): shard_path
                    for shard_path in shard_paths
                }

                for future in as_completed(futures):
                    results.extend(future.result())

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def search_text(
        self, query: str, model, top_k: int = 10, return_text: bool = False, num_workers: int = 1
    ) -> list[SearchResult]:
        """
        Convenience method: encode query text and search.

        Args:
            query: Query text
            model: sentence-transformers model
            top_k: Number of results
            return_text: Whether to load full text
            num_workers: Number of parallel workers
        """
        from .utils import extract_splade_vectors

        encoding = model.encode(query)
        query_tokens, query_weights = extract_splade_vectors(encoding)

        return self.search(
            query_tokens=query_tokens,
            query_weights=query_weights,
            top_k=top_k,
            return_text=return_text,
            num_workers=num_workers,
        )

    def _search_shard(
        self,
        shard_path: Path,
        query_tokens: np.ndarray,
        query_weights: np.ndarray,
        top_k: int,
        return_text: bool,
    ) -> list[SearchResult]:
        """Search a single shard."""
        results = []

        if self.mode == "memory" and shard_path in self.shard_cache:
            docs = self.shard_cache[shard_path]
        else:
            reader = ShardReader(str(shard_path))
            docs = reader.scan(load_text=return_text)

        for doc in docs:
            if doc["doc_id"] in self.deleted_ids:
                continue

            score = compute_splade_score(
                doc["token_ids"], doc["weights"], query_tokens, query_weights
            )

            if score > 0:
                results.append(
                    SearchResult(
                        doc_id=doc["doc_id"],
                        score=score,
                        metadata=doc["metadata"],
                        text=doc.get("text"),
                    )
                )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def get(self, doc_id: str) -> Optional[dict]:
        """Get document by ID."""
        if doc_id in self.deleted_ids:
            return None

        for shard_path in self._get_shard_paths():
            reader = ShardReader(str(shard_path))
            for doc in reader.scan(load_text=True):
                if doc["doc_id"] == doc_id:
                    return doc

        return None

    def get_batch(self, doc_ids: list[str]) -> list[Optional[dict]]:
        """Get multiple documents by ID."""
        return [self.get(doc_id) for doc_id in doc_ids]
