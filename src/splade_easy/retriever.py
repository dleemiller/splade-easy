# src/splade_easy/retriever.py

import heapq
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .scoring import compute_splade_score, ensure_sorted_splade_vector
from .shard import ShardReader
from .utils import extract_model_id, extract_splade_vectors, get_shard_paths

logger = logging.getLogger(__name__)


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
        """Get shard paths"""
        return get_shard_paths(self.index_dir, self.metadata)

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
        # Ensure query vectors are sorted and deduplicated for optimal scoring
        query_tokens, query_weights = ensure_sorted_splade_vector(
            query_tokens, query_weights, deduplicate=True
        )

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

        # Use heapq.nlargest for efficient top-k merge
        return heapq.nlargest(top_k, results, key=lambda x: x.score)

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
        # Check if query model matches index model
        index_model_id = self.metadata.get("model_id")
        if index_model_id:
            query_model_id = extract_model_id(model)
            if query_model_id != "unknown" and index_model_id != query_model_id:
                logger.warning(
                    f"Model mismatch! Index was created with '{index_model_id}' "
                    f"but querying with '{query_model_id}'. Results may be inaccurate."
                )

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
        """Search a single shard using heapq for efficiency."""
        # Min heap of (score, index, result) - keeps the k largest scores
        # Index is used as tie-breaker to avoid comparing SearchResult objects
        heap = []
        doc_index = 0

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
                result = SearchResult(
                    doc_id=doc["doc_id"],
                    score=score,
                    metadata=doc["metadata"],
                    text=doc.get("text"),
                )

                if len(heap) < top_k:
                    heapq.heappush(heap, (score, doc_index, result))
                elif score > heap[0][0]:  # Better than worst in heap
                    heapq.heapreplace(heap, (score, doc_index, result))

                doc_index += 1

        # Return sorted descending by score (index is just for heap ordering)
        return [result for score, idx, result in sorted(heap, key=lambda x: x[0], reverse=True)]

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