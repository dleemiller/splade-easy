"""
Optimized IndexReader with inverted index support:
- ThreadPool for memory-mode parallel search
- Lazy text loading
- heapq.nlargest for merging
- Inverted index for sub-linear search
"""

import heapq
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numba
import numpy as np
from numba.typed import Dict as NumbaDict

from .cache import NumbaCache
from .index import Index
from .schemas import SearchResult
from .scoring import compute_splade_score, ensure_sorted_splade_vector
from .shard import ShardReader
from .utils import extract_model_id, extract_splade_vectors

logger = logging.getLogger(__name__)


@numba.njit(fastmath=True, cache=True, parallel=False)
def _accumulate_scores(
    query_tokens: np.ndarray,
    query_weights: np.ndarray,
    inv_indices: NumbaDict,
    deleted_doc_indices: np.ndarray,
    num_docs: int,
) -> np.ndarray:
    doc_scores = np.zeros(num_docs, dtype=np.float32)

    # Accumulate scores using direct doc_idx (no dictionary lookup)
    for i in range(len(query_tokens)):
        token_id = query_tokens[i]
        query_weight = query_weights[i]
        if token_id in inv_indices:
            postings_arr, num_postings_arr = inv_indices[token_id]
            num_postings = num_postings_arr[0]

            # Tight inner loop - direct array indexing, no dict lookup
            for j in range(num_postings):
                doc_idx = postings_arr[j]["doc_idx"]
                doc_weight = postings_arr[j]["weight"]
                if doc_idx >= 0:  # Valid doc_idx
                    doc_scores[doc_idx] += query_weight * doc_weight

    # Handle deletions
    if len(deleted_doc_indices) > 0:
        doc_scores[deleted_doc_indices] = 0.0

    return doc_scores


def _search_shard_worker(args):
    """
    Disk worker (process-based) with inverted index support.
    """
    try:
        shard_path, query_tokens, query_weights, top_k, return_text, deleted_ids, use_inv_idx = args

        # Disk mode always uses linear scan
        # (Inverted index search requires memory mode for performance)
        reader = ShardReader(str(shard_path))

        heap = []
        doc_index = 0

        for doc in reader.scan(load_text=return_text):
            if doc["doc_id"] in deleted_ids:
                continue

            score = compute_splade_score(
                doc["token_ids"], doc["weights"], query_tokens, query_weights
            )

            if score > 0:
                result = SearchResult(
                    doc_id=doc["doc_id"],
                    score=score,
                    metadata=doc["metadata"],
                    text=doc.get("text") if return_text else None,
                )

                if len(heap) < top_k:
                    heapq.heappush(heap, (score, doc_index, result))
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, (score, doc_index, result))

                doc_index += 1

        return [r for _, _, r in sorted(heap, key=lambda x: x[0], reverse=True)]

    except Exception as e:
        logger.error(f"Error searching shard {shard_path}: {e}")
        return []


def _search_memory_shard_worker(args):
    """
    Thread worker for memory mode with optional inverted index.
    """
    try:
        docs, query_tokens, query_weights, top_k, return_text, deleted_ids = args

        heap = []
        doc_index = 0

        for doc in docs:
            if doc["doc_id"] in deleted_ids:
                continue

            score = compute_splade_score(
                doc["token_ids"], doc["weights"], query_tokens, query_weights
            )

            if score > 0:
                result = SearchResult(
                    doc_id=doc["doc_id"],
                    score=score,
                    metadata=doc["metadata"],
                    text=doc.get("text") if return_text else None,
                )

                if len(heap) < top_k:
                    heapq.heappush(heap, (score, doc_index, result))
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, (score, doc_index, result))

                doc_index += 1

        return [r for _, _, r in sorted(heap, key=lambda x: x[0], reverse=True)]

    except Exception as e:
        logger.error(f"Error in memory shard worker: {e}")
        return []


class IndexReader:
    """
    High-performance reader for SPLADE indices with automatic optimization.

    Supports multiple search strategies:
    - Inverted index search (~1ms for 26K docs)
    - Full scan search (~55ms for 26K docs)
    - Memory or disk modes
    - Parallel processing

    The reader automatically selects the best strategy based on:
    - Whether inverted indices are available
    - Memory vs disk mode
    - Number of workers

    **Cache Behavior:**
    IndexReader caches data structures for performance in memory mode.
    Cache validity is checked once per search (not per property access).

    After external index modifications, call index.refresh() to reload manifest.
    The cache will detect staleness and rebuild automatically:
        >>> reader = index.reader(memory=True)
        >>> results = reader.search(...)  # Fast, uses cache
        >>>
        >>> # After external modification
        >>> index.refresh()  # Reload manifest
        >>> results = reader.search(...)  # Detects staleness, rebuilds (~30ms)
        >>> results = reader.search(...)  # Fast again (~1ms)

    The cache check has negligible overhead (~1 UUID comparison per search)
    but avoids repeated checks during property access in the hot path.

    Example:
        >>> index = Index("my_index")
        >>> reader = index.reader(memory=True, num_workers=4)
        >>> results = reader.search(query="machine learning", model=model, top_k=10)
    """

    def __init__(self, index: Index, num_workers: int = 4, memory_mode: bool = None):
        self.index = index
        self.num_workers = num_workers
        self.memory_mode = memory_mode if memory_mode is not None else index.memory
        self._deleted_ids_cache: Optional[set[str]] = None
        self._process_pool: Optional[ProcessPoolExecutor] = None
        self._thread_pool: Optional[ThreadPoolExecutor] = None

        # Check if inverted indices exist
        self._inverted_index_available = self._check_inverted_index()
        logger.info(f"Inverted index available: {self._inverted_index_available}")
        logger.info(f"Shards: {[p.name for p in self.index.iter_shards()]}")
        for shard_path in self.index.iter_shards():
            inv_path = shard_path.with_suffix(".inv")
            logger.debug(f"  {shard_path.name} -> inv exists: {inv_path.exists()}")

        # Force memory loading if memory_mode is requested
        if self.memory_mode:
            self.index.memory = True
            if not self.index._opened:
                self.index.load()
            self._deleted_ids_cache = self._load_deleted_ids_cached()

            # Create NumbaCache for inverted index search (only in memory mode)
            self._numba_cache = NumbaCache(self.index) if self._inverted_index_available else None
        else:
            self._numba_cache = None

    def load(self):
        self.index.load()
        self._deleted_ids_cache = self._load_deleted_ids_cached()
        return self

    def _load_deleted_ids_cached(self) -> set[str]:
        deleted_path = self.index.root / "deleted_ids.txt"
        if deleted_path.exists():
            with open(deleted_path) as f:
                return set(line.strip() for line in f if line.strip())
        return set()

    def _check_inverted_index(self) -> bool:
        """Check if inverted index files exist for all shards"""
        for shard_path in self.index.iter_shards():
            inv_path = shard_path.with_suffix(".inv")
            if not inv_path.exists():
                return False
        return True

    def _is_deleted(self, doc_id: str) -> bool:
        if self._deleted_ids_cache is None:
            self._deleted_ids_cache = self._load_deleted_ids_cached()
        return doc_id in self._deleted_ids_cache

    def _prepare_query_vectors(
        self,
        query: Optional[str],
        query_tokens: Optional[np.ndarray],
        query_weights: Optional[np.ndarray],
        model,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare query vectors for search (encode if needed, validate, sort).

        Args:
            query: Query text (if provided, requires model)
            query_tokens: Pre-encoded query token IDs
            query_weights: Pre-encoded query weights
            model: SPLADE model for encoding (required if query is text)

        Returns:
            Tuple of (sorted_tokens, sorted_weights) ready for search

        Raises:
            ValueError: If neither query text nor vectors provided, or model missing
        """
        if query is not None:
            if model is None:
                raise ValueError("Model required for text search")
            # Validate model compatibility
            index_model_id = self.index.manifest.model_id
            query_model_id = extract_model_id(model)
            if query_model_id != "unknown" and index_model_id != query_model_id:
                raise ValueError(
                    f"Model mismatch! Index: {index_model_id}, Query: {query_model_id}"
                )
            encoding = model.encode(query, show_progress_bar=False)
            query_tokens, query_weights = extract_splade_vectors(encoding)
        elif query_tokens is not None and query_weights is not None:
            pass
        else:
            raise ValueError("Must provide either query text or query_tokens/weights")

        # Sort and deduplicate for optimal search performance
        return ensure_sorted_splade_vector(query_tokens, query_weights, deduplicate=True)

    def _select_search_method(self):
        """
        Select the appropriate search method based on configuration.

        Always uses inverted index when available for optimal performance.

        Returns:
            Search method to use
        """
        # Priority order:
        # 1. Inverted index in memory (fastest)
        # 2. Full scan in memory (slower)
        # 3. Full scan on disk (slowest)

        if self.memory_mode and self.index._cache:
            if self._inverted_index_available:
                return self._search_with_inverted_index_memory
            return self._search_with_full_scan_memory
        else:
            # Disk mode always uses full scan (inverted index on disk has no benefit)
            return self._search_with_full_scan_disk

    def _attach_texts(self, results: list[SearchResult], return_text: bool) -> list[SearchResult]:
        if not return_text or not results:
            return results

        needed = {r.doc_id for r in results if r.text is None}
        if not needed:
            return results

        # Build per-shard fetch lists
        per_shard: dict[Path, list[str]] = {}
        for did in needed:
            pos = self.index._pos_index.get(did)
            if pos is None:
                continue
            shard_path, rec_offset, _ = pos
            per_shard.setdefault(shard_path, []).append(did)

        # Fetch only those records' text, zero-copy
        for shard_path, doc_ids in per_shard.items():
            reader = ShardReader(str(shard_path))
            for did in doc_ids:
                # Find offset from position index
                pos_info = self.index._pos_index.get(did)
                if pos_info:
                    _, rec_offset, _ = pos_info
                    text_mv = reader.read_text_at(rec_offset)
                    if text_mv is not None:
                        text = text_mv.tobytes().decode("utf-8")
                        for r in results:
                            if r.doc_id == did and r.text is None:
                                r.text = text
                                break

        return results

    def search(
        self,
        query: Optional[str] = None,
        query_tokens: Optional[np.ndarray] = None,
        query_weights: Optional[np.ndarray] = None,
        model=None,
        *,
        top_k: int = 10,
        return_text: bool = False,
        num_workers: Optional[int] = None,
    ) -> list[SearchResult]:
        """
        Search the index for documents matching the query.

        Accepts either text query (with model) or pre-encoded SPLADE vectors.
        Automatically uses inverted index when available for optimal performance.

        Args:
            query: Query text (requires model parameter)
            query_tokens: Pre-encoded SPLADE token IDs (alternative to query)
            query_weights: Pre-encoded SPLADE weights (alternative to query)
            model: SentenceTransformer model for encoding (required if query is text)
            top_k: Number of results to return (default: 10)
            return_text: Whether to load full document text (default: False)
            num_workers: Number of parallel workers (None = use reader default)

        Returns:
            List of SearchResult objects, sorted by score (highest first)

        Raises:
            ValueError: If neither query nor vectors provided, or model mismatch

        Examples:
            >>> # Text query
            >>> reader = index.reader(memory=True)
            >>> results = reader.search(
            ...     query="machine learning",
            ...     model=model,
            ...     top_k=10
            ... )

            >>> # Pre-encoded vectors (for profiling)
            >>> tokens, weights = IndexReader.encode_query(query, model)
            >>> results = reader.search(
            ...     query_tokens=tokens,
            ...     query_weights=weights,
            ...     top_k=10
            ... )
        """
        if num_workers is None:
            num_workers = self.num_workers

        query_tokens, query_weights = self._prepare_query_vectors(
            query, query_tokens, query_weights, model
        )

        deleted_ids = self._deleted_ids_cache or self._load_deleted_ids_cached()

        search_method = self._select_search_method()
        return search_method(
            query_tokens, query_weights, top_k, return_text, deleted_ids, num_workers
        )

    def _search_with_inverted_index_memory(
        self,
        query_tokens: np.ndarray,
        query_weights: np.ndarray,
        top_k: int,
        return_text: bool,
        deleted_ids: set[str],
        num_workers: int,
    ) -> list[SearchResult]:
        """Search using inverted index in memory mode (fast, ~1ms for 26K docs)."""
        if not self._inverted_index_available or self._numba_cache is None:
            logger.debug("Inverted index not available, falling back to full scan")
            return self._search_with_full_scan_memory(
                query_tokens, query_weights, top_k, return_text, deleted_ids, num_workers
            )

        # Check cache validity once per search (not per property access)
        # This happens after index.refresh() when manifest changes
        if self._numba_cache.is_stale():
            logger.debug("Cache stale, rebuilding")
            self._numba_cache.build()

        deleted_doc_indices = np.array(
            [
                self.index._doc_id_to_idx[did]
                for did in deleted_ids
                if did in self.index._doc_id_to_idx
            ],
            dtype=np.int64,
        )

        # Run Numba-optimized scoring (cache already validated above)
        doc_scores = _accumulate_scores(
            query_tokens,
            query_weights,
            self._numba_cache.postings,  # Lazy-loads if needed, no validation overhead
            deleted_doc_indices,
            len(self.index._idx_to_doc_id),
        )

        # Find top-k scores (argpartition handles k > num_nonzero gracefully)
        top_indices = np.argpartition(doc_scores, -top_k)[-top_k:]
        top_scores = doc_scores[top_indices]

        # Filter out zero scores and sort
        nonzero_mask = top_scores > 0
        if not nonzero_mask.any():
            return []

        top_indices = top_indices[nonzero_mask]
        top_scores = top_scores[nonzero_mask]
        sorted_top_indices = top_indices[np.argsort(top_scores)[::-1]]

        # Build results using cached doc_lookup (list comprehension for speed)
        results = [
            SearchResult(
                doc_id=self.index._idx_to_doc_id[doc_idx],
                score=doc_scores[doc_idx],
                metadata=doc["metadata"],
                text=doc.get("text") if return_text else None,
            )
            for doc_idx in sorted_top_indices
            if (doc := self._numba_cache.doc_lookup.get(self.index._idx_to_doc_id[doc_idx]))
            is not None
        ]

        return self._attach_texts(results, return_text)

    def _search_with_inverted_index_disk(
        self, query_tokens, query_weights, top_k, return_text, deleted_ids, num_workers
    ):
        """
        Search using disk mode (uses linear scan, not inverted index).

        Note: Inverted index search is only available in memory mode.
        Disk mode always uses linear scan for simplicity and correctness.
        """
        # Disk mode doesn't use inverted index - fall back to full scan
        return self._search_with_full_scan_disk(
            query_tokens, query_weights, top_k, return_text, deleted_ids, num_workers
        )

    def _search_with_full_scan_memory(
        self, query_tokens, query_weights, top_k, return_text, deleted_ids, num_workers
    ):
        """Search using full scan (brute-force) in memory mode (slow, ~55ms for 26K docs)."""
        if num_workers == 1 or len(self.index._cache) <= 1:
            heap = []
            idx = 0
            for docs in self.index._cache.values():
                for doc in docs:
                    if doc["doc_id"] in deleted_ids:
                        continue
                    score = compute_splade_score(
                        doc_tokens=doc["token_ids"],
                        doc_weights=doc["weights"],
                        query_tokens=query_tokens,
                        query_weights=query_weights,
                    )
                    if score > 0:
                        result = SearchResult(
                            doc_id=doc["doc_id"],
                            score=score,
                            metadata=doc["metadata"],
                            text=doc.get("text") if return_text else None,
                        )
                        if len(heap) < top_k:
                            heapq.heappush(heap, (score, idx, result))
                        elif score > heap[0][0]:
                            heapq.heapreplace(heap, (score, idx, result))
                        idx += 1
            return [r for _, _, r in sorted(heap, key=lambda x: x[0], reverse=True)]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    _search_memory_shard_worker,
                    (docs, query_tokens, query_weights, top_k, return_text, deleted_ids),
                ): shard_path
                for shard_path, docs in self.index._cache.items()
            }
            results = []
            for future in as_completed(futures):
                results.extend(future.result())

        return heapq.nlargest(top_k, results, key=lambda r: r.score)

    def _search_with_full_scan_disk(
        self, query_tokens, query_weights, top_k, return_text, deleted_ids, num_workers
    ):
        """Search using full scan (brute-force) in disk mode (streaming from disk)."""
        shard_paths = self.index.iter_shards()

        if num_workers == 1 or len(shard_paths) <= 1:
            heap = []
            idx = 0
            for shard_path in shard_paths:
                for doc in self.index.iter_docs(shard_path, load_text=return_text):
                    if doc["doc_id"] in deleted_ids:
                        continue
                    score = compute_splade_score(
                        doc_tokens=doc["token_ids"],
                        doc_weights=doc["weights"],
                        query_tokens=query_tokens,
                        query_weights=query_weights,
                    )
                    if score > 0:
                        result = SearchResult(
                            doc_id=doc["doc_id"],
                            score=score,
                            metadata=doc["metadata"],
                            text=doc.get("text") if return_text else None,
                        )
                        if len(heap) < top_k:
                            heapq.heappush(heap, (score, idx, result))
                        elif score > heap[0][0]:
                            heapq.heapreplace(heap, (score, idx, result))
                        idx += 1
            return [r for _, _, r in sorted(heap, key=lambda x: x[0], reverse=True)]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    _search_shard_worker,
                    (
                        shard_path,
                        query_tokens,
                        query_weights,
                        top_k,
                        return_text,
                        deleted_ids,
                        False,
                    ),
                ): shard_path
                for shard_path in shard_paths
            }
            results = []
            for future in as_completed(futures):
                results.extend(future.result())

        return heapq.nlargest(top_k, results, key=lambda r: r.score)

    @staticmethod
    def encode_query(query: str, model) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode a text query into SPLADE token IDs and weights.

        Useful for:
        - Profiling (separate encoding time from search time)
        - Batch encoding multiple queries
        - Caching encoded queries

        Args:
            query: Query text to encode
            model: SentenceTransformer model with SPLADE tokenizer

        Returns:
            Tuple of (token_ids, weights) as numpy arrays

        Example:
            >>> tokens, weights = IndexReader.encode_query("machine learning", model)
            >>> results = reader.search(query_tokens=tokens, query_weights=weights)
        """
        from splade_easy.utils import extract_splade_vectors

        encoding = model.encode(query, show_progress_bar=False)
        return extract_splade_vectors(encoding)

    def search_batch(
        self,
        queries: list[str],
        model,
        *,
        top_k: int = 10,
        return_text: bool = False,
        num_workers: Optional[int] = None,
    ) -> list[list[SearchResult]]:
        """
        Search the index for multiple queries in batch.

        More efficient than calling search() in a loop when you need to
        encode multiple queries at once.

        Args:
            queries: List of query texts
            model: SentenceTransformer model for encoding
            top_k: Number of results per query (default: 10)
            return_text: Whether to load full document text (default: False)
            num_workers: Number of parallel workers (None = use reader default)

        Returns:
            List of result lists, one per query

        Example:
            >>> queries = ["machine learning", "deep learning", "AI"]
            >>> batch_results = reader.search_batch(queries, model, top_k=5)
            >>> for query, results in zip(queries, batch_results):
            ...     print(f"{query}: {len(results)} results")
        """
        encoded = [self.encode_query(q, model) for q in queries]

        return [
            self.search(
                query_tokens=tokens,
                query_weights=weights,
                top_k=top_k,
                return_text=return_text,
                num_workers=num_workers,
            )
            for tokens, weights in encoded
        ]

    def load_texts(
        self,
        results: list[SearchResult],
        *,
        num_workers: Optional[int] = None,
    ) -> list[SearchResult]:
        """
        Load full document text for search results.

        Useful when you want to separate scoring from text loading for
        profiling or when you only need text for top results.

        Args:
            results: Search results to load text for
            num_workers: Number of parallel workers (None = use reader default)

        Returns:
            Results with text field populated

        Example:
            >>> # Score only, then load text for top 3
            >>> results = reader.search(query, model, top_k=100, return_text=False)
            >>> top_results = results[:3]
            >>> top_with_text = reader.load_texts(top_results)
        """
        return self._attach_texts(results, return_text=True)

    def get(self, doc_id: str) -> Optional[dict]:
        if self._is_deleted(doc_id):
            return None
        return self.index.get(doc_id)

    def get_batch(self, doc_ids: list[str]) -> list[Optional[dict]]:
        return [self.get(doc_id) for doc_id in doc_ids]

    def __del__(self):
        if hasattr(self, "_process_pool") and self._process_pool:
            self._process_pool.shutdown(wait=False)
        if hasattr(self, "_thread_pool") and self._thread_pool:
            self._thread_pool.shutdown(wait=False)

    def close(self):
        if self._process_pool:
            self._process_pool.shutdown(wait=True)
            self._process_pool = None
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None
