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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import numba
from numba.core import types
from numba.typed import Dict

from .index import Index
from .inverted_index import InvertedIndexReader
from .scoring import compute_splade_score, ensure_sorted_splade_vector
from .shard import ShardReader
from .utils import extract_model_id, extract_splade_vectors
from .schemas import SearchResult

logger = logging.getLogger(__name__)

posting_dtype = np.dtype([("doc_idx", "<i4"), ("weight", "<f4")])
posting_record_type = numba.from_dtype(posting_dtype)


@numba.njit(fastmath=True, cache=True, parallel=False)
def _accumulate_scores(
    query_tokens: np.ndarray,
    query_weights: np.ndarray,
    inv_indices: Dict,
    deleted_doc_indices: np.ndarray,
    num_docs: int,
) -> np.ndarray:
    doc_scores = np.zeros(num_docs, dtype=np.float32)
    
    # Accumulate scores using direct doc_idx (no dictionary lookup!)
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

        reader = ShardReader(str(shard_path))

        if use_inv_idx:
            # Use inverted index for efficient search
            inv_path = shard_path.with_suffix(".inv")
            if inv_path.exists():
                inv_reader = InvertedIndexReader(str(inv_path), shard_path)
                results = inv_reader.search(
                    query_tokens=query_tokens,
                    query_weights=query_weights,
                    top_k=top_k,
                    deleted_ids=deleted_ids,
                    doc_scorer=compute_splade_score,
                    shard_reader=reader,
                )

                # Attach text if requested
                if return_text:
                    for result in results:
                        pos = reader._find_offset_by_id(result.doc_id)
                        if pos is not None:
                            text_mv = reader.read_text_at(pos)
                            if text_mv is not None:
                                result.text = text_mv.tobytes().decode("utf-8")

                inv_reader.close()
                return results

        # Fallback to linear scan
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
    def __init__(self, index: Index, num_workers: int = 4, memory_mode: bool = None):
        self.index = index
        self.num_workers = num_workers
        self.memory_mode = memory_mode if memory_mode is not None else index.memory
        self._deleted_ids_cache: Optional[Set[str]] = None
        self._process_pool: Optional[ProcessPoolExecutor] = None
        self._thread_pool: Optional[ThreadPoolExecutor] = None

        # Cache numba data structures for fast search
        self._numba_postings_cache = None
        self._doc_lookup_cache = None  # Cache doc_id -> doc mapping

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

    def load(self):
        self.index.load()
        self._deleted_ids_cache = self._load_deleted_ids_cached()
        return self

    def _load_deleted_ids_cached(self) -> Set[str]:
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

    def _build_numba_cache(self):
        """Build numba-optimized data structures once for reuse across queries.
        
        Converts on-disk (doc_offset, weight) format to (doc_idx, weight) format
        for fast query-time lookups without dictionary overhead.
        """
        all_postings = Dict.empty(
            key_type=types.int64,
            value_type=types.Tuple((numba.types.Array(posting_record_type, 1, 'C'), types.int64[:])),
        )
        
        # On-disk format uses doc_offset
        disk_dtype = np.dtype([("doc_offset", "<u4"), ("weight", "<f4")])
        
        for shard_path, inv_reader in self.index._inv_index_cache.items():
            shard_path_str = str(shard_path)
            
            for token_id, (offset, num_postings) in inv_reader._token_map.items():
                # Read from disk in (doc_offset, weight) format
                disk_postings = np.frombuffer(
                    inv_reader._mmap,
                    dtype=disk_dtype,
                    count=num_postings,
                    offset=offset,
                )
                
                # Convert to (doc_idx, weight) format
                converted_postings = np.empty(num_postings, dtype=posting_dtype)
                valid_count = 0
                
                for i in range(num_postings):
                    doc_offset = disk_postings[i]["doc_offset"]
                    weight = disk_postings[i]["weight"]
                    
                    # Map doc_offset -> doc_id -> doc_idx
                    doc_id = self.index._offset_to_doc_id.get((shard_path_str, doc_offset))
                    if doc_id is not None:
                        doc_idx = self.index._doc_id_to_idx.get(doc_id, -1)
                        if doc_idx >= 0:
                            converted_postings[valid_count]["doc_idx"] = doc_idx
                            converted_postings[valid_count]["weight"] = weight
                            valid_count += 1
                
                # Only store valid postings
                if valid_count > 0:
                    converted_postings = converted_postings[:valid_count]
                    all_postings[token_id] = (converted_postings, np.array([valid_count]))

        self._numba_postings_cache = all_postings
        self._numba_offset_cache = None  # No longer needed!
        
        # Also build doc_lookup cache once
        self._doc_lookup_cache = {
            doc["doc_id"]: doc 
            for docs in self.index._cache.values() 
            for doc in docs
        }

    def _attach_texts(self, results: List[SearchResult], return_text: bool) -> List[SearchResult]:
        if not return_text or not results:
            return results

        needed = {r.doc_id for r in results if r.text is None}
        if not needed:
            return results

        # Build per-shard fetch lists
        per_shard: Dict[Path, List[str]] = {}
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
        top_k: int = 10,
        return_text: bool = False,
        num_workers: Optional[int] = None,
        use_inverted_index: bool = True,
    ) -> List[SearchResult]:
        if num_workers is None:
            num_workers = self.num_workers

        # Use inverted index if available and requested
        use_inv_idx = use_inverted_index and self._inverted_index_available

        if use_inv_idx:
            logger.debug(
                f"Using inverted index for search (available={self._inverted_index_available})"
            )

        if query is not None:
            if model is None:
                raise ValueError("Model required for text search")
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

        query_tokens, query_weights = ensure_sorted_splade_vector(
            query_tokens, query_weights, deduplicate=True
        )
        deleted_ids = self._deleted_ids_cache or self._load_deleted_ids_cached()

        # Memory mode with inverted index
        if self.memory_mode and self.index._cache and use_inv_idx:
            # In memory mode, we can use cached inverted indices
            return self._search_memory_optimized(
                query_tokens, query_weights, top_k, return_text, deleted_ids, num_workers
            )

        # Disk mode with inverted index
        if use_inv_idx:
            return self._search_disk_inverted(
                query_tokens, query_weights, top_k, return_text, deleted_ids, num_workers
            )

        # Fallback to linear scan (original behavior)
        if self.memory_mode and self.index._cache:
            return self._search_memory_linear(
                query_tokens, query_weights, top_k, return_text, deleted_ids, num_workers
            )

        return self._search_disk_linear(
            query_tokens, query_weights, top_k, return_text, deleted_ids, num_workers
        )

    def _search_memory_optimized(
        self,
        query_tokens: np.ndarray,
        query_weights: np.ndarray,
        top_k: int,
        return_text: bool,
        deleted_ids: Set[str],
        num_workers: int,
    ) -> List[SearchResult]:
        if not self._inverted_index_available:
            logger.debug("Inverted index not available, falling back to linear scan")
            return self._search_memory_linear(
                query_tokens, query_weights, top_k, return_text, deleted_ids, num_workers
            )

        # Build cache once on first query
        if self._numba_postings_cache is None:
            self._build_numba_cache()

        # Use cached structures (postings now contain doc_idx directly!)
        all_postings = self._numba_postings_cache

        deleted_doc_indices = np.array([self.index._doc_id_to_idx[did] for did in deleted_ids if did in self.index._doc_id_to_idx], dtype=np.int64)
        
        # Run Numba-optimized scoring (no dictionary lookup needed!)
        doc_scores = _accumulate_scores(
            query_tokens,
            query_weights,
            all_postings,
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
            if (doc := self._doc_lookup_cache.get(self.index._idx_to_doc_id[doc_idx])) is not None
        ]

        return self._attach_texts(results, return_text)
    
    def _search_disk_inverted(
        self, query_tokens, query_weights, top_k, return_text, deleted_ids, num_workers
    ):
        """Disk mode with inverted index"""
        shard_paths = list(self.index.iter_shards())

        if num_workers == 1 or len(shard_paths) <= 1:
            results = []
            for shard_path in shard_paths:
                inv_path = shard_path.with_suffix(".inv")
                if inv_path.exists():
                    inv_reader = InvertedIndexReader(str(inv_path), shard_path)
                    shard_results = inv_reader.search(
                        query_tokens=query_tokens,
                        query_weights=query_weights,
                        top_k=top_k,
                        deleted_ids=deleted_ids,
                        doc_scorer=compute_splade_score,
                        shard_reader=ShardReader(str(shard_path)),
                    )
                    results.extend(shard_results)
                    inv_reader.close()

            # Deduplicate and merge results from all shards
            unique_results = {r.doc_id: r for r in results}
            top_results = heapq.nlargest(top_k, unique_results.values(), key=lambda r: r.score)
            return self._attach_texts(top_results, return_text)

        # Parallel search across shards
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    _search_shard_worker,
                    (
                        shard_path,
                        query_tokens,
                        query_weights,
                        top_k,
                        False,  # Don't return text from workers
                        deleted_ids,
                        True,  # Use inverted index
                    ),
                ): shard_path
                for shard_path in shard_paths
            }

            results = []
            for future in as_completed(futures):
                results.extend(future.result())

        # Deduplicate and get top-k
        unique_results = {r.doc_id: r for r in results}
        top_results = heapq.nlargest(top_k, unique_results.values(), key=lambda r: r.score)
        return self._attach_texts(top_results, return_text)

    def _search_memory_linear(
        self, query_tokens, query_weights, top_k, return_text, deleted_ids, num_workers
    ):
        """Original linear scan memory mode"""
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

        # Multi-threaded
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

    def _search_disk_linear(
        self, query_tokens, query_weights, top_k, return_text, deleted_ids, num_workers
    ):
        """Original linear scan disk mode"""
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

        # Parallel disk search
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
                        False,  # Don't use inverted index
                    ),
                ): shard_path
                for shard_path in shard_paths
            }
            results = []
            for future in as_completed(futures):
                results.extend(future.result())

        return heapq.nlargest(top_k, results, key=lambda r: r.score)

    def search_text(self, query: str, model, top_k=10, return_text=False, num_workers=None):
        return self.search(
            query=query, model=model, top_k=top_k, return_text=return_text, num_workers=num_workers
        )

    def search_vectors(
        self,
        query_tokens: np.ndarray,
        query_weights: np.ndarray,
        top_k=10,
        return_text=False,
        num_workers=None,
    ):
        return self.search(
            query_tokens=query_tokens,
            query_weights=query_weights,
            top_k=top_k,
            return_text=return_text,
            num_workers=num_workers,
        )

    def _get_candidates_from_memory_inv(self, query_tokens: np.ndarray) -> Set[str]:
        """Get candidate doc_ids using inverted indices in memory mode"""
        candidates = set()
        for shard_path, docs in self.index._cache.items():
            inv_path = shard_path.with_suffix(".inv")
            if inv_path.exists():
                inv_reader = InvertedIndexReader(str(inv_path), shard_path)
                offsets = inv_reader.get_candidate_offsets(query_tokens)

                # Convert offsets to doc_ids using position index
                for did, (sp, off, _) in self.index._pos_index.items():
                    if sp == shard_path and off in offsets:
                        candidates.add(did)

                inv_reader.close()
        return candidates

    def _score_candidates(
        self, candidates: Set[str], query_tokens, query_weights, top_k, return_text, deleted_ids
    ):
        """Score only candidate documents"""
        results = []
        idx = 0

        for docs in self.index._cache.values():
            for doc in docs:
                if doc["doc_id"] not in candidates or doc["doc_id"] in deleted_ids:
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
                    results.append((score, idx, result))
                    idx += 1

        top_results = [r for _, _, r in heapq.nlargest(top_k, results, key=lambda x: x[0])]
        return self._attach_texts(top_results, return_text)

    def get(self, doc_id: str) -> Optional[dict]:
        if self._is_deleted(doc_id):
            return None
        return self.index.get(doc_id)

    def get_batch(self, doc_ids: List[str]) -> List[Optional[dict]]:
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
