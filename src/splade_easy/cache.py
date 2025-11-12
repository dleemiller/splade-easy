# src/splade_easy/cache.py
"""
Cache management for fast SPLADE search operations.

This module provides caching mechanisms that optimize search performance:
- NumbaCache: JIT-compiled data structures for inverted index search
- Cache validation: Ensures caches are invalidated when index changes
"""

import logging

import numba
import numpy as np
from numba.core import types
from numba.typed import Dict as TypedDict

logger = logging.getLogger(__name__)


# Numba types for posting lists
posting_dtype = np.dtype([("doc_idx", "<i4"), ("weight", "<f4")])
posting_record_type = numba.from_dtype(posting_dtype)


class NumbaCache:
    """
    Numba-optimized cache for fast inverted index search with automatic validation.

    This class handles:
    - Conversion from on-disk (doc_offset, weight) to in-memory (doc_idx, weight)
    - Automatic validation and rebuild when index changes
    - Lazy loading via properties

    The cache automatically detects index changes (via commit_id) and rebuilds
    as needed, making it safe to use across index modifications.

    Example:
        >>> cache = NumbaCache(index)
        >>> postings = cache.postings  # Auto-validates and lazy-loads
        >>> doc_lookup = cache.doc_lookup  # Auto-validates and lazy-loads
    """

    def __init__(self, index):
        """
        Initialize NumbaCache for an index.

        Args:
            index: Index instance to build cache for (must be in memory mode)
        """
        self.index = index
        self._postings_cache = None
        self._doc_lookup_cache = None
        self._commit_id = None  # Track which index version this cache is for

    def is_stale(self) -> bool:
        """
        Check if cache is stale (index has changed since cache was built).

        Returns:
            True if cache needs rebuilding, False if still valid
        """
        if self._commit_id is None:
            return True  # Not yet built
        return self._commit_id != self.index.manifest.commit_id

    def ensure_valid(self) -> bool:
        """
        Manually validate and rebuild cache if index has changed.

        Call this after index.refresh() if you want to ensure the cache
        is up-to-date with the current index state.

        Returns:
            True if cache was rebuilt, False if already valid

        Example:
            >>> index.refresh()  # After external modification
            >>> cache.ensure_valid()  # Explicitly rebuild if needed
        """
        if not self.is_stale():
            return False

        if self._commit_id is not None:
            logger.info(
                f"Index changed (commit {self._commit_id[:8]}... → "
                f"{self.index.manifest.commit_id[:8]}...), rebuilding NumbaCache"
            )
        self.build()
        return True

    def build(self) -> None:
        """
        Build all caches for current index state.

        This is called automatically by ensure_valid() when needed.
        Rebuilds ~1000-3000 postings in ~20-30ms for typical indices.
        """
        logger.debug("Building NumbaCache...")
        self._build_postings_cache()
        self._build_doc_lookup_cache()
        self._commit_id = self.index.manifest.commit_id
        logger.debug(f"NumbaCache built for commit_id={self._commit_id[:8]}...")

    def _build_postings_cache(self) -> None:
        """
        Convert inverted index postings to numba-compatible format.

        Transforms on-disk (doc_offset, weight) to in-memory (doc_idx, weight)
        format, enabling O(1) array indexing instead of O(log n) dict lookups.
        """
        all_postings = TypedDict.empty(
            key_type=types.int64,
            value_type=types.Tuple(
                (numba.types.Array(posting_record_type, 1, "C"), types.int64[:])
            ),
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

                    # Map doc_offset → doc_id → doc_idx
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

        self._postings_cache = all_postings

    def _build_doc_lookup_cache(self) -> None:
        """
        Build doc_id to document mapping for fast result construction.

        Creates a flat dictionary for O(1) lookup of documents by ID,
        avoiding nested iteration through shard caches.
        """
        self._doc_lookup_cache = {
            doc["doc_id"]: doc for docs in self.index._cache.values() for doc in docs
        }

    @property
    def postings(self):
        """
        Get the numba-optimized postings dictionary.

        Lazy-loads on first access. Returns a numba TypedDict mapping
        token_id → (postings_array, count).

        Note: Does not auto-validate. Call ensure_valid() after index.refresh()
        if you need to guarantee freshness.
        """
        if self._postings_cache is None:
            self.build()
        return self._postings_cache

    @property
    def doc_lookup(self):
        """
        Get the doc_id to document mapping.

        Lazy-loads on first access. Returns a dict mapping doc_id → document.

        Note: Does not auto-validate. Call ensure_valid() after index.refresh()
        if you need to guarantee freshness.
        """
        if self._doc_lookup_cache is None:
            self.build()
        return self._doc_lookup_cache
