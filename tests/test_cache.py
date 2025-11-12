# tests/test_cache.py
"""Tests for cache management and invalidation."""

import numpy as np

from splade_easy.cache import NumbaCache
from splade_easy.index import Index


class TestNumbaCache:
    """Test NumbaCache building and validation."""

    def test_cache_builds_postings(self, tmp_path):
        """Test that NumbaCache builds postings correctly."""
        index_dir = tmp_path / "test_index"
        index = Index(str(index_dir))

        with index.writer() as writer:
            writer.insert(
                doc_id="doc1",
                text="test document",
                token_ids=np.array([1, 2, 3], dtype=np.int64),
                weights=np.array([0.5, 0.8, 0.3], dtype=np.float32),
            )

        # Load in memory mode to populate cache
        index.load()

        cache = NumbaCache(index)
        cache.build()

        # Verify cache is built
        assert cache._postings_cache is not None
        assert cache._doc_lookup_cache is not None
        assert cache._commit_id == index.manifest.commit_id

    def test_cache_staleness_detection(self, tmp_path):
        """Test that cache can detect staleness and rebuild manually."""
        index_dir = tmp_path / "test_index"
        index = Index(str(index_dir))

        with index.writer() as writer:
            writer.insert(
                doc_id="doc1",
                text="test",
                token_ids=np.array([1], dtype=np.int64),
                weights=np.array([1.0], dtype=np.float32),
            )

        index.load()
        cache = NumbaCache(index)

        # Cache not yet built - should be stale
        assert cache.is_stale() is True

        # Build cache
        cache.build()
        assert cache.is_stale() is False
        assert cache._commit_id == index.manifest.commit_id

        # Modify index (changes commit_id)
        with index.writer() as writer:
            writer.insert(
                doc_id="doc2",
                text="test2",
                token_ids=np.array([2], dtype=np.int64),
                weights=np.array([1.0], dtype=np.float32),
            )
        index.refresh()

        # Cache should now be stale
        assert cache.is_stale() is True

        # ensure_valid rebuilds
        rebuilt = cache.ensure_valid()
        assert rebuilt is True
        assert cache.is_stale() is False
        assert cache._commit_id == index.manifest.commit_id

        # Second ensure_valid should not rebuild
        rebuilt = cache.ensure_valid()
        assert rebuilt is False


class TestCacheInvalidationWithReader:
    """Integration tests for cache invalidation with IndexReader."""

    def test_reader_cache_invalidates_on_document_addition(self, tmp_path):
        """Test that reader cache is invalidated after adding documents."""
        index_dir = tmp_path / "test_index"
        index = Index(str(index_dir))

        # Create initial index
        with index.writer() as writer:
            for i in range(5):
                writer.insert(
                    doc_id=f"doc{i}",
                    text=f"test document {i}",
                    token_ids=np.array([1, 2, i], dtype=np.int64),
                    weights=np.array([1.0, 0.5, 0.3], dtype=np.float32),
                )

        # Create reader
        reader = index.reader(memory=True)

        # Search (builds cache)
        reader.search(
            query_tokens=np.array([1, 2], dtype=np.int64),
            query_weights=np.array([1.0, 0.5], dtype=np.float32),
            top_k=5,
        )

        # Cache should be populated
        old_cache = reader._numba_cache
        assert old_cache is not None
        old_cache_commit_id = old_cache._commit_id

        old_commit_id = index.manifest.commit_id

        # Add more documents (triggers shard finalization and commit bump)
        with index.writer(shard_size_mb=32) as writer:
            for i in range(5, 10):
                writer.insert(
                    doc_id=f"doc{i}",
                    text=f"test document {i}",
                    token_ids=np.array([1, 2, i], dtype=np.int64),
                    weights=np.array([1.0, 0.5, 0.3], dtype=np.float32),
                )

        # Refresh to pick up new manifest
        index.refresh()
        new_commit_id = index.manifest.commit_id

        # Commit ID should have changed
        assert new_commit_id != old_commit_id

        # Search again (should detect stale cache and rebuild via property access)
        results2 = reader.search(
            query_tokens=np.array([1, 2], dtype=np.int64),
            query_weights=np.array([1.0, 0.5], dtype=np.float32),
            top_k=5,
        )

        # Cache commit_id should have been updated (same object, rebuilt internally)
        assert reader._numba_cache._commit_id == new_commit_id
        assert reader._numba_cache._commit_id != old_cache_commit_id

        # Results should still be correct
        assert len(results2) > 0
        assert all(r.score > 0 for r in results2)

    def test_reader_cache_survives_when_no_changes(self, tmp_path):
        """Test that cache is reused when index hasn't changed."""
        index_dir = tmp_path / "test_index"
        index = Index(str(index_dir))

        with index.writer() as writer:
            for i in range(5):
                writer.insert(
                    doc_id=f"doc{i}",
                    text=f"test {i}",
                    token_ids=np.array([i], dtype=np.int64),
                    weights=np.array([1.0], dtype=np.float32),
                )

        reader = index.reader(memory=True)

        # Search twice
        reader.search(
            query_tokens=np.array([1], dtype=np.int64),
            query_weights=np.array([1.0], dtype=np.float32),
            top_k=5,
        )

        commit_id1 = reader._numba_cache._commit_id

        reader.search(
            query_tokens=np.array([2], dtype=np.int64),
            query_weights=np.array([1.0], dtype=np.float32),
            top_k=5,
        )

        commit_id2 = reader._numba_cache._commit_id

        # Should be same cache (same commit_id)
        assert commit_id1 == commit_id2
        assert commit_id1 == index.manifest.commit_id
