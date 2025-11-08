# tests/test_retriever.py

import tempfile

import numpy as np
import pytest

from splade_easy import Document, SpladeIndex


class TestSpladeRetriever:
    """Test SpladeRetriever search operations."""

    @pytest.fixture
    def populated_index(self):
        """Create an index with sample documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = SpladeIndex(tmpdir)

            # Add documents with overlapping tokens
            docs = [
                Document(
                    doc_id="ml_doc",
                    text="Machine learning and AI",
                    metadata={"topic": "AI"},
                    token_ids=np.array([1, 2, 3], dtype=np.uint32),
                    weights=np.array([0.9, 0.7, 0.5], dtype=np.float32),
                ),
                Document(
                    doc_id="dl_doc",
                    text="Deep learning neural networks",
                    metadata={"topic": "AI"},
                    token_ids=np.array([1, 4, 5], dtype=np.uint32),
                    weights=np.array([0.8, 0.6, 0.4], dtype=np.float32),
                ),
                Document(
                    doc_id="python_doc",
                    text="Python programming language",
                    metadata={"topic": "Programming"},
                    token_ids=np.array([10, 11, 12], dtype=np.uint32),
                    weights=np.array([0.7, 0.5, 0.3], dtype=np.float32),
                ),
            ]

            index.add_batch(docs)
            # Finalize shard to make docs available to retriever
            index._finalize_current_shard()
            yield tmpdir

    def test_basic_search_disk_mode(self, populated_index):
        """Test basic search in disk mode."""
        retriever = SpladeIndex.retriever(populated_index, mode="disk")

        # Query that matches first two docs (token 1)
        query_tokens = np.array([1, 2], dtype=np.uint32)
        query_weights = np.array([1.0, 0.5], dtype=np.float32)

        results = retriever.search(query_tokens, query_weights, top_k=5)

        assert len(results) > 0
        assert results[0].doc_id in ["ml_doc", "dl_doc"]
        assert results[0].score > 0

    def test_basic_search_memory_mode(self, populated_index):
        """Test basic search in memory mode."""
        retriever = SpladeIndex.retriever(populated_index, mode="memory")

        query_tokens = np.array([1, 2], dtype=np.uint32)
        query_weights = np.array([1.0, 0.5], dtype=np.float32)

        results = retriever.search(query_tokens, query_weights, top_k=5)

        assert len(results) > 0
        assert results[0].score > 0

    def test_search_no_matches(self, populated_index):
        """Test search with no matching tokens."""
        retriever = SpladeIndex.retriever(populated_index)

        # Query with tokens not in any document
        query_tokens = np.array([999, 1000], dtype=np.uint32)
        query_weights = np.array([1.0, 1.0], dtype=np.float32)

        results = retriever.search(query_tokens, query_weights, top_k=5)

        assert len(results) == 0

    def test_search_top_k_limit(self, populated_index):
        """Test that search respects top_k limit."""
        retriever = SpladeIndex.retriever(populated_index)

        # Query that matches all docs
        query_tokens = np.array([1, 4, 10], dtype=np.uint32)
        query_weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        results = retriever.search(query_tokens, query_weights, top_k=2)

        assert len(results) <= 2

    def test_search_return_text(self, populated_index):
        """Test search with return_text flag."""
        retriever = SpladeIndex.retriever(populated_index)

        query_tokens = np.array([1], dtype=np.uint32)
        query_weights = np.array([1.0], dtype=np.float32)

        # Without text
        results = retriever.search(query_tokens, query_weights, return_text=False)
        if len(results) > 0:
            assert results[0].text is None

        # With text
        results = retriever.search(query_tokens, query_weights, return_text=True)
        if len(results) > 0:
            assert results[0].text is not None

    def test_search_parallel(self, populated_index):
        """Test parallel search with multiple workers."""
        retriever = SpladeIndex.retriever(populated_index)

        query_tokens = np.array([1, 2], dtype=np.uint32)
        query_weights = np.array([1.0, 0.5], dtype=np.float32)

        # Search with multiple workers
        results = retriever.search(query_tokens, query_weights, top_k=5, num_workers=2)

        assert len(results) > 0

    def test_search_respects_deleted(self):
        """Test that search respects deleted documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = SpladeIndex(tmpdir)

            # Add documents
            doc1 = Document(
                doc_id="doc_1",
                text="Test document one",
                metadata={},
                token_ids=np.array([1, 2], dtype=np.uint32),
                weights=np.array([0.8, 0.6], dtype=np.float32),
            )
            doc2 = Document(
                doc_id="doc_2",
                text="Test document two",
                metadata={},
                token_ids=np.array([1, 3], dtype=np.uint32),
                weights=np.array([0.7, 0.5], dtype=np.float32),
            )

            index.add_batch([doc1, doc2])

            # Delete one
            index.delete("doc_1")

            # Search
            retriever = index.retriever(tmpdir)
            query_tokens = np.array([1], dtype=np.uint32)
            query_weights = np.array([1.0], dtype=np.float32)

            results = retriever.search(query_tokens, query_weights, top_k=10)

            # Should only find doc_2
            doc_ids = [r.doc_id for r in results]
            assert "doc_1" not in doc_ids
            assert "doc_2" in doc_ids

    def test_get_document(self, populated_index):
        """Test getting a single document."""
        retriever = SpladeIndex.retriever(populated_index)

        doc = retriever.get("ml_doc")

        assert doc is not None
        assert doc["doc_id"] == "ml_doc"
        assert doc["text"] == "Machine learning and AI"
        assert doc["metadata"]["topic"] == "AI"

    def test_get_nonexistent(self, populated_index):
        """Test getting a document that doesn't exist."""
        retriever = SpladeIndex.retriever(populated_index)

        doc = retriever.get("nonexistent")
        assert doc is None

    def test_get_batch(self, populated_index):
        """Test batch document retrieval."""
        retriever = SpladeIndex.retriever(populated_index)

        docs = retriever.get_batch(["ml_doc", "python_doc", "nonexistent"])

        assert len(docs) == 3
        assert docs[0]["doc_id"] == "ml_doc"
        assert docs[1]["doc_id"] == "python_doc"
        assert docs[2] is None

    def test_search_score_ordering(self, populated_index):
        """Test that results are ordered by score."""
        retriever = SpladeIndex.retriever(populated_index)

        # Query that matches multiple docs with different scores
        query_tokens = np.array([1, 2, 3], dtype=np.uint32)
        query_weights = np.array([1.0, 0.8, 0.6], dtype=np.float32)

        results = retriever.search(query_tokens, query_weights, top_k=5)

        # Verify descending order
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_search_with_duplicates_in_query(self, populated_index):
        """Test search handles duplicate tokens in query."""
        retriever = SpladeIndex.retriever(populated_index)

        # Query with duplicate tokens (should be deduplicated)
        query_tokens = np.array([1, 1, 2], dtype=np.uint32)
        query_weights = np.array([0.5, 0.8, 0.3], dtype=np.float32)

        results = retriever.search(query_tokens, query_weights, top_k=5)
        assert len(results) >= 0  # Should not crash

    def test_search_unsorted_query(self, populated_index):
        """Test search handles unsorted query vectors."""
        retriever = SpladeIndex.retriever(populated_index)

        # Unsorted query (should be sorted internally)
        query_tokens = np.array([5, 1, 3], dtype=np.uint32)
        query_weights = np.array([0.4, 0.9, 0.6], dtype=np.float32)

        results = retriever.search(query_tokens, query_weights, top_k=5)
        assert len(results) >= 0  # Should not crash

    def test_metadata_preserved(self, populated_index):
        """Test that metadata is preserved correctly."""
        retriever = SpladeIndex.retriever(populated_index)

        doc = retriever.get("ml_doc")
        assert doc["metadata"]["topic"] == "AI"

        doc = retriever.get("python_doc")
        assert doc["metadata"]["topic"] == "Programming"

    def test_empty_shard_handling(self):
        """Test handling of empty or missing shards."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = SpladeIndex(tmpdir)

            # Don't add any docs, but create retriever
            retriever = index.retriever(tmpdir)

            results = retriever.search(
                np.array([1], dtype=np.uint32), np.array([1.0], dtype=np.float32), top_k=10
            )

            assert len(results) == 0

    def test_parallel_search_with_equal_scores(self):
        """Regression test for Bug 3: parallel search fails when documents have equal scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = SpladeIndex(tmpdir)

            # Add documents with identical vectors (will have equal scores)
            docs = [
                Document(
                    doc_id=f"doc_{i}",
                    text=f"Document {i}",
                    metadata={"index": str(i)},
                    token_ids=np.array([1, 2, 3], dtype=np.uint32),  # Same tokens
                    weights=np.array([0.5, 0.5, 0.5], dtype=np.float32),  # Same weights
                )
                for i in range(10)
            ]

            index.add_batch(docs)
            index._finalize_current_shard()

            # Search with parallel workers - this should not crash
            retriever = index.retriever(tmpdir)
            query_tokens = np.array([1, 2, 3], dtype=np.uint32)
            query_weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)

            # This used to fail with: TypeError: '<' not supported between instances of 'SearchResult'
            results = retriever.search(query_tokens, query_weights, top_k=5, num_workers=2)

            # All documents match and have equal scores
            assert len(results) == 5  # top_k limit
            assert all(r.score > 0 for r in results)

            # Verify scores are equal (or very close due to floating point)
            scores = [r.score for r in results]
            assert all(abs(s - scores[0]) < 1e-6 for s in scores), "All scores should be equal"
