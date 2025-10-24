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
