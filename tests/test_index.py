# tests/test_index.py

import tempfile
from pathlib import Path

import numpy as np
import pytest

from splade_easy import Document, SpladeIndex


class TestSpladeIndex:
    """Test SpladeIndex writing and maintenance."""

    @pytest.fixture
    def temp_index_dir(self):
        """Create temporary index directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_doc(self):
        """Create a sample document."""
        return Document(
            doc_id="test_1",
            text="This is a test document about machine learning.",
            metadata={"source": "test", "category": "AI"},
            token_ids=np.array([1, 5, 10, 42], dtype=np.uint32),
            weights=np.array([0.8, 0.6, 0.4, 0.2], dtype=np.float32),
        )

    def test_create_new_index(self, temp_index_dir):
        """Test creating a new index."""
        index = SpladeIndex(temp_index_dir)

        assert len(index) == 0
        assert (Path(temp_index_dir) / "metadata.json").exists()

        stats = index.stats()
        assert stats["num_docs"] == 0
        assert stats["num_shards"] == 0

    def test_add_single_document(self, temp_index_dir, sample_doc):
        """Test adding a single document."""
        index = SpladeIndex(temp_index_dir)
        index.add(sample_doc)

        assert len(index) == 1

        stats = index.stats()
        assert stats["num_docs"] == 1

        # Verify document can be retrieved
        retriever = index.retriever(temp_index_dir)
        doc = retriever.get("test_1")
        assert doc is not None
        assert doc["doc_id"] == "test_1"
        assert doc["text"] == sample_doc.text

    def test_add_batch(self, temp_index_dir):
        """Test adding multiple documents."""
        index = SpladeIndex(temp_index_dir)

        docs = [
            Document(
                doc_id=f"doc_{i}",
                text=f"Document {i}",
                metadata={"index": str(i)},
                token_ids=np.array([i, i + 1], dtype=np.uint32),
                weights=np.array([0.5, 0.5], dtype=np.float32),
            )
            for i in range(10)
        ]

        index.add_batch(docs)

        assert len(index) == 10
        assert index.stats()["num_docs"] == 10

    def test_shard_rotation(self, temp_index_dir):
        """Test that shards rotate when reaching size limit."""
        # Small shard size to force rotation
        index = SpladeIndex(temp_index_dir, shard_size_mb=0.001)  # 1KB

        # Add documents until we get multiple shards
        for i in range(20):
            doc = Document(
                doc_id=f"doc_{i}",
                text=f"Document {i} " * 100,  # Make it larger
                metadata={"index": str(i)},
                token_ids=np.random.randint(0, 1000, 50, dtype=np.uint32),
                weights=np.random.rand(50).astype(np.float32),
            )
            index.add(doc)

        stats = index.stats()
        assert stats["num_shards"] > 1, "Should have created multiple shards"

    def test_delete_document(self, temp_index_dir, sample_doc):
        """Test soft-deleting a document."""
        index = SpladeIndex(temp_index_dir)
        index.add(sample_doc)

        assert len(index) == 1

        # Delete
        result = index.delete("test_1")
        assert result is True
        assert len(index) == 0

        # Verify it's in deleted_ids
        assert "test_1" in index.deleted_ids

        # Verify retriever respects deletion
        retriever = index.retriever(temp_index_dir)
        doc = retriever.get("test_1")
        assert doc is None

    def test_delete_nonexistent(self, temp_index_dir):
        """Test deleting a document that doesn't exist."""
        index = SpladeIndex(temp_index_dir)
        result = index.delete("nonexistent")
        assert result is False

    def test_compact(self, temp_index_dir):
        """Test compacting removes deleted documents."""
        index = SpladeIndex(temp_index_dir)

        # Add 10 documents
        for i in range(10):
            doc = Document(
                doc_id=f"doc_{i}",
                text=f"Document {i}",
                metadata={},
                token_ids=np.array([i], dtype=np.uint32),
                weights=np.array([0.5], dtype=np.float32),
            )
            index.add(doc)

        # Delete 5 documents
        for i in range(5):
            index.delete(f"doc_{i}")

        assert len(index) == 5
        assert len(index.deleted_ids) == 5

        # Compact
        index.compact()

        # Verify deleted_ids cleared
        assert len(index.deleted_ids) == 0
        assert len(index) == 5

        # Verify only 5 docs remain
        retriever = index.retriever(temp_index_dir)
        for i in range(5):
            doc = retriever.get(f"doc_{i}")
            assert doc is None  # First 5 deleted

        for i in range(5, 10):
            doc = retriever.get(f"doc_{i}")
            assert doc is not None  # Last 5 remain

    def test_stats(self, temp_index_dir):
        """Test stats reporting."""
        index = SpladeIndex(temp_index_dir, shard_size_mb=0.01)

        # Add some documents
        for i in range(5):
            doc = Document(
                doc_id=f"doc_{i}",
                text=f"Document {i}",
                metadata={},
                token_ids=np.array([i], dtype=np.uint32),
                weights=np.array([0.5], dtype=np.float32),
            )
            index.add(doc)

        stats = index.stats()

        assert stats["num_docs"] == 5
        assert stats["num_shards"] >= 1
        assert stats["deleted_docs"] == 0
        assert stats["total_size_mb"] > 0

        # Delete one
        index.delete("doc_0")
        stats = index.stats()

        assert stats["num_docs"] == 4
        assert stats["deleted_docs"] == 1

    def test_context_manager(self, temp_index_dir, sample_doc):
        """Test using index as context manager."""
        with SpladeIndex(temp_index_dir) as index:
            index.add(sample_doc)

        # Verify data persisted
        retriever = SpladeIndex.retriever(temp_index_dir)
        doc = retriever.get("test_1")
        assert doc is not None
