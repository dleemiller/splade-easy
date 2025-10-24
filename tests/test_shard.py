# tests/test_shard.py

import tempfile
from pathlib import Path

import numpy as np
import pytest

from splade_easy.shard import ShardReader, ShardWriter


class TestShard:
    """Test shard reading and writing."""

    @pytest.fixture
    def temp_shard(self):
        """Create temporary shard file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".fb") as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)

    def test_write_and_read_single_doc(self, temp_shard):
        """Test writing and reading a single document."""
        # Write
        writer = ShardWriter(temp_shard)
        writer.append(
            doc_id="test_1",
            text="This is a test document",
            metadata={"source": "test", "category": "demo"},
            token_ids=np.array([1, 5, 10], dtype=np.uint32),
            weights=np.array([0.8, 0.5, 0.3], dtype=np.float32),
        )
        writer.close()

        # Read
        reader = ShardReader(temp_shard)
        docs = list(reader.scan())

        assert len(docs) == 1
        doc = docs[0]

        assert doc["doc_id"] == "test_1"
        assert doc["text"] == "This is a test document"
        assert doc["metadata"] == {"source": "test", "category": "demo"}
        np.testing.assert_array_equal(doc["token_ids"], np.array([1, 5, 10]))
        np.testing.assert_array_almost_equal(doc["weights"], np.array([0.8, 0.5, 0.3]))

    def test_write_and_read_multiple_docs(self, temp_shard):
        """Test writing and reading multiple documents."""
        writer = ShardWriter(temp_shard)

        for i in range(10):
            writer.append(
                doc_id=f"doc_{i}",
                text=f"Document number {i}",
                metadata={"index": str(i)},
                token_ids=np.array([i, i + 1, i + 2], dtype=np.uint32),
                weights=np.array([0.1 * i, 0.2 * i, 0.3 * i], dtype=np.float32),
            )

        writer.close()

        reader = ShardReader(temp_shard)
        docs = list(reader.scan())

        assert len(docs) == 10

        for i, doc in enumerate(docs):
            assert doc["doc_id"] == f"doc_{i}"
            assert doc["text"] == f"Document number {i}"
            assert doc["metadata"]["index"] == str(i)

    def test_skip_text_loading(self, temp_shard):
        """Test that text can be skipped during reading."""
        writer = ShardWriter(temp_shard)
        writer.append(
            doc_id="test",
            text="A" * 10000,  # Large text
            metadata={},
            token_ids=np.array([1], dtype=np.uint32),
            weights=np.array([1.0], dtype=np.float32),
        )
        writer.close()

        reader = ShardReader(temp_shard)
        docs = list(reader.scan(load_text=False))

        assert len(docs) == 1
        assert docs[0]["text"] is None
        assert docs[0]["doc_id"] == "test"

    def test_empty_metadata(self, temp_shard):
        """Test document with no metadata."""
        writer = ShardWriter(temp_shard)
        writer.append(
            doc_id="minimal",
            text="Minimal doc",
            metadata={},
            token_ids=np.array([1], dtype=np.uint32),
            weights=np.array([1.0], dtype=np.float32),
        )
        writer.close()

        reader = ShardReader(temp_shard)
        docs = list(reader.scan())

        assert docs[0]["metadata"] == {}

    def test_large_sparse_vector(self, temp_shard):
        """Test with realistic SPLADE vector size."""
        writer = ShardWriter(temp_shard)

        # Realistic SPLADE: ~200 non-zero dimensions
        n_tokens = 200
        token_ids = np.array(
            sorted(np.random.choice(30000, n_tokens, replace=False)), dtype=np.uint32
        )
        weights = np.random.rand(n_tokens).astype(np.float32)

        writer.append(
            doc_id="large_vec",
            text="Document with large sparse vector",
            metadata={"size": str(n_tokens)},
            token_ids=token_ids,
            weights=weights,
        )
        writer.close()

        reader = ShardReader(temp_shard)
        docs = list(reader.scan())

        assert len(docs) == 1
        np.testing.assert_array_equal(docs[0]["token_ids"], token_ids)
        np.testing.assert_array_almost_equal(docs[0]["weights"], weights)

    def test_shard_size_tracking(self, temp_shard):
        """Test that shard size is tracked correctly."""
        writer = ShardWriter(temp_shard)

        initial_size = writer.size()
        assert initial_size == 0

        writer.append(
            doc_id="test",
            text="test",
            metadata={},
            token_ids=np.array([1], dtype=np.uint32),
            weights=np.array([1.0], dtype=np.float32),
        )

        after_size = writer.size()
        assert after_size > 0

        writer.close()

        # File size should match tracked size
        actual_size = Path(temp_shard).stat().st_size
        assert actual_size == after_size
