import tempfile
from pathlib import Path

import numpy as np
import pytest

from splade_easy.shard import ShardReader, ShardWriter


class TestShardWriter:
    """Test ShardWriter functionality including batched writes."""

    def test_basic_write_and_read(self):
        """Test basic write and read cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "test.fb"

            # Write documents
            writer = ShardWriter(str(shard_path))
            writer.append(
                doc_id="doc1",
                text="Test document",
                metadata={"key": "value"},
                token_ids=np.array([1, 2, 3], dtype=np.uint32),
                weights=np.array([0.5, 0.6, 0.7], dtype=np.float32),
            )
            writer.close()

            # Read back
            reader = ShardReader(str(shard_path))
            docs = list(reader.scan())

            assert len(docs) == 1
            assert docs[0]["doc_id"] == "doc1"
            assert docs[0]["text"] == "Test document"
            assert docs[0]["metadata"]["key"] == "value"
            np.testing.assert_array_equal(docs[0]["token_ids"], [1, 2, 3])
            np.testing.assert_array_almost_equal(docs[0]["weights"], [0.5, 0.6, 0.7])

    def test_batched_writes(self):
        """Test that batched writes work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "test.fb"

            # Write many documents with small batch size
            writer = ShardWriter(str(shard_path), write_batch_size=10)

            for i in range(50):
                writer.append(
                    doc_id=f"doc_{i}",
                    text=f"Document {i}",
                    metadata={"index": str(i)},
                    token_ids=np.array([i, i + 1], dtype=np.uint32),
                    weights=np.array([0.5, 0.5], dtype=np.float32),
                )

            writer.close()

            # Verify all documents readable
            reader = ShardReader(str(shard_path))
            docs = list(reader.scan())

            assert len(docs) == 50
            for i, doc in enumerate(docs):
                assert doc["doc_id"] == f"doc_{i}"
                assert doc["metadata"]["index"] == str(i)

    def test_large_batch_size(self):
        """Test with large batch size (fewer flushes)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "test.fb"

            # Large batch size means almost everything buffered
            writer = ShardWriter(str(shard_path), write_batch_size=1000)

            for i in range(100):
                writer.append(
                    doc_id=f"doc_{i}",
                    text=f"Document {i}",
                    metadata={},
                    token_ids=np.array([i], dtype=np.uint32),
                    weights=np.array([0.5], dtype=np.float32),
                )

            # Everything should be in buffer, not yet written
            assert len(writer.write_buffer) == 100

            writer.close()

            # Now verify written
            reader = ShardReader(str(shard_path))
            docs = list(reader.scan())
            assert len(docs) == 100

    def test_empty_shard(self):
        """Test reading empty shard."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "test.fb"

            # Create empty file
            writer = ShardWriter(str(shard_path))
            writer.close()

            # Should read as empty
            reader = ShardReader(str(shard_path))
            docs = list(reader.scan())
            assert len(docs) == 0

    def test_size_tracking(self):
        """Test that size is tracked correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "test.fb"

            writer = ShardWriter(str(shard_path))

            initial_size = writer.size()
            assert initial_size == 0

            # Add document
            writer.append(
                doc_id="doc1",
                text="Test",
                metadata={},
                token_ids=np.array([1], dtype=np.uint32),
                weights=np.array([0.5], dtype=np.float32),
            )

            # Size should increase
            assert writer.size() > initial_size

            size_before_close = writer.size()
            writer.close()

            # File size should match tracked size
            actual_size = shard_path.stat().st_size
            assert actual_size == size_before_close

    def test_scan_without_text(self):
        """Test scan with load_text=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "test.fb"

            writer = ShardWriter(str(shard_path))
            writer.append(
                doc_id="doc1",
                text="This is the text",
                metadata={},
                token_ids=np.array([1], dtype=np.uint32),
                weights=np.array([0.5], dtype=np.float32),
            )
            writer.close()

            # Read without text
            reader = ShardReader(str(shard_path))
            docs = list(reader.scan(load_text=False))

            assert len(docs) == 1
            assert docs[0]["doc_id"] == "doc1"
            assert docs[0]["text"] is None
            assert docs[0]["token_ids"] is not None

    def test_sparse_vectors(self):
        """Test with realistic sparse vectors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "test.fb"

            # Create sparse vector (many zero weights filtered out)
            writer = ShardWriter(str(shard_path))
            writer.append(
                doc_id="doc1",
                text="Sparse document",
                metadata={},
                token_ids=np.array([10, 50, 100, 500, 1000], dtype=np.uint32),
                weights=np.array([0.9, 0.7, 0.5, 0.3, 0.1], dtype=np.float32),
            )
            writer.close()

            reader = ShardReader(str(shard_path))
            docs = list(reader.scan())

            assert len(docs) == 1
            np.testing.assert_array_equal(docs[0]["token_ids"], [10, 50, 100, 500, 1000])
            np.testing.assert_array_almost_equal(docs[0]["weights"], [0.9, 0.7, 0.5, 0.3, 0.1])

    def test_metadata_types(self):
        """Test that metadata values are converted to strings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "test.fb"

            writer = ShardWriter(str(shard_path))
            writer.append(
                doc_id="doc1",
                text="Test",
                metadata={
                    "string": "value",
                    "int": 42,
                    "float": 3.14,
                    "bool": True,
                },
                token_ids=np.array([1], dtype=np.uint32),
                weights=np.array([0.5], dtype=np.float32),
            )
            writer.close()

            reader = ShardReader(str(shard_path))
            docs = list(reader.scan())

            # All should be strings
            assert docs[0]["metadata"]["string"] == "value"
            assert docs[0]["metadata"]["int"] == "42"
            assert docs[0]["metadata"]["float"] == "3.14"
            assert docs[0]["metadata"]["bool"] == "True"

    def test_invalid_write_batch_size_zero(self):
        """write_batch_size=0 should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "test.fb"
            with pytest.raises(ValueError, match="write_batch_size must be positive"):
                ShardWriter(str(shard_path), write_batch_size=0)

    def test_invalid_write_batch_size_negative(self):
        """Negative write_batch_size should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "test.fb"
            with pytest.raises(ValueError, match="write_batch_size must be positive"):
                ShardWriter(str(shard_path), write_batch_size=-1)
