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

    def test_reshard(self, temp_index_dir):
        """Test resharding with different target sizes."""
        index = SpladeIndex(temp_index_dir, shard_size_mb=0.01)

        # Add many documents to create multiple shards
        for i in range(50):
            doc = Document(
                doc_id=f"doc_{i}",
                text=f"Document {i} " * 50,
                metadata={"index": str(i)},
                token_ids=np.random.randint(0, 1000, 20, dtype=np.uint32),
                weights=np.random.rand(20).astype(np.float32),
            )
            index.add(doc)

        initial_shards = index.stats()["num_shards"]
        assert initial_shards > 1

        # Reshard to larger size
        stats = index.reshard(target_shard_size_mb=1)

        assert stats["docs_written"] == 50
        assert index.stats()["num_docs"] == 50

        # Verify all content-addressed shards exist
        assert len(index.metadata["shard_hashes"]) == index.metadata["num_shards"]
        for shard_hash in index.metadata["shard_hashes"]:
            shard_path = index.index_dir / f"{shard_hash}.fb"
            assert shard_path.exists()

    def test_empty_index_operations(self, temp_index_dir):
        """Test operations on empty index."""
        index = SpladeIndex(temp_index_dir)

        assert len(index) == 0

        # Search should return empty
        retriever = index.retriever(temp_index_dir)
        results = retriever.search(
            np.array([1], dtype=np.uint32), np.array([1.0], dtype=np.float32)
        )
        assert len(results) == 0

        # Get should return None
        assert retriever.get("nonexistent") is None

        # Delete should return False
        assert index.delete("nonexistent") is False

        # Compact should work
        index.compact()
        assert len(index) == 0

    def test_content_addressed_shards(self, temp_index_dir, sample_doc):
        """Test that shards are content-addressed (hash-based names)."""
        index = SpladeIndex(temp_index_dir)
        index.add(sample_doc)

        # Force rotation to finalize shard
        if index.current_writer:
            index._rotate_shard()

        # Check that shard_hashes is populated
        assert len(index.metadata["shard_hashes"]) > 0

        # Check that shard files exist with hash names
        for shard_hash in index.metadata["shard_hashes"]:
            shard_path = index.index_dir / f"{shard_hash}.fb"
            assert shard_path.exists()
            assert len(shard_hash) == 64  # SHA256 hex length

        # Ensure no temp shard files remain after rotation
        temp_files = list(Path(temp_index_dir).glob("_temp_shard_*.fb"))
        assert temp_files == []

    def test_add_batch_does_not_finalize_shard(self, temp_index_dir):
        """Regression: add_batch should not immediately finalize shard (Bug 1)."""
        index = SpladeIndex(temp_index_dir, shard_size_mb=32)

        # Add a batch of 100 docs
        docs = [
            Document(
                doc_id=f"doc_{i}",
                text=f"Document {i}",
                metadata={},
                token_ids=np.array([i], dtype=np.uint32),
                weights=np.array([0.5], dtype=np.float32),
            )
            for i in range(100)
        ]
        index.add_batch(docs)

        # Should still have current_writer open (shard not finalized)
        assert index.current_writer is not None, "Shard should still be open after add_batch"

        # Should have 0 finalized shards (all docs still in buffer)
        assert index.metadata["num_shards"] == 0, "Should not have finalized any shards yet"

    def test_shards_reach_target_size(self, temp_index_dir):
        """Regression: shards should grow to ~32MB, not ~350KB per batch (Bug 1)."""
        target_size_mb = 1  # Use 1MB for faster test
        index = SpladeIndex(temp_index_dir, shard_size_mb=target_size_mb)

        # Add documents in batches until we get a rotated shard
        batch_size = 100
        for batch_num in range(100):  # Add up to 10,000 docs
            docs = [
                Document(
                    doc_id=f"doc_{batch_num}_{i}",
                    text=f"Document {batch_num}_{i} " * 50,  # Make docs larger
                    metadata={},
                    token_ids=np.random.randint(0, 1000, 30, dtype=np.uint32),
                    weights=np.random.rand(30).astype(np.float32),
                )
                for i in range(batch_size)
            ]
            index.add_batch(docs)

            # Check if we've rotated a shard
            if index.metadata["num_shards"] > 0:
                break

        # We should have rotated at least one shard
        assert index.metadata["num_shards"] > 0, "Should have created at least one shard"

        # Finalize current shard to check its size
        index._finalize_current_shard()

        # Get shard sizes
        shard_paths = index._get_shard_paths()
        assert len(shard_paths) > 0, "Should have at least one shard"

        # Check that shards are reasonably close to target size
        # They should be close to 1MB, definitely not ~350KB
        for shard_path in shard_paths[:-1]:  # Check all but last (may be partial)
            size_mb = shard_path.stat().st_size / (1024 * 1024)
            # Allow 50% tolerance (0.5MB to 1.5MB for 1MB target)
            assert size_mb >= target_size_mb * 0.5, (
                f"Shard {shard_path.name} is {size_mb:.2f}MB, should be ~{target_size_mb}MB"
            )

    def test_reshard_updates_metadata(self, temp_index_dir):
        """Regression: reshard must update metadata.json (Bug 2)."""
        index = SpladeIndex(temp_index_dir, shard_size_mb=0.01)

        # Add documents to create multiple small shards
        for i in range(100):
            doc = Document(
                doc_id=f"doc_{i}",
                text=f"Document {i} " * 20,
                metadata={},
                token_ids=np.random.randint(0, 1000, 10, dtype=np.uint32),
                weights=np.random.rand(10).astype(np.float32),
            )
            index.add(doc)

        initial_num_shards = index.metadata["num_shards"]
        assert initial_num_shards > 1, "Should have multiple shards before reshard"

        # Reshard to larger size
        new_target_size = 1
        stats = index.reshard(target_shard_size_mb=new_target_size)

        # Verify metadata was updated
        assert index.metadata["num_docs"] == 100, "Doc count should be preserved"
        assert index.metadata["num_shards"] == stats["new_shards"], (
            "num_shards should match reshard result"
        )
        assert len(index.metadata["shard_hashes"]) == stats["new_shards"], (
            "shard_hashes length should match num_shards"
        )

        # Bug 2 specific: verify shard_size_mb was updated
        assert index.metadata["shard_size_mb"] == new_target_size, (
            f"shard_size_mb should be updated to {new_target_size}"
        )

        # Verify all shards exist
        for shard_hash in index.metadata["shard_hashes"]:
            shard_path = index.index_dir / f"{shard_hash}.fb"
            assert shard_path.exists(), f"Shard {shard_hash} should exist on disk"
