# tests/test_reshard.py

import tempfile
from pathlib import Path

import numpy as np

from splade_easy import Document, SpladeIndex
from splade_easy.utils import hash_file


class TestReshard:
    """Test resharding functionality."""

    def test_basic_reshard(self):
        """Test basic resharding reduces shard count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = SpladeIndex(tmpdir, shard_size_mb=0.01)

            # Add documents to create multiple shards
            for i in range(30):
                doc = Document(
                    doc_id=f"doc_{i}",
                    text=f"Document {i} " * 50,
                    metadata={"idx": str(i)},
                    token_ids=np.random.randint(0, 1000, 20, dtype=np.uint32),
                    weights=np.random.rand(20).astype(np.float32),
                )
                index.add(doc)

            index._finalize_current_shard()
            initial_shards = len(index.metadata["shard_hashes"])

            # Reshard to larger size
            stats = index.reshard(target_shard_size_mb=1)

            assert stats["new_shards"] < initial_shards
            assert stats["docs_written"] == 30

    def test_reshard_with_deleted_docs(self):
        """Test resharding excludes deleted documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = SpladeIndex(tmpdir)

            # Add and delete some docs
            for i in range(20):
                doc = Document(
                    doc_id=f"doc_{i}",
                    text=f"Document {i}",
                    metadata={},
                    token_ids=np.array([i], dtype=np.uint32),
                    weights=np.array([0.5], dtype=np.float32),
                )
                index.add(doc)

            index._finalize_current_shard()

            for i in range(10):
                index.delete(f"doc_{i}")

            # Reshard
            stats = index.reshard(target_shard_size_mb=1)

            assert stats["docs_written"] == 10
            assert len(index.deleted_ids) == 0

            # Verify only non-deleted docs exist
            retriever = index.retriever(tmpdir)
            for i in range(10):
                assert retriever.get(f"doc_{i}") is None
            for i in range(10, 20):
                assert retriever.get(f"doc_{i}") is not None

    def test_reshard_preserves_content(self):
        """Test that document content is preserved during reshard."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = SpladeIndex(tmpdir, shard_size_mb=0.01)

            # Add documents
            for i in range(10):
                doc = Document(
                    doc_id=f"doc_{i}",
                    text=f"Content {i}",
                    metadata={"num": str(i)},
                    token_ids=np.array([i, i + 1], dtype=np.uint32),
                    weights=np.array([0.8, 0.6], dtype=np.float32),
                )
                index.add(doc)

            index._finalize_current_shard()
            index.reshard(target_shard_size_mb=1)

            # Create fresh retriever after reshard (metadata has changed)
            retriever = SpladeIndex.retriever(tmpdir)

            # Verify content intact
            for i in range(10):
                doc = retriever.get(f"doc_{i}")
                assert doc["text"] == f"Content {i}"
                assert doc["metadata"]["num"] == str(i)

    def test_reshard_hash_verification(self):
        """Test that resharded shards have correct hash-based names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = SpladeIndex(tmpdir)

            for i in range(10):
                doc = Document(
                    doc_id=f"doc_{i}",
                    text=f"Doc {i}",
                    metadata={},
                    token_ids=np.array([i], dtype=np.uint32),
                    weights=np.array([0.5], dtype=np.float32),
                )
                index.add(doc)

            index._finalize_current_shard()
            index.reshard(target_shard_size_mb=1)

            # Reload index to get fresh metadata
            index2 = SpladeIndex(tmpdir)

            # Verify each shard's filename matches its content hash
            for shard_hash in index2.metadata["shard_hashes"]:
                shard_path = Path(tmpdir) / f"{shard_hash}.fb"
                assert shard_path.exists()
                assert hash_file(shard_path) == shard_hash
