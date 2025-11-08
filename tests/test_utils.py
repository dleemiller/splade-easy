import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest

from splade_easy.utils import extract_splade_vectors, get_shard_paths, hash_file


class TestGetShardPaths:
    """Test shard path retrieval."""

    def test_get_existing_shards(self):
        """Test getting paths for existing shards."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir)

            # Create shard files
            hash1 = "a" * 64
            hash2 = "b" * 64

            (index_dir / f"{hash1}.fb").touch()
            (index_dir / f"{hash2}.fb").touch()

            metadata = {"shard_hashes": [hash1, hash2]}
            paths = get_shard_paths(index_dir, metadata)

            assert len(paths) == 2
            assert all(p.exists() for p in paths)

    def test_missing_shards_non_strict_logs_warning(self, caplog):
        """Missing shards should log a warning but still return existing ones."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir)

            hash_existing = "a" * 64
            hash_missing = "b" * 64

            # Only create one of the shards
            (index_dir / f"{hash_existing}.fb").touch()
            metadata = {"shard_hashes": [hash_existing, hash_missing]}

            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="splade_easy.utils"):
                paths = get_shard_paths(index_dir, metadata)

            # Only existing shard is returned
            assert len(paths) == 1
            assert paths[0].name == f"{hash_existing}.fb"

            # A warning about missing shards should be logged
            warning_messages = [rec.message for rec in caplog.records if rec.levelname == "WARNING"]
            assert any("missing on disk" in msg for msg in warning_messages)

    def test_missing_shards_strict_raises(self):
        """Strict mode should raise when expected shards are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir)

            hash_existing = "a" * 64
            hash_missing = "b" * 64

            (index_dir / f"{hash_existing}.fb").touch()
            metadata = {"shard_hashes": [hash_existing, hash_missing]}

            with pytest.raises(FileNotFoundError):
                get_shard_paths(index_dir, metadata, strict=True)

    def test_extract_from_dict(self):
        """Test extraction from dictionary format."""
        encoding = {"indices": [1, 5, 10], "values": [0.8, 0.5, 0.3]}
        token_ids, weights = extract_splade_vectors(encoding)

        np.testing.assert_array_equal(token_ids, np.array([1, 5, 10], dtype=np.uint32))
        np.testing.assert_array_almost_equal(weights, np.array([0.8, 0.5, 0.3], dtype=np.float32))

    def test_extract_from_dense_array(self):
        """Test extraction from dense numpy array."""
        encoding = np.zeros(10)
        encoding[1] = 0.8
        encoding[5] = 0.5
        encoding[9] = 0.3

        token_ids, weights = extract_splade_vectors(encoding)

        np.testing.assert_array_equal(token_ids, np.array([1, 5, 9], dtype=np.uint32))
        np.testing.assert_array_almost_equal(weights, np.array([0.8, 0.5, 0.3], dtype=np.float32))

    def test_extract_invalid_format(self):
        """Test that invalid format raises error."""
        with pytest.raises(ValueError, match="Unsupported encoding format"):
            extract_splade_vectors("invalid")


class TestHashFile:
    """Test file hashing functionality."""

    def test_hash_file(self):
        """Test hashing produces consistent SHA256."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_path = Path(f.name)

        try:
            hash1 = hash_file(temp_path)
            hash2 = hash_file(temp_path)

            assert len(hash1) == 64  # SHA256 hex length
            assert hash1 == hash2  # Deterministic
        finally:
            temp_path.unlink()
