# tests/test_utils.py

import numpy as np
import pytest

from splade_easy.utils import extract_splade_vectors


class TestUtils:
    """Test utility functions."""

    def test_extract_from_dict(self):
        """Test extraction from dictionary format."""
        encoding = {"indices": [1, 5, 10], "values": [0.8, 0.5, 0.3]}

        token_ids, weights = extract_splade_vectors(encoding)

        np.testing.assert_array_equal(token_ids, np.array([1, 5, 10], dtype=np.uint32))
        np.testing.assert_array_almost_equal(weights, np.array([0.8, 0.5, 0.3], dtype=np.float32))

    def test_extract_from_dense_array(self):
        """Test extraction from dense numpy array."""
        # Dense array with some non-zero values
        encoding = np.zeros(10)
        encoding[1] = 0.8
        encoding[5] = 0.5
        encoding[9] = 0.3

        token_ids, weights = extract_splade_vectors(encoding)

        np.testing.assert_array_equal(token_ids, np.array([1, 5, 9], dtype=np.uint32))
        np.testing.assert_array_almost_equal(weights, np.array([0.8, 0.5, 0.3], dtype=np.float32))

    def test_extract_from_object_with_attributes(self):
        """Test extraction from object with indices/values attributes."""

        class MockEncoding:
            def __init__(self):
                self.indices = [1, 5, 10]
                self.values = [0.8, 0.5, 0.3]

        encoding = MockEncoding()
        token_ids, weights = extract_splade_vectors(encoding)

        np.testing.assert_array_equal(token_ids, np.array([1, 5, 10], dtype=np.uint32))
        np.testing.assert_array_almost_equal(weights, np.array([0.8, 0.5, 0.3], dtype=np.float32))

    def test_extract_invalid_format(self):
        """Test that invalid format raises error."""
        with pytest.raises(ValueError, match="Unsupported encoding format"):
            extract_splade_vectors("invalid")

    def test_extract_empty_dense(self):
        """Test extraction from empty dense array."""
        encoding = np.zeros(10)
        token_ids, weights = extract_splade_vectors(encoding)

        assert len(token_ids) == 0
        assert len(weights) == 0
