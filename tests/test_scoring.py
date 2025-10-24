# tests/test_scoring.py

import numpy as np

from splade_easy.scoring import compute_splade_score


class TestSpladeScoring:
    """Test SPLADE scoring logic."""

    def test_exact_match(self):
        """Test scoring with identical vectors."""
        tokens = np.array([1, 5, 10], dtype=np.uint32)
        weights = np.array([0.8, 0.5, 0.3], dtype=np.float32)

        score = compute_splade_score(tokens, weights, tokens, weights)
        expected = 0.8**2 + 0.5**2 + 0.3**2  # 0.64 + 0.25 + 0.09 = 0.98

        assert abs(score - expected) < 1e-5

    def test_partial_overlap(self):
        """Test scoring with partial token overlap."""
        doc_tokens = np.array([1, 5, 10], dtype=np.uint32)
        doc_weights = np.array([0.8, 0.5, 0.3], dtype=np.float32)

        query_tokens = np.array([1, 10, 20], dtype=np.uint32)
        query_weights = np.array([0.9, 0.6, 0.4], dtype=np.float32)

        score = compute_splade_score(doc_tokens, doc_weights, query_tokens, query_weights)
        # Overlap: token 1 (0.8 * 0.9) + token 10 (0.3 * 0.6) = 0.72 + 0.18 = 0.90
        expected = 0.90

        assert abs(score - expected) < 1e-5

    def test_no_overlap(self):
        """Test scoring with no common tokens."""
        doc_tokens = np.array([1, 2, 3], dtype=np.uint32)
        doc_weights = np.array([0.8, 0.5, 0.3], dtype=np.float32)

        query_tokens = np.array([10, 20, 30], dtype=np.uint32)
        query_weights = np.array([0.9, 0.6, 0.4], dtype=np.float32)

        score = compute_splade_score(doc_tokens, doc_weights, query_tokens, query_weights)
        assert score == 0.0

    def test_single_token_match(self):
        """Test scoring with single token match."""
        doc_tokens = np.array([42], dtype=np.uint32)
        doc_weights = np.array([0.7], dtype=np.float32)

        query_tokens = np.array([42], dtype=np.uint32)
        query_weights = np.array([0.5], dtype=np.float32)

        score = compute_splade_score(doc_tokens, doc_weights, query_tokens, query_weights)
        expected = 0.7 * 0.5

        assert abs(score - expected) < 1e-5

    def test_empty_vectors(self):
        """Test scoring with empty vectors."""
        empty_tokens = np.array([], dtype=np.uint32)
        empty_weights = np.array([], dtype=np.float32)

        doc_tokens = np.array([1, 2], dtype=np.uint32)
        doc_weights = np.array([0.5, 0.3], dtype=np.float32)

        # Empty query vs doc
        score = compute_splade_score(doc_tokens, doc_weights, empty_tokens, empty_weights)
        assert score == 0.0

        # Empty doc vs query
        score = compute_splade_score(empty_tokens, empty_weights, doc_tokens, doc_weights)
        assert score == 0.0

    def test_large_vectors(self):
        """Test scoring with realistic SPLADE vector sizes."""
        # Typical SPLADE has ~100-300 non-zero dimensions
        np.random.seed(42)
        n_doc = 200
        n_query = 150
        overlap = 50

        # Create vectors with some overlap
        doc_tokens = np.array(
            sorted(np.random.choice(30000, n_doc, replace=False)), dtype=np.uint32
        )
        doc_weights = np.random.rand(n_doc).astype(np.float32)

        # Query shares first 'overlap' tokens
        query_tokens = np.array(
            sorted(
                list(doc_tokens[:overlap])
                + list(np.random.choice(30000, n_query - overlap, replace=False))
            ),
            dtype=np.uint32,
        )
        query_weights = np.random.rand(n_query).astype(np.float32)

        score = compute_splade_score(doc_tokens, doc_weights, query_tokens, query_weights)

        # Score should be positive (have overlap) and reasonable magnitude
        assert score > 0
        assert score < 100  # Sanity check

    def test_score_symmetry_not_required(self):
        """SPLADE scoring is NOT symmetric (doc vs query order matters)."""
        doc_tokens = np.array([1, 2], dtype=np.uint32)
        doc_weights = np.array([0.8, 0.5], dtype=np.float32)

        query_tokens = np.array([1, 2], dtype=np.uint32)
        query_weights = np.array([0.3, 0.9], dtype=np.float32)

        # Different weights, but same tokens
        score = compute_splade_score(doc_tokens, doc_weights, query_tokens, query_weights)
        expected = 0.8 * 0.3 + 0.5 * 0.9  # 0.24 + 0.45 = 0.69

        assert abs(score - expected) < 1e-5
