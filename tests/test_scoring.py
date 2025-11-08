# tests/test_scoring.py

import numpy as np

from splade_easy.scoring import compute_splade_score


class TestSpladeScoring:
    """Test SPLADE scoring logic."""

    def test_exact_match_cosine(self):
        """Test cosine similarity with identical vectors (should be 1.0)."""
        tokens = np.array([1, 5, 10], dtype=np.uint32)
        weights = np.array([0.8, 0.5, 0.3], dtype=np.float32)

        score = compute_splade_score(tokens, weights, tokens, weights, use_cosine=True)
        # Cosine of identical vectors is always 1.0
        expected = 1.0

        assert abs(score - expected) < 1e-5

    def test_exact_match_dot_product(self):
        """Test dot product with identical vectors."""
        tokens = np.array([1, 5, 10], dtype=np.uint32)
        weights = np.array([0.8, 0.5, 0.3], dtype=np.float32)

        score = compute_splade_score(tokens, weights, tokens, weights, use_cosine=False)
        expected = 0.8**2 + 0.5**2 + 0.3**2  # 0.64 + 0.25 + 0.09 = 0.98

        assert abs(score - expected) < 1e-5

    def test_partial_overlap_cosine(self):
        """Test cosine similarity with partial token overlap."""
        doc_tokens = np.array([1, 5, 10], dtype=np.uint32)
        doc_weights = np.array([0.8, 0.5, 0.3], dtype=np.float32)

        query_tokens = np.array([1, 10, 20], dtype=np.uint32)
        query_weights = np.array([0.9, 0.6, 0.4], dtype=np.float32)

        score = compute_splade_score(
            doc_tokens, doc_weights, query_tokens, query_weights, use_cosine=True
        )

        # Calculate expected cosine similarity
        dot_product = 0.8 * 0.9 + 0.3 * 0.6  # 0.72 + 0.18 = 0.90
        doc_norm = np.sqrt(0.8**2 + 0.5**2 + 0.3**2)  # sqrt(0.98) ≈ 0.9899
        query_norm = np.sqrt(0.9**2 + 0.6**2 + 0.4**2)  # sqrt(1.33) ≈ 1.1533
        expected = dot_product / (doc_norm * query_norm)  # 0.90 / 1.1414 ≈ 0.7883

        assert abs(score - expected) < 1e-5

    def test_partial_overlap_dot_product(self):
        """Test dot product with partial token overlap."""
        doc_tokens = np.array([1, 5, 10], dtype=np.uint32)
        doc_weights = np.array([0.8, 0.5, 0.3], dtype=np.float32)

        query_tokens = np.array([1, 10, 20], dtype=np.uint32)
        query_weights = np.array([0.9, 0.6, 0.4], dtype=np.float32)

        score = compute_splade_score(
            doc_tokens, doc_weights, query_tokens, query_weights, use_cosine=False
        )
        # Overlap: token 1 (0.8 * 0.9) + token 10 (0.3 * 0.6) = 0.72 + 0.18 = 0.90
        expected = 0.90

        assert abs(score - expected) < 1e-5

    def test_no_overlap_cosine(self):
        """Test cosine similarity with no common tokens."""
        doc_tokens = np.array([1, 2, 3], dtype=np.uint32)
        doc_weights = np.array([0.8, 0.5, 0.3], dtype=np.float32)

        query_tokens = np.array([10, 20, 30], dtype=np.uint32)
        query_weights = np.array([0.9, 0.6, 0.4], dtype=np.float32)

        score = compute_splade_score(
            doc_tokens, doc_weights, query_tokens, query_weights, use_cosine=True
        )
        assert score == 0.0

    def test_no_overlap_dot_product(self):
        """Test dot product with no common tokens."""
        doc_tokens = np.array([1, 2, 3], dtype=np.uint32)
        doc_weights = np.array([0.8, 0.5, 0.3], dtype=np.float32)

        query_tokens = np.array([10, 20, 30], dtype=np.uint32)
        query_weights = np.array([0.9, 0.6, 0.4], dtype=np.float32)

        score = compute_splade_score(
            doc_tokens, doc_weights, query_tokens, query_weights, use_cosine=False
        )
        assert score == 0.0

    def test_single_token_match_cosine(self):
        """Test cosine similarity with single token match."""
        doc_tokens = np.array([42], dtype=np.uint32)
        doc_weights = np.array([0.7], dtype=np.float32)

        query_tokens = np.array([42], dtype=np.uint32)
        query_weights = np.array([0.5], dtype=np.float32)

        score = compute_splade_score(
            doc_tokens, doc_weights, query_tokens, query_weights, use_cosine=True
        )
        # Cosine of aligned single-dimension vectors is always 1.0
        expected = 1.0

        assert abs(score - expected) < 1e-5

    def test_single_token_match_dot_product(self):
        """Test dot product with single token match."""
        doc_tokens = np.array([42], dtype=np.uint32)
        doc_weights = np.array([0.7], dtype=np.float32)

        query_tokens = np.array([42], dtype=np.uint32)
        query_weights = np.array([0.5], dtype=np.float32)

        score = compute_splade_score(
            doc_tokens, doc_weights, query_tokens, query_weights, use_cosine=False
        )
        expected = 0.7 * 0.5

        assert abs(score - expected) < 1e-5

    def test_empty_vectors(self):
        """Test scoring with empty vectors (works same for both modes)."""
        empty_tokens = np.array([], dtype=np.uint32)
        empty_weights = np.array([], dtype=np.float32)

        doc_tokens = np.array([1, 2], dtype=np.uint32)
        doc_weights = np.array([0.5, 0.3], dtype=np.float32)

        # Empty query vs doc
        score = compute_splade_score(
            doc_tokens, doc_weights, empty_tokens, empty_weights, use_cosine=True
        )
        assert score == 0.0

        # Empty doc vs query
        score = compute_splade_score(
            empty_tokens, empty_weights, doc_tokens, doc_weights, use_cosine=True
        )
        assert score == 0.0

        # Also test dot product mode
        score = compute_splade_score(
            doc_tokens, doc_weights, empty_tokens, empty_weights, use_cosine=False
        )
        assert score == 0.0

    def test_large_vectors_cosine(self):
        """Test cosine similarity with realistic SPLADE vector sizes."""
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

        score = compute_splade_score(
            doc_tokens, doc_weights, query_tokens, query_weights, use_cosine=True
        )

        # Cosine similarity should be in [0, 1] range
        assert score > 0
        assert score <= 1.0

    def test_large_vectors_dot_product(self):
        """Test dot product with realistic SPLADE vector sizes."""
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

        score = compute_splade_score(
            doc_tokens, doc_weights, query_tokens, query_weights, use_cosine=False
        )

        # Score should be positive (have overlap) and reasonable magnitude
        assert score > 0
        assert score < 100  # Sanity check

    def test_cosine_vs_dot_product_ranking(self):
        """Test that cosine and dot product can produce different rankings."""
        # Query
        query_tokens = np.array([1, 2, 3], dtype=np.uint32)
        query_weights = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        # Doc 1: Similar direction, normal magnitude
        doc1_tokens = np.array([1, 2, 3], dtype=np.uint32)
        doc1_weights = np.array([0.4, 0.4, 0.4], dtype=np.float32)

        # Doc 2: Similar direction, INFLATED magnitude
        doc2_tokens = np.array([1, 2, 3], dtype=np.uint32)
        doc2_weights = np.array([2.0, 2.0, 2.0], dtype=np.float32)

        # Dot product scores
        score1_dot = compute_splade_score(
            doc1_tokens, doc1_weights, query_tokens, query_weights, use_cosine=False
        )
        score2_dot = compute_splade_score(
            doc2_tokens, doc2_weights, query_tokens, query_weights, use_cosine=False
        )

        # Cosine scores
        score1_cos = compute_splade_score(
            doc1_tokens, doc1_weights, query_tokens, query_weights, use_cosine=True
        )
        score2_cos = compute_splade_score(
            doc2_tokens, doc2_weights, query_tokens, query_weights, use_cosine=True
        )

        # Dot product: Doc 2 should win (larger magnitude)
        assert score2_dot > score1_dot

        # Cosine: Both should be nearly identical (same direction)
        assert abs(score1_cos - score2_cos) < 1e-5

    def test_default_is_cosine(self):
        """Test that default behavior is cosine similarity."""
        tokens = np.array([1, 2], dtype=np.uint32)
        weights = np.array([0.8, 0.5], dtype=np.float32)

        # Default call (no use_cosine specified)
        default_score = compute_splade_score(tokens, weights, tokens, weights)

        # Explicit cosine call
        cosine_score = compute_splade_score(tokens, weights, tokens, weights, use_cosine=True)

        # Should be the same
        assert abs(default_score - cosine_score) < 1e-7

        # Should be 1.0 for identical vectors
        assert abs(default_score - 1.0) < 1e-5
