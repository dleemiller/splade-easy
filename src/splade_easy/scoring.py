# src/splade_easy/scoring.py

import numpy as np


def compute_splade_score(
    doc_tokens: np.ndarray,
    doc_weights: np.ndarray,
    query_tokens: np.ndarray,
    query_weights: np.ndarray,
) -> float:
    """
    Compute SPLADE score between document and query.

    SPLADE score is the sum of products of matching token weights:
    score = sum(doc_weight[i] * query_weight[i]) for all matching tokens

    Args:
        doc_tokens: Document token IDs (uint32 array)
        doc_weights: Document token weights (float32 array)
        query_tokens: Query token IDs (uint32 array)
        query_weights: Query token weights (float32 array)

    Returns:
        Similarity score (higher is better)
    """
    # Create lookup dictionary for doc tokens -> weights
    doc_dict = dict(zip(doc_tokens, doc_weights))

    # Vectorized lookup and multiply
    score = 0.0
    for q_token, q_weight in zip(query_tokens, query_weights):
        doc_weight = doc_dict.get(q_token, 0.0)
        if doc_weight > 0:
            score += doc_weight * q_weight

    return score
