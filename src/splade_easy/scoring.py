# src/splade_easy/scoring.py

import numba
import numpy as np


@numba.njit(fastmath=True, cache=True)
def compute_splade_score(
    doc_tokens: np.ndarray,
    doc_weights: np.ndarray,
    query_tokens: np.ndarray,
    query_weights: np.ndarray,
) -> float:
    """
    Compute SPLADE score between document and query using two-pointer merge.

    SPLADE score is the sum of products of matching token weights:
    score = sum(doc_weight[i] * query_weight[i]) for all matching tokens

    IMPORTANT: Assumes both token arrays are sorted and DEDUPLICATED.
    Use ensure_sorted_splade_vector() during indexing to guarantee this.

    Args:
        doc_tokens: Document token IDs (uint32 array, SORTED, UNIQUE)
        doc_weights: Document token weights (float32 array)
        query_tokens: Query token IDs (uint32 array, SORTED, UNIQUE)
        query_weights: Query token weights (float32 array)

    Returns:
        Similarity score (higher is better)
    """
    score = 0.0
    i, j = 0, 0
    n_doc = len(doc_tokens)
    n_query = len(query_tokens)

    while i < n_doc and j < n_query:
        doc_tok = doc_tokens[i]
        query_tok = query_tokens[j]

        if doc_tok == query_tok:
            score += doc_weights[i] * query_weights[j]
            i += 1
            j += 1
        elif doc_tok < query_tok:
            i += 1
        else:
            j += 1

    return score


def ensure_sorted_splade_vector(
    token_ids: np.ndarray, weights: np.ndarray, deduplicate: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensure SPLADE vector is sorted by token ID and optionally deduplicated.

    Call this during indexing/encoding to prepare vectors for fast scoring.

    For duplicate tokens, keeps the MAXIMUM weight (typical for SPLADE).
    This matches the behavior of properly-encoded SPLADE vectors.

    Args:
        token_ids: Token IDs
        weights: Corresponding weights
        deduplicate: If True, remove duplicate tokens (recommended)

    Returns:
        (sorted_token_ids, sorted_weights)
    """
    if len(token_ids) == 0:
        return token_ids, weights

    # First, check if we need to deduplicate
    if deduplicate and len(token_ids) != len(np.unique(token_ids)):
        # Has duplicates - need to deduplicate
        # For each unique token, keep the maximum weight
        unique_tokens = np.unique(token_ids)
        dedup_weights = np.zeros(len(unique_tokens), dtype=weights.dtype)

        for i, token in enumerate(unique_tokens):
            # Find all occurrences of this token
            mask = token_ids == token
            # Keep maximum weight (typical SPLADE behavior)
            dedup_weights[i] = np.max(weights[mask])

        token_ids = unique_tokens
        weights = dedup_weights

    # Now sort if needed
    if len(token_ids) <= 1:
        return token_ids, weights

    # Check if already sorted
    if np.all(token_ids[:-1] <= token_ids[1:]):
        return token_ids, weights

    # Sort both arrays by token_ids
    sort_idx = np.argsort(token_ids)
    return token_ids[sort_idx], weights[sort_idx]
