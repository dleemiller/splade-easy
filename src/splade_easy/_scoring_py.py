"""Reference numpy implementation of SPLADE scoring + top-k. Used in tests and as the fallback when the Cython extension is not built."""

from __future__ import annotations

import numpy as np


def score_csc(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    q_ids: np.ndarray,
    q_weights: np.ndarray,
    n_docs: int,
) -> np.ndarray:
    """Score all docs for one query against a CSC inverted index. Returns float32 (n_docs,)."""
    scores = np.zeros(n_docs, dtype=np.float32)
    for tid, qw in zip(q_ids, q_weights, strict=True):
        s = int(indptr[tid])
        e = int(indptr[tid + 1])
        if s == e:
            continue
        scores[indices[s:e]] += np.float32(qw) * data[s:e]
    return scores


def topk(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k by score, stable (deterministic) on ties. Returns (doc_ids, scores), both length min(k, n_docs)."""
    n = scores.shape[0]
    k_eff = min(k, n)
    if k_eff == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    if k_eff == n:
        order = np.argsort(-scores, kind="stable")
    else:
        part = np.argpartition(-scores, k_eff - 1)[:k_eff]
        order = part[np.argsort(-scores[part], kind="stable")]
    return order.astype(np.int32, copy=False), scores[order]


def score_topk_batch(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    q_ids_list: list[np.ndarray],
    q_weights_list: list[np.ndarray],
    n_docs: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run score+topk over a batch of queries. Returns (ids, scores), both (n_queries, k_eff)."""
    n_queries = len(q_ids_list)
    k_eff = min(k, n_docs)
    out_ids = np.empty((n_queries, k_eff), dtype=np.int32)
    out_scores = np.empty((n_queries, k_eff), dtype=np.float32)
    for i, (qids, qws) in enumerate(zip(q_ids_list, q_weights_list, strict=True)):
        scores = score_csc(indptr, indices, data, qids, qws, n_docs)
        ids, scs = topk(scores, k_eff)
        out_ids[i] = ids
        out_scores[i] = scs
    return out_ids, out_scores
