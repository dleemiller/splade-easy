"""IR metrics: NDCG@k, Recall@k, MRR@k. Numpy-only, no pytrec_eval."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _dcg(rels: np.ndarray) -> float:
    if rels.size == 0:
        return 0.0
    gains = np.power(2.0, rels, dtype=np.float64) - 1.0
    discounts = np.log2(np.arange(2, rels.size + 2, dtype=np.float64))
    return float((gains / discounts).sum())


def ndcg_at_k(
    ranked_doc_ids: np.ndarray,
    qrels: Mapping[int, int],
    k: int,
) -> float:
    """NDCG@k. `qrels` maps doc_id -> non-negative relevance grade."""
    rels = np.fromiter(
        (qrels.get(int(d), 0) for d in ranked_doc_ids[:k]),
        dtype=np.float64,
        count=min(k, len(ranked_doc_ids)),
    )
    dcg = _dcg(rels)
    ideal_rels = np.fromiter(
        sorted((g for g in qrels.values() if g > 0), reverse=True),
        dtype=np.float64,
    )[:k]
    idcg = _dcg(ideal_rels)
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(
    ranked_doc_ids: np.ndarray,
    qrels: Mapping[int, int],
    k: int,
) -> float:
    relevant = {did for did, rel in qrels.items() if rel > 0}
    if not relevant:
        return 0.0
    top_k = {int(d) for d in ranked_doc_ids[:k]}
    return len(top_k & relevant) / len(relevant)


def mrr_at_k(
    ranked_doc_ids: np.ndarray,
    qrels: Mapping[int, int],
    k: int,
) -> float:
    relevant = {did for did, rel in qrels.items() if rel > 0}
    if not relevant:
        return 0.0
    for rank, did in enumerate(ranked_doc_ids[:k], start=1):
        if int(did) in relevant:
            return 1.0 / rank
    return 0.0


def evaluate(
    rankings: Mapping[str, np.ndarray],
    qrels_by_query: Mapping[str, Mapping[int, int]],
    k: int = 10,
) -> dict:
    """Mean NDCG@k, Recall@k, MRR@k across all queries that have qrels."""
    ndcgs, recalls, mrrs = [], [], []
    for qid, ranking in rankings.items():
        qrels = qrels_by_query.get(qid)
        if not qrels:
            continue
        ndcgs.append(ndcg_at_k(ranking, qrels, k))
        recalls.append(recall_at_k(ranking, qrels, k))
        mrrs.append(mrr_at_k(ranking, qrels, k))
    n = len(ndcgs)
    return {
        "ndcg@k": float(np.mean(ndcgs)) if n else 0.0,
        "recall@k": float(np.mean(recalls)) if n else 0.0,
        "mrr@k": float(np.mean(mrrs)) if n else 0.0,
        "n_queries": n,
        "k": k,
    }
