"""Tests for IR metrics."""

from __future__ import annotations

import math

import numpy as np

from splade_easy.eval.metrics import evaluate, mrr_at_k, ndcg_at_k, recall_at_k


def test_ndcg_perfect_ranking():
    ranking = np.array([2, 5, 7], dtype=np.int32)
    qrels = {2: 1, 5: 1, 7: 1}
    assert ndcg_at_k(ranking, qrels, k=3) == 1.0


def test_ndcg_zero_when_no_relevant():
    ranking = np.array([2, 5, 7], dtype=np.int32)
    qrels = {1: 1}  # not in top-k
    assert ndcg_at_k(ranking, qrels, k=3) == 0.0


def test_ndcg_known_value():
    # Single relevant doc at rank 2
    ranking = np.array([10, 20, 30], dtype=np.int32)
    qrels = {20: 1}
    # DCG = (2^1 - 1)/log2(3) = 1/log2(3); IDCG = 1/log2(2) = 1
    expected = 1.0 / math.log2(3)
    assert math.isclose(ndcg_at_k(ranking, qrels, k=3), expected, abs_tol=1e-6)


def test_recall_at_k():
    ranking = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    qrels = {2: 1, 5: 1, 99: 1}  # 3 relevant; 2 of them in top-5
    assert recall_at_k(ranking, qrels, k=5) == 2 / 3
    # k=3: only doc 2 hit
    assert math.isclose(recall_at_k(ranking, qrels, k=3), 1 / 3, abs_tol=1e-6)


def test_mrr_first_relevant_at_rank_3():
    ranking = np.array([1, 2, 3], dtype=np.int32)
    qrels = {3: 1}
    assert math.isclose(mrr_at_k(ranking, qrels, k=10), 1 / 3, abs_tol=1e-6)


def test_mrr_none_relevant():
    ranking = np.array([1, 2, 3], dtype=np.int32)
    qrels = {99: 1}
    assert mrr_at_k(ranking, qrels, k=10) == 0.0


def test_evaluate_aggregates_means():
    rankings = {
        "q1": np.array([1, 2, 3], dtype=np.int32),
        "q2": np.array([3, 2, 1], dtype=np.int32),
    }
    qrels = {
        "q1": {1: 1},  # MRR=1, NDCG=1
        "q2": {1: 1},  # MRR=1/3, NDCG=1/log2(4)=0.5
    }
    out = evaluate(rankings, qrels, k=3)
    assert out["n_queries"] == 2
    assert math.isclose(out["mrr@k"], (1.0 + 1 / 3) / 2, abs_tol=1e-6)
