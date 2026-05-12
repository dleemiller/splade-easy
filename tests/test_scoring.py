"""Tests for the numpy reference scoring path."""

from __future__ import annotations

import numpy as np

from splade_easy import sparse
from splade_easy._scoring_py import score_csc, score_topk_batch, topk


def _build_csc(corpus_per_doc, vocab):
    ids = [np.array([t for t, _ in d], dtype=np.int32) for d in corpus_per_doc]
    ws = [np.array([w for _, w in d], dtype=np.float32) for d in corpus_per_doc]
    sc = sparse.from_per_doc(ids, ws, vocab_size=vocab)
    return sparse.csr_to_csc(sc)


def test_score_csc_basic():
    # 3 docs, vocab 5
    ip, idx, d = _build_csc(
        [
            [(1, 0.5), (3, 0.2)],
            [(0, 0.4), (2, 0.3)],
            [(3, 0.6), (4, 0.1)],
        ],
        vocab=5,
    )

    q_ids = np.array([3], dtype=np.int32)
    q_ws = np.array([1.0], dtype=np.float32)
    scores = score_csc(ip, idx, d, q_ids, q_ws, n_docs=3)
    np.testing.assert_allclose(scores, [0.2, 0.0, 0.6], atol=1e-6)

    q_ids = np.array([0, 3], dtype=np.int32)
    q_ws = np.array([2.0, 1.0], dtype=np.float32)
    scores = score_csc(ip, idx, d, q_ids, q_ws, n_docs=3)
    # doc 0: 0 + 0.2 = 0.2; doc 1: 0.8 + 0 = 0.8; doc 2: 0 + 0.6 = 0.6
    np.testing.assert_allclose(scores, [0.2, 0.8, 0.6], atol=1e-6)


def test_score_csc_unknown_token_silent():
    """A query token whose column is empty contributes 0."""
    ip, idx, d = _build_csc([[(0, 1.0)]], vocab=4)
    # Token 3 exists in vocab but no doc has it
    scores = score_csc(
        ip, idx, d, np.array([3], dtype=np.int32), np.array([1.0], dtype=np.float32), n_docs=1
    )
    np.testing.assert_array_equal(scores, [0.0])


def test_topk_stable_on_ties():
    """When scores are tied, lower doc index should come first (deterministic)."""
    scores = np.array([0.5, 0.5, 0.5, 0.1, 0.5], dtype=np.float32)
    ids, top = topk(scores, k=3)
    # Stable: doc 0, 1, 2 (all tied at 0.5)
    assert ids.tolist() == [0, 1, 2]
    np.testing.assert_allclose(top, [0.5, 0.5, 0.5])


def test_topk_full_sort_when_k_ge_n():
    scores = np.array([0.1, 0.7, 0.3], dtype=np.float32)
    ids, top = topk(scores, k=10)
    # Should clamp to n=3
    assert ids.tolist() == [1, 2, 0]
    np.testing.assert_allclose(top, [0.7, 0.3, 0.1])


def test_topk_k_zero():
    ids, top = topk(np.array([0.1, 0.2], dtype=np.float32), k=0)
    assert ids.size == 0
    assert top.size == 0


def test_score_topk_batch_shapes():
    ip, idx, d = _build_csc(
        [[(0, 1.0)], [(1, 0.5)], [(0, 0.3), (1, 0.4)]],
        vocab=2,
    )
    q_ids_list = [np.array([0], np.int32), np.array([1], np.int32)]
    q_ws_list = [np.array([1.0], np.float32), np.array([1.0], np.float32)]
    ids, scores = score_topk_batch(ip, idx, d, q_ids_list, q_ws_list, n_docs=3, k=2)
    assert ids.shape == (2, 2)
    assert scores.shape == (2, 2)
    assert ids.dtype == np.int32
    assert scores.dtype == np.float32

    # Q=token 0: doc 0=1.0, doc 2=0.3 → [0, 2]
    assert ids[0].tolist() == [0, 2]
    # Q=token 1: doc 2=0.4, doc 1=0.5 → [1, 2]
    assert ids[1].tolist() == [1, 2]
