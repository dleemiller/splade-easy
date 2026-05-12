"""Cython _scoring must produce identical output to the numpy reference `_scoring_py`."""

from __future__ import annotations

import numpy as np
import pytest

from splade_easy import _scoring_py as ref
from splade_easy import sparse

cy = pytest.importorskip("splade_easy._scoring", reason="Cython extension not built")


def _random_csc(n_docs: int, vocab: int, nnz_per_doc: int, seed: int):
    rng = np.random.default_rng(seed)
    ids_list = []
    ws_list = []
    for _ in range(n_docs):
        ids = np.unique(rng.integers(0, vocab, size=nnz_per_doc).astype(np.int32))
        ids_list.append(ids)
        ws_list.append(rng.random(size=len(ids), dtype=np.float32) + 0.01)
    sc = sparse.from_per_doc(ids_list, ws_list, vocab_size=vocab)
    return sparse.csr_to_csc(sc)


def _random_query(vocab: int, nq: int, seed: int):
    rng = np.random.default_rng(seed + 1000)
    q_ids = np.unique(rng.integers(0, vocab, size=nq).astype(np.int32))
    q_ws = (rng.random(size=len(q_ids), dtype=np.float32) + 0.1).astype(np.float32)
    return q_ids, q_ws


@pytest.mark.parametrize("seed", range(8))
def test_score_csc_parity(seed):
    n_docs, vocab = 200, 128
    ip, idx, d = _random_csc(n_docs, vocab, nnz_per_doc=30, seed=seed)
    q_ids, q_ws = _random_query(vocab, nq=15, seed=seed)

    s_ref = ref.score_csc(ip, idx, d, q_ids, q_ws, n_docs)
    s_cy = cy.score_csc(ip, idx, d, q_ids, q_ws, n_docs)
    np.testing.assert_allclose(s_ref, s_cy, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("seed", range(8))
def test_topk_parity(seed):
    rng = np.random.default_rng(seed)
    scores = rng.random(size=500, dtype=np.float32)
    ids_ref, sc_ref = ref.topk(scores, k=10)
    ids_cy, sc_cy = cy.topk(scores, k=10)
    np.testing.assert_array_equal(ids_ref, ids_cy)
    np.testing.assert_allclose(sc_ref, sc_cy, rtol=1e-6)


def test_topk_ties_stable():
    """Ties must break deterministically — lower doc index first."""
    scores = np.array([0.5, 0.5, 0.5, 0.1, 0.5], dtype=np.float32)
    ids_ref, _ = ref.topk(scores, k=3)
    ids_cy, _ = cy.topk(scores, k=3)
    assert ids_cy.tolist() == ids_ref.tolist() == [0, 1, 2]


def test_topk_k_zero_and_overflow():
    scores = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    for k in (0, 1, 3, 100):
        ids_ref, sc_ref = ref.topk(scores, k=k)
        ids_cy, sc_cy = cy.topk(scores, k=k)
        np.testing.assert_array_equal(ids_ref, ids_cy)
        np.testing.assert_allclose(sc_ref, sc_cy)


@pytest.mark.parametrize("seed", range(8))
def test_heap_topk_parity_random(seed):
    """heap_topk must match topk on random inputs."""
    rng = np.random.default_rng(seed + 50)
    scores = rng.random(size=500, dtype=np.float32)
    for k in (1, 5, 10, 50, 500, 600):
        ids_ref, sc_ref = cy.topk(scores, k=k)
        ids_h, sc_h = cy.heap_topk(scores, k=k)
        np.testing.assert_array_equal(ids_ref, ids_h, err_msg=f"k={k}")
        np.testing.assert_allclose(sc_ref, sc_h, rtol=1e-6, err_msg=f"k={k}")


def test_heap_topk_all_ties():
    """All-tied scores → lowest doc IDs in ascending order."""
    scores = np.full(20, 0.5, dtype=np.float32)
    ids, sc = cy.heap_topk(scores, k=5)
    assert ids.tolist() == [0, 1, 2, 3, 4]
    np.testing.assert_allclose(sc, [0.5] * 5)


def test_heap_topk_all_zeros():
    """All-zero scores → lowest doc IDs (matches numpy)."""
    scores = np.zeros(20, dtype=np.float32)
    ids_ref, _ = cy.topk(scores, k=10)
    ids_h, _ = cy.heap_topk(scores, k=10)
    np.testing.assert_array_equal(ids_ref, ids_h)


def test_heap_topk_partial_ties():
    """Mixture of unique and tied scores."""
    scores = np.array([0.9, 0.5, 0.5, 0.5, 0.1, 0.7, 0.7], dtype=np.float32)
    ids_ref, sc_ref = cy.topk(scores, k=4)
    ids_h, sc_h = cy.heap_topk(scores, k=4)
    np.testing.assert_array_equal(ids_ref, ids_h)
    np.testing.assert_allclose(sc_ref, sc_h)


def test_heap_topk_k_edge_cases():
    scores = np.array([3.0, 1.0, 2.0], dtype=np.float32)
    for k in (0, 1, 3, 100):
        ids_ref, sc_ref = cy.topk(scores, k=k)
        ids_h, sc_h = cy.heap_topk(scores, k=k)
        np.testing.assert_array_equal(ids_ref, ids_h, err_msg=f"k={k}")
        np.testing.assert_allclose(sc_ref, sc_h, err_msg=f"k={k}")


@pytest.mark.parametrize("seed", range(4))
def test_score_topk_batch_parity(seed):
    n_docs, vocab = 1000, 256
    ip, idx, d = _random_csc(n_docs, vocab, nnz_per_doc=40, seed=seed)
    n_queries = 20
    q_ids_list = []
    q_ws_list = []
    for q in range(n_queries):
        qi, qw = _random_query(vocab, nq=10, seed=seed * 100 + q)
        q_ids_list.append(qi)
        q_ws_list.append(qw)

    ids_ref, sc_ref = ref.score_topk_batch(ip, idx, d, q_ids_list, q_ws_list, n_docs, k=10)
    ids_cy, sc_cy = cy.score_topk_batch(ip, idx, d, q_ids_list, q_ws_list, n_docs, k=10)
    np.testing.assert_array_equal(ids_ref, ids_cy)
    np.testing.assert_allclose(sc_ref, sc_cy, rtol=1e-5, atol=1e-6)
