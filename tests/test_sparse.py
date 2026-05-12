"""Tests for the sparse-matrix container + CSR->CSC transpose."""
from __future__ import annotations

import numpy as np
import pytest

from splade_easy import sparse


def _dense_from_csr(sc: sparse.SparseCorpus) -> np.ndarray:
    out = np.zeros((sc.n_docs, sc.vocab_size), dtype=np.float32)
    for d in range(sc.n_docs):
        s, e = int(sc.indptr[d]), int(sc.indptr[d + 1])
        out[d, sc.indices[s:e]] = sc.data[s:e]
    return out


def _dense_from_csc(indptr, indices, data, vocab_size, n_docs) -> np.ndarray:
    out = np.zeros((n_docs, vocab_size), dtype=np.float32)
    for t in range(vocab_size):
        s, e = int(indptr[t]), int(indptr[t + 1])
        out[indices[s:e], t] = data[s:e]
    return out


def test_from_per_doc_simple():
    ids = [np.array([1, 3], dtype=np.int32), np.array([0, 2], dtype=np.int32)]
    ws = [np.array([0.5, 0.2], dtype=np.float32), np.array([0.4, 0.3], dtype=np.float32)]
    sc = sparse.from_per_doc(ids, ws, vocab_size=4)

    assert sc.n_docs == 2
    assert sc.vocab_size == 4
    assert sc.indptr.tolist() == [0, 2, 4]
    assert sc.indices.tolist() == [1, 3, 0, 2]
    assert sc.data.dtype == np.float32
    np.testing.assert_allclose(sc.data, [0.5, 0.2, 0.4, 0.3])


def test_from_per_doc_empty_doc():
    ids = [np.array([1, 2], dtype=np.int32), np.array([], dtype=np.int32)]
    ws = [np.array([0.5, 0.5], dtype=np.float32), np.array([], dtype=np.float32)]
    sc = sparse.from_per_doc(ids, ws, vocab_size=4)
    assert sc.indptr.tolist() == [0, 2, 2]
    assert sc.indices.tolist() == [1, 2]


def test_csr_to_csc_roundtrip_random():
    rng = np.random.default_rng(0)
    n_docs, vocab = 50, 32
    rows = rng.integers(0, n_docs, size=300)
    cols = rng.integers(0, vocab, size=300)
    vals = rng.random(size=300, dtype=np.float32)
    # Deduplicate (row, col) pairs — sparse repr has no duplicates
    seen = {}
    for r, c, v in zip(rows, cols, vals, strict=True):
        seen[(int(r), int(c))] = float(v)

    ids_list = []
    ws_list = []
    for d in range(n_docs):
        entries = sorted(c for (r, c) in seen if r == d)
        ids_list.append(np.array(entries, dtype=np.int32))
        ws_list.append(np.array([seen[(d, c)] for c in entries], dtype=np.float32))

    sc = sparse.from_per_doc(ids_list, ws_list, vocab_size=vocab)
    dense_csr = _dense_from_csr(sc)

    ip, idx, d = sparse.csr_to_csc(sc)
    dense_csc = _dense_from_csc(ip, idx, d, vocab, n_docs)

    np.testing.assert_allclose(dense_csr, dense_csc)


def test_csr_to_csc_empty_columns():
    """A vocab token with no docs should still appear in indptr with zero range."""
    ids = [np.array([0, 5], dtype=np.int32)]
    ws = [np.array([1.0, 1.0], dtype=np.float32)]
    sc = sparse.from_per_doc(ids, ws, vocab_size=8)
    ip, idx, d = sparse.csr_to_csc(sc)
    assert ip.shape == (9,)
    # tokens 1,2,3,4,6,7 have no docs
    for t in (1, 2, 3, 4, 6, 7):
        assert ip[t] == ip[t + 1]


def test_validation_bad_shapes():
    with pytest.raises(ValueError):
        sparse.SparseCorpus(
            indptr=np.array([0, 1], dtype=np.int32),
            indices=np.array([0], dtype=np.int32),
            data=np.array([1.0, 2.0], dtype=np.float32),  # mismatched
            n_docs=1,
            vocab_size=5,
        )
