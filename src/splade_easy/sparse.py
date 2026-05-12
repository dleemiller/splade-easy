"""Minimal CSR sparse container + CSR->CSC transpose. Avoids the scipy dependency."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class SparseCorpus:
    """CSR layout: rows = docs, cols = vocab. Dtype contract: data=float32, indices/indptr=int32."""

    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    n_docs: int
    vocab_size: int

    def __post_init__(self) -> None:
        if self.indptr.shape != (self.n_docs + 1,):
            raise ValueError(f"indptr shape {self.indptr.shape} != (n_docs+1,)")
        if self.indices.shape != self.data.shape:
            raise ValueError("indices and data must have the same shape")
        if int(self.indptr[-1]) != self.indices.shape[0]:
            raise ValueError("indptr[-1] does not match nnz")

    @property
    def nnz(self) -> int:
        return int(self.indices.shape[0])


def from_per_doc(
    token_ids_list: Sequence[np.ndarray],
    weights_list: Sequence[np.ndarray],
    vocab_size: int,
) -> SparseCorpus:
    """Build a CSR SparseCorpus from per-doc (ids, weights) arrays."""
    n_docs = len(token_ids_list)
    if len(weights_list) != n_docs:
        raise ValueError("token_ids_list and weights_list must have same length")

    nnz_per = np.fromiter((len(ids) for ids in token_ids_list), dtype=np.int32, count=n_docs)
    indptr = np.empty(n_docs + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(nnz_per, out=indptr[1:])
    nnz = int(indptr[-1])

    indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float32)
    for i in range(n_docs):
        s, e = int(indptr[i]), int(indptr[i + 1])
        indices[s:e] = token_ids_list[i]
        data[s:e] = weights_list[i]
    return SparseCorpus(
        indptr=indptr, indices=indices, data=data, n_docs=n_docs, vocab_size=vocab_size
    )


def csr_to_csc(sc: SparseCorpus) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transpose CSR (doc-major) to CSC (term-major). Returns (indptr, indices, data)."""
    n_docs = sc.n_docs
    vocab_size = sc.vocab_size
    indices = sc.indices
    data = sc.data
    indptr = sc.indptr

    # row index for each nnz, in row-major order
    nnz_per_row = np.diff(indptr).astype(indices.dtype, copy=False)
    rows = np.repeat(np.arange(n_docs, dtype=indices.dtype), nnz_per_row)

    # stable sort by column index — groups all entries by column, preserving row order
    order = np.argsort(indices, kind="stable")
    indices_c = rows[order]
    data_c = data[order]

    # column pointers via bincount
    col_counts = np.bincount(indices, minlength=vocab_size).astype(np.int32, copy=False)
    indptr_c = np.empty(vocab_size + 1, dtype=np.int32)
    indptr_c[0] = 0
    np.cumsum(col_counts, out=indptr_c[1:])

    return indptr_c, indices_c, data_c
