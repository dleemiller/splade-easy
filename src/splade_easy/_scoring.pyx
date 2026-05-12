"""Cython hot path for SPLADE scoring + top-k. Mirrors `_scoring_py` exactly.

Compiler directives (boundscheck, wraparound, cdivision, initializedcheck) are set
via cythonize() in setup.py, not in this file.
"""

import numpy as np

cimport numpy as cnp
from libc.stdint cimport int32_t
from libc.string cimport memset


def score_csc(indptr, indices, data, q_ids, q_weights, int n_docs):
    """Score all docs for one query against a CSC inverted index."""
    cdef const int32_t[::1] indptr_mv = indptr
    cdef const int32_t[::1] indices_mv = indices
    cdef const float[::1] data_mv = data
    cdef const int32_t[::1] q_ids_mv = q_ids
    cdef const float[::1] q_weights_mv = q_weights

    cdef cnp.ndarray[float, ndim=1] scores_arr = np.zeros(n_docs, dtype=np.float32)
    cdef float[::1] scores = scores_arr

    cdef Py_ssize_t i, j, s, e
    cdef int32_t tid
    cdef float qw
    cdef Py_ssize_t n_q = q_ids_mv.shape[0]

    for i in range(n_q):
        tid = q_ids_mv[i]
        qw = q_weights_mv[i]
        s = indptr_mv[tid]
        e = indptr_mv[tid + 1]
        for j in range(s, e):
            scores[indices_mv[j]] += qw * data_mv[j]

    return scores_arr


def topk(scores, int k):
    """Top-k indices/scores by descending score. Stable on ties (lower index first).

    Numpy-backed reference. Kept for direct testing + parity tests against `heap_topk`.
    """
    scores = np.ascontiguousarray(scores, dtype=np.float32)
    cdef Py_ssize_t n = scores.shape[0]
    if k > n:
        k = n
    if k <= 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

    cdef cnp.ndarray neg = -scores
    if k == n:
        order = np.argsort(neg, kind='stable')
    else:
        part = np.argpartition(neg, k - 1)[:k]
        order = part[np.argsort(neg[part], kind='stable')]
    return order.astype(np.int32, copy=False), scores[order]


# ---------- min-heap top-k ----------
# Ordering: higher score is better; on ties, lower doc_id wins (matches numpy
# stable-argsort semantics). The heap holds the WORST entry currently in top-k
# at its root, so we only need one compare per candidate to decide eviction.


cdef inline bint _hp_better(
    float s_a, int32_t d_a, float s_b, int32_t d_b,
) noexcept nogil:
    if s_a > s_b:
        return True
    if s_a < s_b:
        return False
    return d_a < d_b


cdef inline bint _hp_worse(
    float s_a, int32_t d_a, float s_b, int32_t d_b,
) noexcept nogil:
    if s_a < s_b:
        return True
    if s_a > s_b:
        return False
    return d_a > d_b


cdef inline void _hp_sift_up(float[::1] hs, int32_t[::1] hd, int i) noexcept nogil:
    cdef float s = hs[i]
    cdef int32_t d = hd[i]
    cdef int parent
    while i > 0:
        parent = (i - 1) >> 1
        if _hp_worse(s, d, hs[parent], hd[parent]):
            hs[i] = hs[parent]
            hd[i] = hd[parent]
            i = parent
        else:
            break
    hs[i] = s
    hd[i] = d


cdef inline void _hp_sift_down(float[::1] hs, int32_t[::1] hd, int k) noexcept nogil:
    cdef int i = 0, worse_child, child_l, child_r
    cdef float s = hs[0]
    cdef int32_t d = hd[0]
    while True:
        child_l = (i << 1) + 1
        if child_l >= k:
            break
        child_r = child_l + 1
        worse_child = child_l
        if (child_r < k and
                _hp_worse(hs[child_r], hd[child_r], hs[child_l], hd[child_l])):
            worse_child = child_r
        if _hp_worse(hs[worse_child], hd[worse_child], s, d):
            hs[i] = hs[worse_child]
            hd[i] = hd[worse_child]
            i = worse_child
        else:
            break
    hs[i] = s
    hd[i] = d


def heap_topk(scores, int k):
    """Top-k via min-heap of size k over the full scores array.

    Same semantics as `topk`: descending by score, stable on ties (lower doc_id first).
    Single tight C loop with one compare per element — ~2x faster than argpartition
    for k <<< n at the corpus sizes we target.
    """
    cdef const float[::1] scores_mv = np.ascontiguousarray(scores, dtype=np.float32)
    cdef Py_ssize_t n = scores_mv.shape[0]
    cdef int k_eff = k if k < n else <int>n
    if k_eff <= 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

    cdef cnp.ndarray[float, ndim=1] hs_arr = np.empty(k_eff, dtype=np.float32)
    cdef cnp.ndarray[int32_t, ndim=1] hd_arr = np.empty(k_eff, dtype=np.int32)
    cdef float[::1] hs = hs_arr
    cdef int32_t[::1] hd = hd_arr

    cdef Py_ssize_t i
    cdef float sc

    # Fill: first k_eff entries go straight in, then sift up.
    for i in range(k_eff):
        hs[i] = scores_mv[i]
        hd[i] = <int32_t>i
        _hp_sift_up(hs, hd, <int>i)

    # Remaining: replace root iff strictly better.
    for i in range(k_eff, n):
        sc = scores_mv[i]
        if _hp_better(sc, <int32_t>i, hs[0], hd[0]):
            hs[0] = sc
            hd[0] = <int32_t>i
            _hp_sift_down(hs, hd, k_eff)

    # Heap-sort descending: pop root k_eff times, placing each at the back.
    cdef cnp.ndarray[int32_t, ndim=1] out_ids = np.empty(k_eff, dtype=np.int32)
    cdef cnp.ndarray[float, ndim=1] out_scores = np.empty(k_eff, dtype=np.float32)
    cdef int32_t[::1] out_ids_mv = out_ids
    cdef float[::1] out_scores_mv = out_scores
    cdef int heap_size = k_eff
    cdef int pos
    for pos in range(k_eff - 1, -1, -1):
        out_scores_mv[pos] = hs[0]
        out_ids_mv[pos] = hd[0]
        heap_size -= 1
        if heap_size > 0:
            hs[0] = hs[heap_size]
            hd[0] = hd[heap_size]
            _hp_sift_down(hs, hd, heap_size)
    return out_ids, out_scores


def score_topk_batch(
    indptr,
    indices,
    data,
    list q_ids_list,
    list q_weights_list,
    int n_docs,
    int k,
):
    """Score+top-k over a batch of queries; re-uses a single scores buffer."""
    cdef const int32_t[::1] indptr_mv = indptr
    cdef const int32_t[::1] indices_mv = indices
    cdef const float[::1] data_mv = data

    cdef Py_ssize_t n_queries = len(q_ids_list)
    cdef int k_eff = k if k < n_docs else n_docs
    if k_eff < 0:
        k_eff = 0

    out_ids = np.empty((n_queries, k_eff), dtype=np.int32)
    out_scores = np.empty((n_queries, k_eff), dtype=np.float32)

    cdef cnp.ndarray[float, ndim=1] scores_buf = np.empty(n_docs, dtype=np.float32)
    cdef float[::1] scores = scores_buf

    cdef const int32_t[::1] qids_mv
    cdef const float[::1] qws_mv
    cdef Py_ssize_t qi, i, j, s, e, n_q
    cdef int32_t tid
    cdef float qw

    for qi in range(n_queries):
        memset(<void*>&scores[0], 0, n_docs * sizeof(float))

        qids_mv = q_ids_list[qi]
        qws_mv = q_weights_list[qi]
        n_q = qids_mv.shape[0]
        for i in range(n_q):
            tid = qids_mv[i]
            qw = qws_mv[i]
            s = indptr_mv[tid]
            e = indptr_mv[tid + 1]
            for j in range(s, e):
                scores[indices_mv[j]] += qw * data_mv[j]

        ids_arr, scs_arr = heap_topk(scores_buf, k_eff)
        out_ids[qi, :] = ids_arr
        out_scores[qi, :] = scs_arr

    return out_ids, out_scores
