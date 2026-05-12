"""Query-time latency benchmark.

Generates a synthetic SPLADE-like sparse corpus (no torch dependency), builds the inverted index, then times batched and single-query retrieval. cProfile output saved to /tmp/.

Run:
    uv run python benchmarks/bench_query.py
    uv run python benchmarks/bench_query.py --docs 50000 --queries 1000

cProfile inspection:
    uv run python -c "import pstats; pstats.Stats('/tmp/bench_query.prof').sort_stats('cumulative').print_stats(30)"
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import time
from pathlib import Path

import numpy as np

from splade_easy import _scoring_py as ref
from splade_easy import sparse

try:
    from splade_easy import _scoring as cy
except ImportError:
    cy = None


VOCAB = 30522  # BERT-base vocab size


def make_synthetic_corpus(n_docs: int, vocab: int = VOCAB, seed: int = 0) -> sparse.SparseCorpus:
    """Synthesize a SPLADE-like sparse corpus. Token IDs follow a Zipf distribution; per-doc nnz ~ NB-ish."""
    rng = np.random.default_rng(seed)
    # Zipf weights over vocab — gives a power-law term frequency
    zipf_weights = 1.0 / (np.arange(1, vocab + 1, dtype=np.float64))
    zipf_weights /= zipf_weights.sum()

    nnz_per_doc = rng.integers(80, 250, size=n_docs).astype(np.int32)
    indptr = np.empty(n_docs + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(nnz_per_doc, out=indptr[1:])
    total_nnz = int(indptr[-1])

    indices = np.empty(total_nnz, dtype=np.int32)
    data = np.empty(total_nnz, dtype=np.float32)

    for d in range(n_docs):
        s, e = int(indptr[d]), int(indptr[d + 1])
        nz = e - s
        # Sample unique token IDs weighted by Zipf
        tokens = rng.choice(vocab, size=nz, replace=False, p=zipf_weights)
        tokens.sort()
        indices[s:e] = tokens
        # Weights: roughly exponential, like SPLADE log-saturated activations
        data[s:e] = rng.exponential(scale=0.3, size=nz).astype(np.float32) + 0.05

    return sparse.SparseCorpus(
        indptr=indptr, indices=indices, data=data, n_docs=n_docs, vocab_size=vocab
    )


def make_synthetic_queries(n_queries: int, vocab: int = VOCAB, seed: int = 1) -> list[np.ndarray]:
    """Per-query token IDs (5-20 tokens, Zipf-weighted)."""
    rng = np.random.default_rng(seed)
    zipf_weights = 1.0 / np.arange(1, vocab + 1, dtype=np.float64)
    zipf_weights /= zipf_weights.sum()
    out = []
    for _ in range(n_queries):
        nq = int(rng.integers(5, 21))
        out.append(
            np.unique(rng.choice(vocab, size=nq, replace=True, p=zipf_weights).astype(np.int32))
        )
    return out


def _percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p))


def _time_backend(backend, indptr, indices, data, q_ids, q_weights, n_docs, k):
    """Single-query latencies + a batched timing for one backend."""
    latencies = []
    for qi, qw in zip(q_ids, q_weights, strict=True):
        t = time.perf_counter()
        backend.score_topk_batch(indptr, indices, data, [qi], [qw], n_docs=n_docs, k=k)
        latencies.append((time.perf_counter() - t) * 1000.0)
    t0 = time.perf_counter()
    backend.score_topk_batch(indptr, indices, data, q_ids, q_weights, n_docs=n_docs, k=k)
    batch_ms = (time.perf_counter() - t0) * 1000.0
    return latencies, batch_ms


def bench(n_docs: int, n_queries: int, k: int, prof_path: Path | None) -> dict:
    print(f"Synthesizing {n_docs} docs x vocab {VOCAB}...", flush=True)
    t0 = time.time()
    sc = make_synthetic_corpus(n_docs)
    print(f"  built CSR in {time.time() - t0:.2f}s (nnz={sc.nnz:,})")

    t0 = time.time()
    indptr, indices, data = sparse.csr_to_csc(sc)
    print(f"  CSR->CSC in {time.time() - t0:.2f}s")

    queries = make_synthetic_queries(n_queries)
    rng = np.random.default_rng(2)
    idf = rng.uniform(0.5, 3.0, size=VOCAB).astype(np.float32)
    q_ids = queries
    q_weights = [idf[q] for q in queries]

    backends = [("numpy", ref)]
    if cy is not None:
        backends.append(("cython", cy))

    out = {"n_docs": n_docs, "n_queries": n_queries, "k": k, "backends": {}}
    for name, backend in backends:
        print(f"Timing {name} backend...")
        lat, batch = _time_backend(backend, indptr, indices, data, q_ids, q_weights, n_docs, k)
        out["backends"][name] = {
            "mean_ms": float(np.mean(lat)),
            "p50_ms": _percentile(lat, 50),
            "p95_ms": _percentile(lat, 95),
            "p99_ms": _percentile(lat, 99),
            "batch_per_query_ms": batch / n_queries,
        }

    # cProfile only the Cython backend if available, otherwise numpy
    if prof_path is not None:
        backend = cy or ref
        prof_path.parent.mkdir(parents=True, exist_ok=True)
        prof = cProfile.Profile()
        prof.enable()
        backend.score_topk_batch(indptr, indices, data, q_ids, q_weights, n_docs=n_docs, k=k)
        prof.disable()
        prof.dump_stats(str(prof_path))
        print(f"  cProfile written to {prof_path}")

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--docs", type=int, default=10000)
    p.add_argument("--queries", type=int, default=500)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--profile", default="/tmp/bench_query.prof")
    args = p.parse_args()

    result = bench(args.docs, args.queries, args.k, Path(args.profile) if args.profile else None)
    print()
    print(f"  docs={result['n_docs']:,}  queries={result['n_queries']:,}  k={result['k']}")
    print()
    header = f"  {'backend':<10} {'mean ms':>9} {'p50':>9} {'p95':>9} {'p99':>9} {'batch ms/q':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, b in result["backends"].items():
        print(
            f"  {name:<10} {b['mean_ms']:>9.3f} {b['p50_ms']:>9.3f} {b['p95_ms']:>9.3f}"
            f" {b['p99_ms']:>9.3f} {b['batch_per_query_ms']:>12.3f}"
        )

    backends = result["backends"]
    if "cython" in backends and "numpy" in backends:
        speedup = backends["numpy"]["batch_per_query_ms"] / backends["cython"]["batch_per_query_ms"]
        print(f"\n  cython/numpy speedup: {speedup:.2f}x")

    if args.profile:
        print()
        print("--- top 20 by cumulative time ---")
        s = pstats.Stats(args.profile)
        s.sort_stats("cumulative").print_stats(20)


if __name__ == "__main__":
    main()
