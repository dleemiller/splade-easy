"""Compare bm25s vs splade-easy (two SPLADE models) on NanoBEIR.

Reports per-dataset NDCG@10 / Recall@10 / MRR@10 and end-to-end query latency
(p50 / p95 ms) on a single CPU thread. Each retriever's full `retrieve(text, k=10)`
path is timed — tokenizing + scoring + top-k — i.e. what a user actually pays.

Run:
    uv run python benchmarks/bench_compare.py
    uv run python benchmarks/bench_compare.py --datasets scifact,nq
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable

import bm25s
import numpy as np
from Stemmer import Stemmer

from splade_easy import SpladeRetriever, encode_corpus
from splade_easy.eval.metrics import evaluate
from splade_easy.eval.nanobeir import load_nanobeir

DATASETS_DEFAULT = ["scifact", "nq", "fiqa", "nfcorpus"]
DISTILL = "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill"
GTE = "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte"
WARMUP = 5


def _run_and_time(
    qids: list[str],
    qtexts: list[str],
    call: Callable[[str], np.ndarray],
) -> tuple[dict[str, np.ndarray], list[float]]:
    """Warm up briefly, then time each retrieve() call. Returns (rankings, latencies_ms)."""
    for q in qtexts[: min(WARMUP, len(qtexts))]:
        call(q)
    rankings: dict[str, np.ndarray] = {}
    lats: list[float] = []
    for qid, q in zip(qids, qtexts, strict=True):
        t = time.perf_counter()
        ranking = call(q)
        lats.append((time.perf_counter() - t) * 1000.0)
        rankings[qid] = ranking
    return rankings, lats


def bench_bm25s(texts, queries, qrels, k=10):
    stemmer = Stemmer("english")
    corpus_tok = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tok, show_progress=False)

    def call(q: str) -> np.ndarray:
        toks = bm25s.tokenize([q], stopwords="en", stemmer=stemmer, show_progress=False)
        results, _ = retriever.retrieve(toks, k=k, show_progress=False)
        return results[0]

    qids = list(queries.keys())
    qtexts = [queries[qid] for qid in qids]
    rankings, lats = _run_and_time(qids, qtexts, call)
    return evaluate(rankings, qrels, k=k), lats


def bench_splade(texts, queries, qrels, model: str, k=10, device: str | None = None):
    # trust_remote_code, code_revision, and the position_ids / weight-tying
    # patches for the gte backbone are all auto-applied by encode_corpus based
    # on the known-models registry.
    sparse_docs = encode_corpus(
        texts,
        model=model,
        batch_size=32,
        show_progress=False,
        device=device,
    )
    retriever = SpladeRetriever(model=model)
    retriever.index(sparse_docs)

    def call(q: str) -> np.ndarray:
        ranking, _ = retriever.retrieve(q, k=k)
        return ranking

    qids = list(queries.keys())
    qtexts = [queries[qid] for qid in qids]
    rankings, lats = _run_and_time(qids, qtexts, call)
    return evaluate(rankings, qrels, k=k), lats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(DATASETS_DEFAULT))
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args(argv)

    rows: list[dict] = []
    names = [n.strip() for n in args.datasets.split(",") if n.strip()]

    backends: list[tuple[str, Callable]] = [
        ("bm25s", lambda t, q, r: bench_bm25s(t, q, r, args.k)),
        ("splade-distill", lambda t, q, r: bench_splade(t, q, r, DISTILL, args.k)),
        ("splade-gte", lambda t, q, r: bench_splade(t, q, r, GTE, args.k)),
    ]

    for ds in names:
        print(f"\n=== {ds} ===", file=sys.stderr, flush=True)
        texts, queries, qrels = load_nanobeir(ds)
        print(f"  corpus={len(texts)} queries={len(queries)}", file=sys.stderr)
        for name, fn in backends:
            print(f"  [{name}] running...", file=sys.stderr, flush=True)
            metrics, lats = fn(texts, queries, qrels)
            rows.append(
                {
                    "dataset": ds,
                    "backend": name,
                    **metrics,
                    "p50_ms": float(np.percentile(lats, 50)),
                    "p95_ms": float(np.percentile(lats, 95)),
                }
            )

    print()
    print("| dataset | model | NDCG@10 | Recall@10 | MRR@10 | p50 ms/q | p95 ms/q |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['dataset']} | {r['backend']} | {r['ndcg@k']:.3f} | "
            f"{r['recall@k']:.3f} | {r['mrr@k']:.3f} | "
            f"{r['p50_ms']:.3f} | {r['p95_ms']:.3f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
