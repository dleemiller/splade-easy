"""NanoBEIR evaluation runner.

NanoBEIR is a set of small BEIR subsets (~50 queries each) intended for fast iteration on retrieval models. We use the `zeta-alpha-ai/Nano*` HuggingFace datasets — each ships `corpus`, `queries`, and `qrels` configs.

Usage:
    uv run splade-eval-nanobeir --datasets nq,fiqa,scifact
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Iterable

import numpy as np

from .. import SpladeRetriever, encode_corpus
from . import metrics


NANOBEIR_REPOS: dict[str, str] = {
    "nq": "zeta-alpha-ai/NanoNQ",
    "fiqa": "zeta-alpha-ai/NanoFiQA2018",
    "scifact": "zeta-alpha-ai/NanoSciFact",
    "msmarco": "zeta-alpha-ai/NanoMSMARCO",
    "hotpotqa": "zeta-alpha-ai/NanoHotpotQA",
    "trec-covid": "zeta-alpha-ai/NanoTRECCovid",
    "arguana": "zeta-alpha-ai/NanoArguAna",
    "dbpedia": "zeta-alpha-ai/NanoDBPedia",
    "fever": "zeta-alpha-ai/NanoFEVER",
    "climate-fever": "zeta-alpha-ai/NanoClimateFEVER",
    "nfcorpus": "zeta-alpha-ai/NanoNFCorpus",
    "quora": "zeta-alpha-ai/NanoQuoraRetrieval",
    "scidocs": "zeta-alpha-ai/NanoSCIDOCS",
    "touche": "zeta-alpha-ai/NanoTouche2020",
}


def _row_field(row: dict, *candidates: str) -> str:
    for c in candidates:
        if c in row:
            return c
    raise KeyError(f"None of {candidates} found in row with keys {list(row.keys())}")


def load_nanobeir(name: str) -> tuple[list[str], dict[str, str], dict[str, dict[int, int]]]:
    """Returns (corpus_texts, queries_by_qid, qrels_by_qid). Doc IDs are positional indices into `corpus_texts`."""
    if name not in NANOBEIR_REPOS:
        raise KeyError(f"Unknown NanoBEIR dataset {name!r}. Options: {list(NANOBEIR_REPOS)}")
    repo = NANOBEIR_REPOS[name]

    from datasets import load_dataset

    corpus_ds = load_dataset(repo, "corpus", split="train")
    queries_ds = load_dataset(repo, "queries", split="train")
    qrels_ds = load_dataset(repo, "qrels", split="train")

    # Build corpus list — preserve order; map external doc_id -> position
    corpus_id_field = _row_field(corpus_ds[0], "_id", "id", "doc_id")
    corpus_text_field = _row_field(corpus_ds[0], "text", "document", "body")
    doc_id_to_idx: dict[str, int] = {}
    texts: list[str] = []
    for i, row in enumerate(corpus_ds):
        doc_id_to_idx[str(row[corpus_id_field])] = i
        texts.append(str(row[corpus_text_field]))

    qid_field = _row_field(queries_ds[0], "_id", "id", "query_id")
    qtext_field = _row_field(queries_ds[0], "text", "query")
    queries: dict[str, str] = {str(row[qid_field]): str(row[qtext_field]) for row in queries_ds}

    qrel_qid_field = _row_field(qrels_ds[0], "query-id", "query_id", "qid")
    qrel_did_field = _row_field(qrels_ds[0], "corpus-id", "doc_id", "docid", "corpus_id")
    qrel_score_field = None
    for c in ("score", "relevance", "label"):
        if c in qrels_ds[0]:
            qrel_score_field = c
            break

    qrels: dict[str, dict[int, int]] = {}
    for row in qrels_ds:
        qid = str(row[qrel_qid_field])
        did = str(row[qrel_did_field])
        if did not in doc_id_to_idx:
            continue
        grade = int(row[qrel_score_field]) if qrel_score_field else 1
        qrels.setdefault(qid, {})[doc_id_to_idx[did]] = grade

    return texts, queries, qrels


def run_eval(
    name: str,
    model: str | None = None,
    k: int = 10,
    batch_size: int = 32,
    device: str | None = None,
) -> dict:
    """Encode the NanoBEIR corpus, index, retrieve all queries, return metrics."""
    t0 = time.time()
    texts, queries, qrels = load_nanobeir(name)
    print(
        f"[{name}] corpus={len(texts)} queries={len(queries)} qrels_queries={len(qrels)}",
        file=sys.stderr,
    )

    sparse_docs = encode_corpus(
        texts, model=model, batch_size=batch_size, device=device, show_progress=False
    )
    print(
        f"[{name}] encoded in {time.time() - t0:.1f}s (nnz={sparse_docs.nnz})",
        file=sys.stderr,
    )

    retriever = SpladeRetriever(model=model)
    retriever.index(sparse_docs)

    qids = list(queries.keys())
    query_texts = [queries[qid] for qid in qids]
    t1 = time.time()
    ranked, _scores = retriever.retrieve(query_texts, k=k)
    dt = time.time() - t1
    print(
        f"[{name}] retrieved {len(qids)} queries in {dt:.2f}s ({dt / max(len(qids), 1) * 1000:.2f} ms/query)",
        file=sys.stderr,
    )

    rankings = {qid: ranked[i] for i, qid in enumerate(qids)}
    return metrics.evaluate(rankings, qrels, k=k)


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="NanoBEIR evaluation for splade-easy")
    p.add_argument(
        "--datasets",
        default="nq,fiqa,scifact",
        help="Comma-separated NanoBEIR dataset names (see NANOBEIR_REPOS)",
    )
    p.add_argument("--model", default=None, help="Override the doc model id")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None, help="torch device (cpu, cuda, mps, ...)")
    p.add_argument("--json", action="store_true", help="Emit results as JSON on stdout")
    args = p.parse_args(list(argv) if argv is not None else None)

    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    results: dict[str, dict] = {}
    for name in names:
        results[name] = run_eval(
            name,
            model=args.model,
            k=args.k,
            batch_size=args.batch_size,
            device=args.device,
        )
        print(f"  {name}: {results[name]}", file=sys.stderr)

    keys = ["ndcg@k", "recall@k", "mrr@k"]
    mean = {k_: float(np.mean([r[k_] for r in results.values()])) for k_ in keys}

    if args.json:
        json.dump({"per_dataset": results, "mean": mean}, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print()
        print(f"{'dataset':<16} {'NDCG@k':>8} {'Recall@k':>10} {'MRR@k':>8}")
        for name, r in results.items():
            print(f"{name:<16} {r['ndcg@k']:>8.4f} {r['recall@k']:>10.4f} {r['mrr@k']:>8.4f}")
        print(f"{'MEAN':<16} {mean['ndcg@k']:>8.4f} {mean['recall@k']:>10.4f} {mean['mrr@k']:>8.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
