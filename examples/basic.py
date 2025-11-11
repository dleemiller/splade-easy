#!/usr/bin/env python3
"""
Benchmark (NEW API): SPLADE-easy Index/IndexReader memory-only search

Measures:
  - encode: model.encode(query)
  - score_only: reader.search_vectors(..., return_text=False)
  - attach_text: reader.search_vectors(..., return_text=True) for top_k

Notes:
  - Forces true memory mode by loading shards into RAM.
  - Avoids network on the timed path (set HF_*_OFFLINE to skip Hub traffic).
  - Uses identical top_k and num_workers across runs for clean comparison.
"""

import argparse
import os
import time
from pathlib import Path

# Keep tokenizer quiet
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Optional: run fully offline if caches are populated
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cProfile
import io
import pstats

from sentence_transformers import SentenceTransformer

import splade_easy as se
from splade_easy.reader import IndexReader
from splade_easy.utils import extract_splade_vectors


def report(label: str, secs: float):
    print(f"{label:>12}: {secs * 1000:8.2f} ms")


def time_encode(model, query: str):
    t0 = time.perf_counter()
    enc = model.encode(query, show_progress_bar=False)
    t1 = time.perf_counter()
    return t1 - t0, enc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="./agnews_index", help="Path to index directory")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1, help="num_workers for search")
    parser.add_argument(
        "--queries",
        nargs="*",
        default=[
            "machine learning artificial intelligence",
            "sports football basketball",
            "technology computers software",
            "business economy finance",
        ],
    )
    args = parser.parse_args()

    index_path = Path(args.index)
    if not index_path.exists():
        raise SystemExit(f"Index not found: {index_path}")

    print(f"Loading model...")
    model = SentenceTransformer("naver/splade-v3")

    # Warm up to avoid lazy loads in the timed sections
    _ = model.encode("warmup", show_progress_bar=False)

    print(f"\nLoading index (new API) from {index_path} ...")
    index = se.Index(str(index_path))
    reader: IndexReader = index.reader(memory=True)  # forces cache load

    # Sanity: how many shards are cached?
    print(f"Memory cache shards: {len(index._cache)}")

    # Warmup a search path (threadpools/JIT/etc.)
    _ = reader.search_text("warmup", model=model, top_k=1, return_text=False, num_workers=1)

    print("\n=== NEW API (memory-only benchmark) ===")
    for q in args.queries:
        # 1) Encode (measured)
        enc_time, enc = time_encode(model, q)
        tok, w = extract_splade_vectors(enc)

        # 2) Scoring only (no text attachment)
        t0 = time.perf_counter()
        _ = reader.search_vectors(
            tok, w, top_k=args.top_k, return_text=False, num_workers=args.workers
        )
        t1 = time.perf_counter()
        score_time = t1 - t0

        # 3) Attach text (retrieve text for top_k)
        t2 = time.perf_counter()
        pr = cProfile.Profile()
        pr.enable()
        results = reader.search_vectors(
            tok, w, top_k=args.top_k, return_text=True, num_workers=args.workers
        )
        pr.disable()
        s = io.StringIO()
        t3 = time.perf_counter()
        attach_time = t3 - t2

        print(f"\nQuery: {q}")
        report("encode", enc_time)
        report("score_only", score_time)
        report("attach_text", attach_time)
        print(f"  results: {len(results)}")
        if results:
            print(f"  top score: {results[0].score:.3f}")
            preview = (results[0].text or "")[:80].replace("\n", " ")
            print(f"  top text: {preview} ...")

        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(30)
        print(s.getvalue())
if __name__ == "__main__":
    main()
