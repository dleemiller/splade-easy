#!/usr/bin/env python3
"""
Benchmark (OLD API): SpladeIndex/SpladeRetriever memory-only search

Measures:
  - encode: model.encode(query)
  - score_only: retriever.search(tokens, weights, return_text=False)
  - attach_text: retriever.search(tokens, weights, return_text=True)

Fixes included:
  - Robust shard discovery fallback so the old retriever actually scans shards
    even if metadata doesn't list shard hashes (content-addressed files).
  - Warmups to avoid cold-start noise.
"""

import os
import time
import types
import argparse
from pathlib import Path

# Keep tokenizer quiet
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Optional: run fully offline if caches are populated
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import SentenceTransformer
from splade_easy import SpladeIndex
from splade_easy.utils import extract_splade_vectors, get_shard_paths


def report(label: str, secs: float):
    print(f"{label:>12}: {secs * 1000:8.2f} ms")


def time_encode(model, query: str):
    t0 = time.perf_counter()
    enc = model.encode(query, show_progress_bar=False)
    t1 = time.perf_counter()
    return t1 - t0, enc


def patch_shard_paths(retriever):
    """
    Ensure the old retriever can find shards even if metadata lacks shard_hashes.
    """
    def _get_shard_paths(self):
        paths = get_shard_paths(self.index_dir, self.metadata)
        if not paths:
            paths = sorted(self.index_dir.glob("*.fb"))
        return paths
    retriever._get_shard_paths = types.MethodType(_get_shard_paths, retriever)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="./agnews_index", help="Path to index directory")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1, help="num_workers for search")
    parser.add_argument("--queries", nargs="*", default=[
        "machine learning artificial intelligence",
        "sports football basketball",
        "technology computers software",
        "business economy finance",
    ])
    args = parser.parse_args()

    index_path = Path(args.index)
    if not index_path.exists():
        raise SystemExit(f"Index not found: {index_path}")

    print(f"Loading model...")
    model = SentenceTransformer("naver/splade-v3")
    _ = model.encode("warmup", show_progress_bar=False)

    print(f"\nLoading index (old API) from {index_path} ...")
    index = SpladeIndex(str(index_path))
    retriever = SpladeIndex.retriever(str(index_path), mode="memory")

    # Patch shard discovery if metadata is missing/outdated
    patch_shard_paths(retriever)
    shard_count = len(retriever._get_shard_paths())
    print(f"Memory shards (old retriever): {shard_count}")
    if shard_count == 0:
        print("WARNING: no shards visible to old retriever; results will be empty.")

    # Warmup a search path (threadpools/JIT/etc.)
    _ = retriever.search_text("warmup", model=model, top_k=1, return_text=False, num_workers=1)

    print("\n=== OLD API (memory-only benchmark) ===")
    for q in args.queries:
        # 1) Encode (measured)
        enc_time, enc = time_encode(model, q)
        tok, w = extract_splade_vectors(enc)

        # 2) Scoring only (no text)
        t0 = time.perf_counter()
        _ = retriever.search(tok, w, top_k=args.top_k, return_text=False, num_workers=args.workers)
        t1 = time.perf_counter()
        score_time = t1 - t0

        # 3) Attach text
        t2 = time.perf_counter()
        results = retriever.search(tok, w, top_k=args.top_k, return_text=True, num_workers=args.workers)
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


if __name__ == "__main__":
    main()
