#!/usr/bin/env python3
import os, time, types, argparse
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
os.environ.setdefault("HF_HUB_OFFLINE","1")
os.environ.setdefault("TRANSFORMERS_OFFLINE","1")

from sentence_transformers import SentenceTransformer
from splade_easy import SpladeIndex
from splade_easy.utils import extract_splade_vectors, get_shard_paths
from splade_easy.scoring import compute_splade_score, ensure_sorted_splade_vector
import heapq

def report(label, secs): print(f"{label:>12}: {secs*1000:8.2f} ms")
def time_encode(model, q):
    t0=time.perf_counter(); enc=model.encode(q, show_progress_bar=False); t1=time.perf_counter()
    return t1-t0, enc

def patch_shard_paths(retriever):
    def _get_shard_paths(self):
        paths = get_shard_paths(self.index_dir, self.metadata)
        if not paths: paths = sorted(self.index_dir.glob("*.fb"))
        return paths
    retriever._get_shard_paths = types.MethodType(_get_shard_paths, retriever)

def patch_memory_loading(retriever):
    # Force memory cache WITHOUT text payloads
    from splade_easy.shard import ShardReader
    def _load_shards_to_memory(self):
        self.shard_cache = {}
        for shard_path in self._get_shard_paths():
            r = ShardReader(str(shard_path))
            self.shard_cache[shard_path] = list(r.scan(load_text=False))
    retriever._load_shards_to_memory = types.MethodType(_load_shards_to_memory, retriever)
    retriever._load_shards_to_memory()

def patch_search_shard(retriever):
    # Respect return_text; keep hot locals; minimal allocs
    def _search_shard(self, shard_path, query_tokens, query_weights, top_k, return_text):
        heap = []
        doc_index = 0
        docs = self.shard_cache.get(shard_path)
        if docs is None:
            from splade_easy.shard import ShardReader
            docs = ShardReader(str(shard_path)).scan(load_text=return_text)

        q_tok, q_w = query_tokens, query_weights
        deleted = self.deleted_ids
        push, replace = heapq.heappush, heapq.heapreplace
        get_score = compute_splade_score

        for doc in docs:
            did = doc["doc_id"]
            if did in deleted: continue
            score = get_score(doc["token_ids"], doc["weights"], q_tok, q_w)
            if score <= 0: continue
            res_text = (doc.get("text") if return_text else None)
            result = self.SearchResult(  # refer to class attached below
                doc_id=did, score=score, metadata=doc["metadata"], text=res_text
            )
            if len(heap) < top_k:
                push(heap, (score, doc_index, result))
            elif score > heap[0][0]:
                replace(heap, (score, doc_index, result))
            doc_index += 1
        # sort by score desc
        return [r for _,_,r in sorted(heap, key=lambda x: x[0], reverse=True)]
    retriever.SearchResult = type("SearchResult", (), {"__init__": lambda self, **kw: self.__dict__.update(kw)})
    retriever._search_shard = types.MethodType(_search_shard, retriever)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="./agnews_index")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--queries", nargs="*", default=[
        "machine learning artificial intelligence",
        "sports football basketball",
        "technology computers software",
        "business economy finance",
    ])
    args = ap.parse_args()

    p = Path(args.index)
    if not p.exists(): raise SystemExit(f"Index not found: {p}")

    print("Loading model...")
    model = SentenceTransformer("naver/splade-v3")
    _ = model.encode("warmup", show_progress_bar=False)

    print(f"\nOpening existing index (old API) at {p} ...")
    index = SpladeIndex(str(p))
    retriever = SpladeIndex.retriever(str(p), mode="memory")

    # Patches: shard discovery + memory load without text + respect return_text
    patch_shard_paths(retriever)
    patch_memory_loading(retriever)
    patch_search_shard(retriever)

    shard_count = len(retriever._get_shard_paths())
    print(f"Memory shards (old retriever): {shard_count}")

    # Warmup path (progress bar may appear once if underlying call shows it)
    _ = retriever.search_text("warmup", model=model, top_k=1, return_text=False, num_workers=1)

    print("\n=== OLD API against AG News index (patched) ===")
    for q in args.queries:
        enc_t, enc = time_encode(model, q)
        tok, w = extract_splade_vectors(enc)
        tok, w = ensure_sorted_splade_vector(tok, w, deduplicate=True)

        t0=time.perf_counter()
        _ = retriever.search(tok, w, top_k=args.top_k, return_text=False, num_workers=args.workers)
        t1=time.perf_counter()
        score_t = t1 - t0

        t2=time.perf_counter()
        results = retriever.search(tok, w, top_k=args.top_k, return_text=True, num_workers=args.workers)
        t3=time.perf_counter()
        attach_t = t3 - t2

        print(f"\nQuery: {q}")
        report("encode", enc_t)
        report("score_only", score_t)
        report("attach_text", attach_t)
        print(f"  results: {len(results)}")
        if results:
            print(f"  top score: {results[0].score:.3f}")
            preview = (results[0].text or "")[:80].replace("\n"," ")
            print(f"  top text: {preview} ...")

if __name__ == "__main__":
    main()
