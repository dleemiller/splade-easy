#!/usr/bin/env python3

import shutil
import time
from pathlib import Path
from typing import Dict, List

from sentence_transformers import SentenceTransformer
from datasets import load_dataset

from splade_easy import Index
from splade_easy.reader import IndexReader


class ShardSizeBenchmark:
    def __init__(self, model_name: str = "naver/splade-v3"):
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model.show_progress_bar = False
        self._encoded_docs = None  # Cache for pre-encoded documents
        
    def pre_encode_documents(self, docs: List[str]) -> List[Dict]:
        """Pre-encode all documents once for reuse across shard sizes"""
        print(f"Pre-encoding {len(docs)} documents...")
        
        encoded_docs = []
        batch_size = 100
        
        for batch_start in range(0, len(docs), batch_size):
            batch_docs = docs[batch_start:batch_start + batch_size]
            
            # Batch encode all docs at once
            encodings = self.model.encode(batch_docs, convert_to_tensor=False, show_progress_bar=False)
            
            # Extract SPLADE vectors
            from splade_easy.utils import extract_splade_vectors
            for i, (doc, encoding) in enumerate(zip(batch_docs, encodings)):
                doc_idx = batch_start + i
                token_ids, weights = extract_splade_vectors(encoding)
                encoded_docs.append({
                    "doc_id": f"doc_{doc_idx}",
                    "text": doc,
                    "metadata": {"idx": str(doc_idx)},
                    "token_ids": token_ids,
                    "weights": weights
                })
            
            if batch_start + batch_size <= len(docs):
                print(f"  Encoded {batch_start + batch_size}/{len(docs)} docs...")
        
        print(f"✓ Pre-encoding complete: {len(encoded_docs)} documents")
        self._encoded_docs = encoded_docs
        return encoded_docs
        
    def benchmark_shard_size(
        self, 
        shard_size_mb: int,
        base_dir: str = "./shard_benchmark"
    ) -> Dict:
        """Benchmark search with specific shard size using pre-encoded docs"""
        if self._encoded_docs is None:
            raise ValueError("Must call pre_encode_documents() first")
            
        index_dir = f"{base_dir}/shard_{shard_size_mb}mb"
        shutil.rmtree(index_dir, ignore_errors=True)
        
        print(f"\n=== Testing shard size: {shard_size_mb}MB ===")
        
        # Index pre-encoded documents (much faster!)
        index = Index(index_dir)
        
        with index.writer(shard_size_mb=shard_size_mb) as writer:
            writer.set_model(self.model)
            
            for doc in self._encoded_docs:
                writer.insert(
                    doc["doc_id"],
                    doc["text"], 
                    doc["metadata"],
                    token_ids=doc["token_ids"],
                    weights=doc["weights"]
                )
                    
            writer._finalize_shard()
            
        # Refresh and get stats
        index.refresh()
        stats = index.stats
        
        # Load into memory for search testing
        print("  Loading into memory...")
        index_mem = Index(index_dir, memory=True).load()
        
        # Debug cache state
        print(f"  Cache loaded: {len(index_mem._cache)} shards")
        if index_mem._cache:
            total_cached_docs = sum(len(docs) for docs in index_mem._cache.values())
            print(f"  Total cached docs: {total_cached_docs}")
            for shard_path, docs in list(index_mem._cache.items())[:3]:  # Show first 3
                print(f"    {shard_path.name}: {len(docs)} docs")
        else:
            print("  WARNING: Cache is empty!")
        
        reader = IndexReader(index_mem, num_workers=4)
        
        # Pre-encode queries ONCE outside the timing loop
        queries = [
            "sports news basketball", 
            "business report economy", 
            "technology update software", 
            "world politics government",
            "science discovery research"
        ]
        
        print("  Pre-encoding queries...")
        encoded_queries = []
        for query in queries:
            encoding = self.model.encode(query, show_progress_bar=False)
            from splade_easy.utils import extract_splade_vectors
            query_tokens, query_weights = extract_splade_vectors(encoding)
            encoded_queries.append((query_tokens, query_weights))
        
        # Warmup
        print("  Warming up...")
        for query_tokens, query_weights in encoded_queries[:3]:
            _ = reader.search_vectors(query_tokens, query_weights, top_k=10)
        
        print("  Running SPLADE vector search benchmark...")
        search_times = []
        for query_tokens, query_weights in encoded_queries * 10:  # 50 queries total
            start = time.perf_counter()
            results = reader.search_vectors(query_tokens, query_weights, top_k=10)
            search_times.append(time.perf_counter() - start)
        
        avg_search_time = sum(search_times) / len(search_times)
        
        return {
            "shard_size_mb": shard_size_mb,
            "avg_search_ms": avg_search_time * 1000,
            "num_shards": stats["num_shards"],
            "total_size_mb": stats.get("total_size_mb", 0),
            "docs_per_shard": len(self._encoded_docs) / max(stats["num_shards"], 1),
        }


def load_agnews_sample(max_docs: int = 10000) -> List[str]:
    """Load AG News dataset sample"""
    print(f"Loading AG News dataset (max {max_docs} docs)...")
    ds = load_dataset("fancyzhx/ag_news", split="train")
    if max_docs:
        ds = ds.select(range(min(max_docs, len(ds))))
    
    texts = [row["text"] for row in ds]
    print(f"Loaded {len(texts)} documents")
    print(f"Sample doc: {texts[0][:100]}...")
    
    return texts


def main():
    print("SPLADE-Easy Shard Size Benchmark")
    print("=" * 50)
    
    # Load dataset
    docs = load_agnews_sample(1000)
    
    # Test different shard sizes
    shard_sizes = [1, 2, 4, 8, 16, 32, 64, 128]  # MB
    benchmark = ShardSizeBenchmark()
    
    # Pre-encode documents once
    benchmark.pre_encode_documents(docs)
    
    results = []
    for size in shard_sizes:
        result = benchmark.benchmark_shard_size(size)
        results.append(result)
        
        print(f"  Search time: {result['avg_search_ms']:.1f}ms")
        print(f"  Shards: {result['num_shards']}")
        print(f"  Docs/shard: {result['docs_per_shard']:.0f}")
    
    # Print summary table
    print(f"\n{'='*60}")
    print("SHARD SIZE BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"{'Size(MB)':<8} {'Shards':<7} {'Docs/Shard':<11} {'Search(ms)':<10}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['shard_size_mb']:<8} {r['num_shards']:<7} {r['docs_per_shard']:<11.0f} "
              f"{r['avg_search_ms']:<10.1f}")
    
    # Find optimal shard size
    best_search = min(results, key=lambda x: x['avg_search_ms'])
    
    print(f"\nOptimal for search speed: {best_search['shard_size_mb']}MB "
          f"({best_search['avg_search_ms']:.1f}ms)")
    
    # Cleanup
    shutil.rmtree("./shard_benchmark", ignore_errors=True)
    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
