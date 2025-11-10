"""
Performance profiling for SPLADE-Easy (new API)
"""

import shutil
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from splade_easy import Index


def generate_synthetic_docs(num_docs: int) -> list[str]:
    """Generate synthetic documents for testing"""
    topics = [
        "machine learning algorithms and neural networks",
        "python programming and software development",
        "natural language processing and text analysis",
        "computer vision and image recognition",
        "database design and query optimization",
        "web development frameworks and APIs",
        "cybersecurity threats and protection methods",
        "cloud computing and distributed systems",
        "mobile application development",
        "blockchain technology and cryptocurrencies",
    ]

    docs = []
    for i in range(num_docs):
        topic = topics[i % len(topics)]
        doc = f"Document {i}: {topic}. " * 10  # Make docs longer
        docs.append(doc)

    return docs


def profile_indexing(docs: list[str], index_dir: str, model) -> dict:
    """Profile document indexing performance"""
    print(f"Profiling indexing of {len(docs)} documents...")

    # Clean up any existing index
    shutil.rmtree(index_dir, ignore_errors=True)

    index = Index(index_dir)

    start_time = time.time()

    with index.writer(shard_size_mb=32) as writer:
        writer.set_model(model)

        batch_start = time.time()
        for i, text in enumerate(docs):
            writer.insert(
                doc_id=f"doc_{i}", text=text, metadata={"batch": str(i // 100), "index": str(i)}
            )

            # Progress every 100 docs
            if (i + 1) % 100 == 0:
                batch_time = time.time() - batch_start
                print(f"  Indexed {i + 1} docs, last 100 took {batch_time:.2f}s")
                batch_start = time.time()

        # Force finalize any pending shard
        writer._finalize_shard()

    total_time = time.time() - start_time
    docs_per_sec = len(docs) / total_time

    # Refresh index to see finalized shards
    index.refresh()
    stats = index.stats

    print(f"  Debug: After indexing - num_docs: {len(index)}, shards: {stats['num_shards']}")

    return {
        "total_time": total_time,
        "docs_per_sec": docs_per_sec,
        "num_docs": len(index),
        "index_size_mb": stats.get("total_size_mb", 0),
        "num_shards": stats["num_shards"],
    }


def profile_search(index_dir: str, model, num_queries: int = 100) -> dict:
    """Profile search performance"""
    print(f"Profiling {num_queries} search queries...")

    queries = [
        "machine learning algorithms",
        "python programming tutorial",
        "natural language processing",
        "computer vision applications",
        "database design patterns",
        "web development frameworks",
        "cybersecurity best practices",
        "cloud computing architecture",
        "mobile app development",
        "blockchain technology",
    ]

    index = Index(index_dir, memory=False)

    search_times = []
    total_results = 0

    reader = index.reader()  # Remove context manager
    for i in range(num_queries):
        query = queries[i % len(queries)]

        start_time = time.time()
        results = reader.search_text(query, model=model, top_k=10)
        search_time = time.time() - start_time

        search_times.append(search_time)
        total_results += len(results)

        if (i + 1) % 20 == 0:
            avg_time = np.mean(search_times[-20:])
            print(f"  Completed {i + 1} queries, avg time: {avg_time * 1000:.1f}ms")

    return {
        "num_queries": num_queries,
        "total_time": sum(search_times),
        "avg_query_time": np.mean(search_times),
        "min_query_time": min(search_times),
        "max_query_time": max(search_times),
        "queries_per_sec": num_queries / sum(search_times),
        "total_results": total_results,
    }


def profile_memory_vs_disk(index_dir: str, model, num_queries: int = 50):
    """Compare memory vs disk mode performance"""
    print("Comparing memory vs disk mode...")

    queries = ["machine learning", "python programming", "data science"]

    # Test disk mode
    print("  Testing disk mode...")
    index_disk = Index(index_dir, memory=False)

    disk_times = []
    reader = index_disk.reader()  # Remove context manager
    for i in range(num_queries):
        query = queries[i % len(queries)]
        start_time = time.time()
        results = reader.search_text(query, model=model, top_k=5)
        disk_times.append(time.time() - start_time)

    # Test memory mode
    print("  Testing memory mode...")
    index_memory = Index(index_dir, memory=True).load()  # Load into memory
    memory_times = []
    reader = index_memory.reader()  # Remove context manager
    for i in range(num_queries):
        query = queries[i % len(queries)]
        start_time = time.time()
        results = reader.search_text(query, model=model, top_k=5)
        memory_times.append(time.time() - start_time)

    return {
        "disk_avg_ms": np.mean(disk_times) * 1000,
        "memory_avg_ms": np.mean(memory_times) * 1000,
        "speedup": np.mean(disk_times) / np.mean(memory_times),
    }


def main():
    print("SPLADE-Easy Performance Profiling (New API)")
    print("=" * 50)

    # Setup
    model_name = "naver/splade-v3"
    num_docs = 1000
    index_dir = "./profile_index"

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # Generate test data
    print(f"Generating {num_docs} synthetic documents...")
    docs = generate_synthetic_docs(num_docs)

    # Profile indexing
    indexing_results = profile_indexing(docs, index_dir, model)

    print("\nIndexing Results:")
    print(f"  Total time: {indexing_results['total_time']:.2f}s")
    print(f"  Docs/sec: {indexing_results['docs_per_sec']:.1f}")
    print(f"  Index size: {indexing_results['index_size_mb']:.1f} MB")
    print(f"  Num shards: {indexing_results['num_shards']}")

    # Profile search
    search_results = profile_search(index_dir, model, num_queries=100)

    print("\nSearch Results:")
    print(f"  Avg query time: {search_results['avg_query_time'] * 1000:.1f}ms")
    print(
        f"  Min/Max: {search_results['min_query_time'] * 1000:.1f}ms / {search_results['max_query_time'] * 1000:.1f}ms"
    )
    print(f"  Queries/sec: {search_results['queries_per_sec']:.1f}")
    print(f"  Total results: {search_results['total_results']}")

    # Profile memory vs disk
    comparison_results = profile_memory_vs_disk(index_dir, model)

    print("\nMemory vs Disk Comparison:")
    print(f"  Disk mode: {comparison_results['disk_avg_ms']:.1f}ms avg")
    print(f"  Memory mode: {comparison_results['memory_avg_ms']:.1f}ms avg")
    print(f"  Speedup: {comparison_results['speedup']:.1f}x")

    # Cleanup
    # shutil.rmtree(index_dir, ignore_errors=True)
    print("\n✓ Profiling complete!")


if __name__ == "__main__":
    main()
