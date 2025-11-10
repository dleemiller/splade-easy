"""
Profiling example for SPLADE-easy performance testing
"""

import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

import splade_easy as se


def generate_synthetic_docs(num_docs: int = 1000) -> list[str]:
    """Generate synthetic documents for testing"""
    topics = [
        "machine learning artificial intelligence neural networks deep learning",
        "python programming data science software development coding",
        "natural language processing text analysis computational linguistics",
        "computer vision image recognition object detection classification",
        "database systems sql nosql data storage retrieval indexing",
        "web development javascript html css frontend backend",
        "cybersecurity encryption authentication network security protocols",
        "cloud computing aws azure distributed systems microservices",
        "mobile development ios android react native flutter",
        "blockchain cryptocurrency bitcoin ethereum smart contracts",
    ]

    docs = []
    for i in range(num_docs):
        # Mix topics and add some variety
        topic = topics[i % len(topics)]
        variation = (
            f"Document {i}: {topic}. This covers various aspects and applications in the field."
        )
        docs.append(variation)

    return docs


def profile_indexing(docs: list[str], index_dir: str, model) -> dict:
    """Profile document indexing performance"""
    print(f"Profiling indexing of {len(docs)} documents...")

    index = se.Index(index_dir)

    start_time = time.time()

    with index.writer() as writer:
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

    total_time = time.time() - start_time
    docs_per_sec = len(docs) / total_time

    stats = index.stats

    return {
        "total_time": total_time,
        "docs_per_sec": docs_per_sec,
        "num_docs": len(docs),
        "index_stats": stats,
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

    index = se.Index(index_dir)

    search_times = []
    total_results = 0

    with index.reader() as reader:
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
    index_disk = se.Index(index_dir, memory=False)

    disk_times = []
    with index_disk.reader() as reader:
        for i in range(num_queries):
            query = queries[i % len(queries)]
            start_time = time.time()
            results = reader.search_text(query, model=model, top_k=5)
            disk_times.append(time.time() - start_time)

    # Test memory mode
    print("  Testing memory mode...")
    index_memory = se.Index(index_dir, memory=True)

    memory_times = []
    with index_memory.reader() as reader:
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
    print("SPLADE-Easy Performance Profiling")
    print("=" * 40)

    # Setup
    model_name = "naver/splade-v3"
    num_docs = 1000

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = Path(tmpdir) / "profile_index"

        # Generate test data
        print(f"Generating {num_docs} synthetic documents...")
        docs = generate_synthetic_docs(num_docs)

        # Profile indexing
        indexing_stats = profile_indexing(docs, str(index_dir), model)

        print("\nIndexing Results:")
        print(f"  Total time: {indexing_stats['total_time']:.2f}s")
        print(f"  Docs/sec: {indexing_stats['docs_per_sec']:.1f}")
        print(f"  Index size: {indexing_stats['index_stats']['total_size_mb']:.1f} MB")
        print(f"  Num shards: {indexing_stats['index_stats']['num_shards']}")

        # Profile search
        search_stats = profile_search(str(index_dir), model, num_queries=100)

        print("\nSearch Results:")
        print(f"  Avg query time: {search_stats['avg_query_time'] * 1000:.1f}ms")
        print(
            f"  Min/Max: {search_stats['min_query_time'] * 1000:.1f}ms / {search_stats['max_query_time'] * 1000:.1f}ms"
        )
        print(f"  Queries/sec: {search_stats['queries_per_sec']:.1f}")
        print(f"  Total results: {search_stats['total_results']}")

if __name__ == "__main__":
    main()