"""
Enhanced Performance profiling for SPLADE-Easy with cProfile support
"""

import cProfile
import io
import pstats
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from splade_easy import Index
from splade_easy.reader import IndexReader


@dataclass
class TimingResult:
    """Store timing results with statistics"""

    times: List[float]
    total: float
    mean: float
    std: float
    min: float
    max: float
    p50: float
    p95: float
    p99: float

    @classmethod
    def from_times(cls, times: List[float]) -> "TimingResult":
        """Create TimingResult from list of times"""
        if not times:
            return cls([], 0, 0, 0, 0, 0, 0, 0, 0)

        times_array = np.array(times)
        return cls(
            times=times,
            total=float(np.sum(times_array)),
            mean=float(np.mean(times_array)),
            std=float(np.std(times_array)),
            min=float(np.min(times_array)),
            max=float(np.max(times_array)),
            p50=float(np.percentile(times_array, 50)),
            p95=float(np.percentile(times_array, 95)),
            p99=float(np.percentile(times_array, 99)),
        )


@contextmanager
def timer(name: str = "Operation"):
    """Context manager for timing operations"""
    start = time.perf_counter()  # Use perf_counter for better precision
    yield
    elapsed = time.perf_counter() - start
    print(f"  {name}: {elapsed:.3f}s")


class PerformanceProfiler:
    """Enhanced performance profiler for SPLADE-Easy"""

    def __init__(self, model_name: str = "naver/splade-v3"):
        self.model_name = model_name
        self.model = None
        self.profiler = None

    def load_model(self):
        """Load the model with timing"""
        with timer("Model loading"):
            self.model = SentenceTransformer(self.model_name)
            self.model.show_progress_bar = False
        return self.model

    def generate_synthetic_docs(self, num_docs: int) -> List[str]:
        """Generate synthetic documents for testing"""
        topics = [
            "machine learning algorithms and neural networks for deep learning",
            "python programming and software development best practices",
            "natural language processing and text analysis techniques",
            "computer vision and image recognition with convolutional networks",
            "database design and query optimization for large scale systems",
            "web development frameworks and RESTful API design patterns",
            "cybersecurity threats and advanced protection methods",
            "cloud computing and distributed systems architecture",
            "mobile application development for iOS and Android",
            "blockchain technology and cryptocurrency implementations",
            "data science and statistical analysis methodologies",
            "DevOps practices and continuous integration pipelines",
        ]

        docs = []
        for i in range(num_docs):
            topic = topics[i % len(topics)]
            # Create more varied documents
            variation = i % 5
            if variation == 0:
                doc = f"Document {i}: This comprehensive guide covers {topic}. "
                doc += f"We explore the fundamentals of {topic} with practical examples. "
                doc += f"Advanced techniques in {topic} are discussed in detail."
            elif variation == 1:
                doc = f"Article {i}: Introduction to {topic}. "
                doc += f"Learn the basics and advanced concepts of {topic}. "
                doc += f"Real-world applications of {topic} are examined."
            elif variation == 2:
                doc = f"Tutorial {i}: Mastering {topic} step by step. "
                doc += f"From beginner to expert in {topic}. "
                doc += f"Best practices and common pitfalls in {topic}."
            elif variation == 3:
                doc = f"Research paper {i}: Novel approaches to {topic}. "
                doc += f"We present new methods for {topic} with empirical results. "
                doc += f"Comparative analysis of {topic} techniques."
            else:
                doc = f"Reference {i}: Complete guide to {topic}. "
                doc += f"Everything you need to know about {topic}. "
                doc += f"Expert insights and future directions in {topic}."

            docs.append(doc)

        return docs

    def profile_indexing(
        self, docs: List[str], index_dir: str, batch_size: int = 100
    ) -> Dict[str, Any]:
        """Profile document indexing performance with detailed timing"""
        print(f"\nProfiling indexing of {len(docs)} documents...")

        # Clean up any existing index
        shutil.rmtree(index_dir, ignore_errors=True)

        # Warm up the model (exclude from timing)
        print("  Warming up model...")
        _ = self.model.encode(docs[:2], convert_to_tensor=False, show_progress_bar=False)

        index = Index(index_dir)

        # Track different timing components
        encoding_times = []
        insertion_times = []
        batch_times = []

        total_start = time.perf_counter()

        with index.writer(shard_size_mb=32) as writer:
            writer.set_model(self.model)

            for batch_idx in range(0, len(docs), batch_size):
                batch_start = time.perf_counter()
                batch_docs = docs[batch_idx : batch_idx + batch_size]

                # Time encoding separately
                encode_start = time.perf_counter()
                # Pre-encode the batch (this is what writer.insert does internally)
                # This helps us measure encoding vs insertion time
                encoded_batch = self.model.encode(batch_docs, convert_to_tensor=False, show_progress_bar=False)
                encoding_time = time.perf_counter() - encode_start
                encoding_times.append(encoding_time)

                # Time insertion
                insert_start = time.perf_counter()
                for i, (text, encoding) in enumerate(zip(batch_docs, encoded_batch)):
                    doc_idx = batch_idx + i
                    writer.insert(
                        doc_id=f"doc_{doc_idx}",
                        text=text,
                        metadata={"batch": str(doc_idx // 100), "index": str(doc_idx)},
                    )
                insertion_time = time.perf_counter() - insert_start
                insertion_times.append(insertion_time)

                batch_time = time.perf_counter() - batch_start
                batch_times.append(batch_time)

                if (batch_idx + batch_size) % 500 == 0:
                    print(
                        f"    Indexed {batch_idx + batch_size} docs, "
                        f"last batch: {batch_time:.2f}s "
                        f"(encode: {encoding_time:.2f}s, insert: {insertion_time:.2f}s)"
                    )

            # Force finalize any pending shard
            finalize_start = time.perf_counter()
            writer._finalize_shard()
            finalize_time = time.perf_counter() - finalize_start

        total_time = time.perf_counter() - total_start

        # Refresh index to see finalized shards
        index.refresh()
        stats = index.stats

        # Calculate statistics
        encoding_stats = TimingResult.from_times(encoding_times)
        insertion_stats = TimingResult.from_times(insertion_times)
        batch_stats = TimingResult.from_times(batch_times)

        return {
            "total_time": total_time,
            "docs_per_sec": len(docs) / total_time,
            "num_docs": len(index),
            "index_size_mb": stats.get("total_size_mb", 0),
            "num_shards": stats["num_shards"],
            "encoding_stats": encoding_stats,
            "insertion_stats": insertion_stats,
            "batch_stats": batch_stats,
            "finalize_time": finalize_time,
            "encoding_pct": (encoding_stats.total / total_time) * 100,
            "insertion_pct": (insertion_stats.total / total_time) * 100,
        }

    def profile_search(
        self, index_dir: str, num_queries: int = 100, warmup_queries: int = 10, memory: bool = False
    ) -> Dict[str, Any]:
        """Profile search performance with warmup and detailed timing"""
        mode_str = "memory" if memory else "disk"
        print(f"\nProfiling {num_queries} search queries in {mode_str} mode (with {warmup_queries} warmup)...")

        queries = [
            "machine learning algorithms",
            "python programming tutorial",
            "natural language processing techniques",
            "computer vision applications",
            "database design patterns",
            "web development frameworks",
            "cybersecurity best practices",
            "cloud computing architecture",
            "mobile app development",
            "blockchain technology",
            "data science methodology",
            "DevOps automation",
        ]

        index = Index(index_dir, memory=memory)
        if memory:
            index.load()  # Load into memory
        reader = IndexReader(index)

        # Warmup phase
        print("  Running warmup queries...")
        for i in range(warmup_queries):
            query = queries[i % len(queries)]
            _ = reader.search_text(query, model=self.model, top_k=10)

        # Actual profiling
        search_times = []
        encoding_times = []
        retrieval_times = []
        total_results = 0

        for i in range(num_queries):
            query = queries[i % len(queries)]

            # Total search time
            search_start = time.perf_counter()

            # Measure encoding separately
            encode_start = time.perf_counter()
            query_encoding = self.model.encode(query, convert_to_tensor=False, show_progress_bar=False)
            encode_time = time.perf_counter() - encode_start
            encoding_times.append(encode_time)

            # Measure retrieval time
            retrieval_start = time.perf_counter()
            results = reader.search_text(query, model=self.model, top_k=10, return_text=True)
            retrieval_time = time.perf_counter() - retrieval_start
            retrieval_times.append(retrieval_time)

            search_time = time.perf_counter() - search_start
            search_times.append(search_time)
            total_results += len(results)

            if (i + 1) % 20 == 0:
                recent_avg = np.mean(search_times[-20:]) * 1000
                print(f"    Completed {i + 1} queries, recent avg: {recent_avg:.1f}ms")

        search_stats = TimingResult.from_times(search_times)
        encoding_stats = TimingResult.from_times(encoding_times)
        retrieval_stats = TimingResult.from_times(retrieval_times)

        return {
            "num_queries": num_queries,
            "search_stats": search_stats,
            "encoding_stats": encoding_stats,
            "retrieval_stats": retrieval_stats,
            "queries_per_sec": num_queries / search_stats.total,
            "total_results": total_results,
            "encoding_pct": (encoding_stats.total / search_stats.total) * 100,
            "retrieval_pct": (retrieval_stats.total / search_stats.total) * 100,
        }

    def profile_memory_vs_disk(
        self, index_dir: str, num_queries: int = 50, warmup_queries: int = 5
    ) -> Dict[str, Any]:
        """Compare memory vs disk mode performance"""
        print("\nComparing memory vs disk mode...")

        queries = [
            "machine learning",
            "python programming",
            "data science",
            "neural networks",
            "software engineering",
            "cloud computing",
        ]

        results = {}

        # Test disk mode
        print("  Testing disk mode...")
        index_disk = Index(index_dir, memory=False)
        reader_disk = IndexReader(index_disk)

        # Warmup
        for i in range(warmup_queries):
            query = queries[i % len(queries)]
            _ = reader_disk.search_text(query, model=self.model, top_k=5)

        # Actual timing
        disk_times = []
        for i in range(num_queries):
            query = queries[i % len(queries)]
            start = time.perf_counter()
            _ = reader_disk.search_text(query, model=self.model, top_k=5)
            disk_times.append(time.perf_counter() - start)

        disk_stats = TimingResult.from_times(disk_times)

        # Test memory mode
        print("  Testing memory mode...")
        load_start = time.perf_counter()
        index_memory = Index(index_dir, memory=True).load()
        load_time = time.perf_counter() - load_start
        print(f"    Index load time: {load_time:.2f}s")

        reader_memory = IndexReader(index_memory)

        # Warmup
        for i in range(warmup_queries):
            query = queries[i % len(queries)]
            _ = reader_memory.search_text(query, model=self.model, top_k=5)

        # Actual timing
        memory_times = []
        for i in range(num_queries):
            query = queries[i % len(queries)]
            start = time.perf_counter()
            _ = reader_memory.search_text(query, model=self.model, top_k=5)
            memory_times.append(time.perf_counter() - start)

        memory_stats = TimingResult.from_times(memory_times)

        return {
            "disk_stats": disk_stats,
            "memory_stats": memory_stats,
            "load_time": load_time,
            "speedup_mean": disk_stats.mean / memory_stats.mean,
            "speedup_p50": disk_stats.p50 / memory_stats.p50,
            "speedup_p95": disk_stats.p95 / memory_stats.p95,
        }

    def run_with_cprofile(self, func, *args, **kwargs):
        """Run a function with cProfile enabled"""
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        # Print stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(20)  # Top 20 functions
        print("\n" + "=" * 50)
        print(f"cProfile results for {func.__name__}:")
        print("=" * 50)
        print(s.getvalue())

        return result


def format_timing_stats(stats: TimingResult, name: str, unit: str = "ms") -> str:
    """Format timing statistics for display"""
    multiplier = 1000 if unit == "ms" else 1
    return (
        f"{name}:\n"
        f"  Mean: {stats.mean * multiplier:.2f}{unit}\n"
        f"  Std:  {stats.std * multiplier:.2f}{unit}\n"
        f"  Min:  {stats.min * multiplier:.2f}{unit}\n"
        f"  P50:  {stats.p50 * multiplier:.2f}{unit}\n"
        f"  P95:  {stats.p95 * multiplier:.2f}{unit}\n"
        f"  P99:  {stats.p99 * multiplier:.2f}{unit}\n"
        f"  Max:  {stats.max * multiplier:.2f}{unit}"
    )


def main():
    print("=" * 60)
    print("SPLADE-Easy Enhanced Performance Profiling")
    print("=" * 60)

    # Configuration
    model_name = "naver/splade-v3"
    num_docs = 1000
    index_dir = "./profile_index"
    use_cprofile = True  # Toggle cProfile

    profiler = PerformanceProfiler(model_name)

    # Load model
    print(f"\nLoading model: {model_name}")
    profiler.load_model()

    # Generate test data
    print(f"\nGenerating {num_docs} synthetic documents...")
    docs = profiler.generate_synthetic_docs(num_docs)
    print(f"  Sample doc length: {len(docs[0])} chars")

    # Profile indexing
    if use_cprofile:
        indexing_results = profiler.run_with_cprofile(profiler.profile_indexing, docs, index_dir)
    else:
        indexing_results = profiler.profile_indexing(docs, index_dir)

    print("\n" + "=" * 60)
    print("INDEXING RESULTS")
    print("=" * 60)
    print(f"Total time: {indexing_results['total_time']:.2f}s")
    print(f"Docs/sec: {indexing_results['docs_per_sec']:.1f}")
    print(f"Index size: {indexing_results['index_size_mb']:.1f} MB")
    print(f"Num shards: {indexing_results['num_shards']}")
    print(f"Finalize time: {indexing_results['finalize_time']:.3f}s")
    print(f"\nTime breakdown:")
    print(f"  Encoding: {indexing_results['encoding_pct']:.1f}%")
    print(f"  Insertion: {indexing_results['insertion_pct']:.1f}%")
    print(f"\n{format_timing_stats(indexing_results['batch_stats'], 'Batch times', 's')}")

    # Profile search - disk mode
    if use_cprofile:
        search_results = profiler.run_with_cprofile(
            profiler.profile_search, index_dir, num_queries=100, memory=False
        )
    else:
        search_results = profiler.profile_search(index_dir, num_queries=100, memory=False)

    # Profile search - memory mode  
    if use_cprofile:
        search_memory_results = profiler.run_with_cprofile(
            profiler.profile_search, index_dir, num_queries=100, memory=True
        )
    else:
        search_memory_results = profiler.profile_search(index_dir, num_queries=100, memory=True)

    print("\n" + "=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)
    print(f"Queries/sec: {search_results['queries_per_sec']:.1f}")
    print(f"Total results: {search_results['total_results']}")
    print(f"\nTime breakdown:")
    print(f"  Encoding: {search_results['encoding_pct']:.1f}%")
    print(f"  Retrieval: {search_results['retrieval_pct']:.1f}%")
    print(f"\n{format_timing_stats(search_results['search_stats'], 'Total search times')}")
    print(f"\n{format_timing_stats(search_results['encoding_stats'], 'Encoding times')}")
    print(f"\n{format_timing_stats(search_results['retrieval_stats'], 'Retrieval times')}")

    # Profile memory vs disk
    comparison_results = profiler.profile_memory_vs_disk(index_dir)

    print("\n" + "=" * 60)
    print("MEMORY VS DISK COMPARISON")
    print("=" * 60)
    print(f"Index load time: {comparison_results['load_time']:.2f}s")
    print(f"\n{format_timing_stats(comparison_results['disk_stats'], 'Disk mode')}")
    print(f"\n{format_timing_stats(comparison_results['memory_stats'], 'Memory mode')}")
    print(f"\nSpeedup factors:")
    print(f"  Mean: {comparison_results['speedup_mean']:.1f}x")
    print(f"  P50:  {comparison_results['speedup_p50']:.1f}x")
    print(f"  P95:  {comparison_results['speedup_p95']:.1f}x")

    # Cleanup
    print("\nCleaning up...")
    shutil.rmtree(index_dir, ignore_errors=True)

    print("\n✓ Profiling complete!")


if __name__ == "__main__":
    main()
