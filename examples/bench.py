"""
Clean, focused performance comparison between Old and New SPLADE-Easy APIs
(Updated: fair num_workers comparison, thread-based memory parallelism, and lazy text)
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text
from sentence_transformers import SentenceTransformer

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

from datasets import load_dataset

# Old API
from splade_easy import SpladeIndex
# New API
from splade_easy import Index
from splade_easy.reader import IndexReader

console = Console()


def load_agnews_sample(max_docs: int = 1000) -> List[str]:
    print(f"Loading AG News dataset (max {max_docs} docs)...")
    ds = load_dataset("fancyzhx/ag_news", split="train")
    if max_docs:
        ds = ds.select(range(min(max_docs, len(ds))))
    texts = [row["text"] for row in ds]
    print(f"Loaded {len(texts)} documents")
    print(f"Sample doc: {texts[0][:100]}...")
    return texts


@dataclass
class BenchmarkResult:
    mean_ms: float
    std_ms: float
    min_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    queries_per_sec: float

    @classmethod
    def from_times(cls, times: List[float]) -> "BenchmarkResult":
        times_ms = np.array(times) * 1000
        return cls(
            mean_ms=float(np.mean(times_ms)),
            std_ms=float(np.std(times_ms)),
            min_ms=float(np.min(times_ms)),
            p50_ms=float(np.percentile(times_ms, 50)),
            p95_ms=float(np.percentile(times_ms, 95)),
            p99_ms=float(np.percentile(times_ms, 99)),
            max_ms=float(np.max(times_ms)),
            queries_per_sec=len(times) / sum(times),
        )


@dataclass
class BenchmarkConfig:
    num_docs: int
    num_queries: int
    shard_size_mb: int
    modes: List[str]
    workers: List[int]
    return_text: bool
    label: str


class CleanBenchmark:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.warmup_queries = 20

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task(f"Loading {config.num_docs} documents from AG News...", total=None)
            self.docs = load_agnews_sample(config.num_docs)
            progress.update(task, completed=True)

            task = progress.add_task("Generating queries...", total=None)
            self.queries = self._generate_queries()
            progress.update(task, completed=True)

            task = progress.add_task("Loading SPLADE model...", total=None)
            self.model = SentenceTransformer("naver/splade-v3")
            self.model.show_progress_bar = False
            progress.update(task, completed=True)

            task = progress.add_task(f"Pre-encoding {len(self.queries)} queries...", total=None)
            self._pre_encode_queries()
            progress.update(task, completed=True)

    def _pre_encode_queries(self):
        self.encoded_queries = []
        batch_size = 50

        for i in range(0, len(self.queries), batch_size):
            batch_queries = self.queries[i : i + batch_size]
            encodings = self.model.encode(batch_queries, convert_to_tensor=False, show_progress_bar=False)
            from splade_easy.utils import extract_splade_vectors
            for encoding in encodings:
                query_tokens, query_weights = extract_splade_vectors(encoding)
                self.encoded_queries.append((query_tokens, query_weights))

    def _generate_queries(self) -> List[str]:
        base_queries = [
            "basketball game results", "football championship", "tennis tournament",
            "baseball world series", "soccer match highlights", "olympic games",
            "stock market analysis", "company earnings report", "economic forecast",
            "business merger news", "financial market trends", "corporate strategy",
            "software development", "artificial intelligence", "machine learning",
            "computer programming", "technology innovation", "digital transformation",
            "international politics", "government policy", "diplomatic relations",
            "global economy", "world events", "international trade",
            "breaking news today", "latest updates", "current events",
            "news analysis", "market report", "industry trends",
        ]
        import random
        random.seed(42)
        queries = (base_queries * (self.config.num_queries // len(base_queries) + 1))[: self.config.num_queries]
        random.shuffle(queries)
        return queries

    # ---------- Old API ----------
    def benchmark_old_api(self) -> Dict[str, Any]:
        index_dir = "./benchmark_old"
        shutil.rmtree(index_dir, ignore_errors=True)

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Indexing documents (Old API)...", total=None)
            index_start = time.perf_counter()
            index = SpladeIndex(index_dir, shard_size_mb=self.config.shard_size_mb)
            batch_size = 100
            doc_ids, texts, metadatas = [], [], []
            for i, doc in enumerate(self.docs):
                doc_ids.append(f"doc_{i}")
                texts.append(doc)
                metadatas.append({"idx": str(i)})
                if len(doc_ids) >= batch_size:
                    index.add_texts(doc_ids, texts, metadatas, self.model)
                    doc_ids, texts, metadatas = [], [], []
            if doc_ids:
                index.add_texts(doc_ids, texts, metadatas, self.model)
            index._finalize_current_shard()
            index_time = time.perf_counter() - index_start
            progress.update(task, completed=True)

        results = {"index_time_s": index_time, "docs_per_sec": self.config.num_docs / index_time, "modes": {}}

        for mode in self.config.modes:
            if mode == "disk":
                results["modes"]["disk"] = self._benchmark_old_disk(index_dir)
            elif mode == "memory":
                results["modes"]["memory"] = self._benchmark_old_memory(index_dir)

        shutil.rmtree(index_dir, ignore_errors=True)
        return results

    def _benchmark_old_disk(self, index_dir: str) -> Dict[str, Any]:
        out = {}
        for w in self.config.workers:
            with Progress(SpinnerColumn(), TextColumn(f"Old API disk (workers={w})..."), console=console):
                retriever = SpladeIndex.retriever(index_dir, mode="disk")

                for query_tokens, query_weights in self.encoded_queries[: self.warmup_queries]:
                    _ = retriever.search(query_tokens, query_weights, top_k=10, return_text=self.config.return_text, num_workers=w)

                times = []
                for query_tokens, query_weights in self.encoded_queries:
                    start = time.perf_counter()
                    _ = retriever.search(query_tokens, query_weights, top_k=10, return_text=self.config.return_text, num_workers=w)
                    times.append(time.perf_counter() - start)
            out[w] = {"total": BenchmarkResult.from_times(times)}
        return out

    def _benchmark_old_memory(self, index_dir: str) -> Dict[str, Any]:
        out = {}
        for w in self.config.workers:
            with Progress(SpinnerColumn(), TextColumn(f"Old API memory (workers={w})..."), console=console):
                load_start = time.perf_counter()
                retriever = SpladeIndex.retriever(index_dir, mode="memory")
                load_time = time.perf_counter() - load_start

                for query_tokens, query_weights in self.encoded_queries[: self.warmup_queries]:
                    _ = retriever.search(query_tokens, query_weights, top_k=10, return_text=self.config.return_text, num_workers=w)

                times = []
                for query_tokens, query_weights in self.encoded_queries:
                    start = time.perf_counter()
                    _ = retriever.search(query_tokens, query_weights, top_k=10, return_text=self.config.return_text, num_workers=w)
                    times.append(time.perf_counter() - start)
            out[w] = {"total": BenchmarkResult.from_times(times), "load_time_s": load_time}
        return out

    # ---------- New API ----------
    def benchmark_new_api(self) -> Dict[str, Any]:
        index_dir = "./benchmark_new"
        shutil.rmtree(index_dir, ignore_errors=True)

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Indexing documents (New API)...", total=None)
            index_start = time.perf_counter()
            index = Index(index_dir)
            with index.writer(shard_size_mb=self.config.shard_size_mb) as writer:
                writer.set_model(self.model)
                for i, doc in enumerate(self.docs):
                    writer.insert(f"doc_{i}", doc, {"idx": str(i)})
                writer._finalize_shard()
            index_time = time.perf_counter() - index_start
            progress.update(task, completed=True)

        results = {"index_time_s": index_time, "docs_per_sec": self.config.num_docs / index_time, "modes": {}}

        for mode in self.config.modes:
            if mode == "disk":
                results["modes"]["disk"] = self._benchmark_new_disk(index_dir)
            elif mode == "memory":
                results["modes"]["memory"] = self._benchmark_new_memory(index_dir)

        shutil.rmtree(index_dir, ignore_errors=True)
        return results

    def _benchmark_new_disk(self, index_dir: str) -> Dict[str, Any]:
        out = {}
        for w in self.config.workers:
            with Progress(SpinnerColumn(), TextColumn(f"New API disk (workers={w})..."), console=console):
                index_disk = Index(index_dir, memory=False)
                reader_disk = IndexReader(index_disk, num_workers=w)

                for query_tokens, query_weights in self.encoded_queries[: self.warmup_queries]:
                    _ = reader_disk.search_vectors(query_tokens, query_weights, top_k=10, return_text=self.config.return_text)

                times = []
                for query_tokens, query_weights in self.encoded_queries:
                    start = time.perf_counter()
                    _ = reader_disk.search_vectors(query_tokens, query_weights, top_k=10, return_text=self.config.return_text)
                    times.append(time.perf_counter() - start)
            out[w] = {"total": BenchmarkResult.from_times(times)}
        return out

    def _benchmark_new_memory(self, index_dir: str) -> Dict[str, Any]:
        out = {}
        for w in self.config.workers:
            with Progress(SpinnerColumn(), TextColumn(f"New API memory (workers={w})..."), console=console):
                load_start = time.perf_counter()
                index_mem = Index(index_dir, memory=True)
                reader_mem = IndexReader(index_mem, num_workers=w)
                reader_mem.load()
                load_time = time.perf_counter() - load_start

                for query_tokens, query_weights in self.encoded_queries[: self.warmup_queries]:
                    _ = reader_mem.search_vectors(query_tokens, query_weights, top_k=10, return_text=self.config.return_text)

                times = []
                for query_tokens, query_weights in self.encoded_queries:
                    start = time.perf_counter()
                    _ = reader_mem.search_vectors(query_tokens, query_weights, top_k=10, return_text=self.config.return_text)
                    times.append(time.perf_counter() - start)
            out[w] = {"total": BenchmarkResult.from_times(times), "load_time_s": load_time}
        return out

    # ---------- Reporting ----------
    def print_results(self, old_results: Dict, new_results: Dict):
        print("\n" + "=" * 80)
        print(f"BENCHMARK RESULTS: {self.config.num_docs} docs, {self.config.num_queries} queries")
        print("=" * 80)
        print("\n📊 INDEXING PERFORMANCE")
        print("-" * 40)
        print(f"{'Metric':<20} {'Old API':>15} {'New API':>15} {'Difference':>15}")
        print(
            f"{'Time (s)':<20} {old_results['index_time_s']:>15.2f} {new_results['index_time_s']:>15.2f} "
            f"{self._format_diff(old_results['index_time_s'], new_results['index_time_s'])}"
        )
        print(
            f"{'Docs/sec':<20} {old_results['docs_per_sec']:>15.1f} {new_results['docs_per_sec']:>15.1f} "
            f"{self._format_diff(new_results['docs_per_sec'], old_results['docs_per_sec'], reverse=True)}"
        )

        for mode in self.config.modes:
            print(f"\n📊 {mode.upper()} MODE SEARCH (mean ms by workers)")
            print("-" * 40)
            print(f"{'workers':<10} {'Old':>12} {'New':>12} {'Δ New vs Old':>16}")
            for w in self.config.workers:
                o = old_results["modes"][mode][w]["total"].mean_ms
                n = new_results["modes"][mode][w]["total"].mean_ms
                diff = self._format_diff(o, n)
                print(f"{w:<10} {o:>12.1f} {n:>12.1f} {diff:>16}")

    def _format_diff(self, old_val: float, new_val: float, reverse: bool = False) -> str:
        if reverse:
            if new_val > old_val:
                pct = (new_val - old_val) / old_val * 100
                return f"+{pct:.1f}% ✅"
            else:
                pct = (old_val - new_val) / old_val * 100
                return f"-{pct:.1f}% ❌"
        else:
            if new_val < old_val:
                pct = (old_val - new_val) / old_val * 100
                return f"-{pct:.1f}% ✅"
            else:
                pct = (new_val - old_val) / old_val * 100
                return f"+{pct:.1f}% ❌"


def create_results_table(results_summary: List[Dict]) -> Table:
    table = Table(title="Benchmark Results Summary", box=box.ROUNDED)
    table.add_column("Config", style="bold cyan")
    table.add_column("Shard (MB)", justify="right")
    table.add_column("Mode", style="bold")
    table.add_column("Workers", justify="right")
    table.add_column("Old API (ms)", justify="right", style="yellow")
    table.add_column("New API (ms)", justify="right", style="green")
    table.add_column("Winner", justify="center")

    for result in results_summary:
        cfg = f"{result['docs']} docs\n{result['queries']} queries"
        for mode in result["modes"]:
            for row in mode["rows"]:
                winner = "🟢 New" if row["new_ms"] < row["old_ms"] else "🔴 Old"
                table.add_row(
                    cfg,
                    str(result["shard_mb"]),
                    mode["mode"].title(),
                    str(row["workers"]),
                    f"{row['old_ms']:.1f}",
                    f"{row['new_ms']:.1f}",
                    winner,
                )
            cfg = ""
    return table


def main():
    parser = argparse.ArgumentParser(description="SPLADE-Easy API Performance Comparison")
    parser.add_argument("--docs", type=int, nargs="+", default=[1000], help="Number of documents to test")
    parser.add_argument("--queries", type=int, default=500, help="Number of queries per test")
    parser.add_argument("--shard-sizes", type=int, nargs="+", default=[32], help="Shard sizes in MB")
    parser.add_argument("--modes", choices=["disk", "memory"], nargs="+", default=["memory"], help="Modes to benchmark")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4], help="Worker counts to test")
    parser.add_argument("--return-text", action="store_true", help="Return text during timing (slower)")
    parser.add_argument("--output-dir", default=".", help="Directory for result files")
    args = parser.parse_args()

    console.print(Panel.fit("🚀 SPLADE-Easy API Performance Comparison", style="bold blue"))

    results_summary = []

    for num_docs in args.docs:
        for shard_size in args.shard_sizes:
            config = BenchmarkConfig(
                num_docs=num_docs,
                num_queries=args.queries,
                shard_size_mb=shard_size,
                modes=args.modes,
                workers=args.workers,
                return_text=args.return_text,
                label=f"{num_docs}docs_{shard_size}mb",
            )

            console.print(f"\n[bold]Testing: {num_docs} docs, {args.queries} queries, {shard_size}MB shards[/bold]")
            benchmark = CleanBenchmark(config)

            old_results = benchmark.benchmark_old_api()
            new_results = benchmark.benchmark_new_api()

            # Build compact table data
            mode_rows = []
            for mode in args.modes:
                rows = []
                for w in args.workers:
                    old_ms = old_results["modes"][mode][w]["total"].mean_ms
                    new_ms = new_results["modes"][mode][w]["total"].mean_ms
                    rows.append({"workers": w, "old_ms": old_ms, "new_ms": new_ms})
                mode_rows.append({"mode": mode, "rows": rows})

            results_summary.append(
                {"docs": num_docs, "queries": args.queries, "shard_mb": shard_size, "modes": mode_rows}
            )

            # Save detailed JSON
            results_file = Path(args.output_dir) / f"benchmark_{config.label}.json"
            with open(results_file, "w") as f:
                json.dump({"config": vars(config), "old_api": old_results, "new_api": new_results}, f, indent=2, default=str)

    console.print("\n")
    console.print(create_results_table(results_summary))
    console.print(f"\n✅ [green]Benchmark complete![/green] Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
