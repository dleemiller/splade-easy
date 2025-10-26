"""
Basic RAG example with SPLADE-easy
"""

import cProfile
import pstats
import shutil

from sentence_transformers import SentenceTransformer

from splade_easy import SpladeIndex

profiler = cProfile.Profile()


def main():
    # Load SPLADE model
    print("Loading SPLADE model...")
    model = SentenceTransformer("naver/splade-v3")

    profiler.enable()

    # Create index
    print("\nCreating index...")
    index = SpladeIndex("./demo_index", shard_size_mb=32)

    # Sample documents
    docs_text = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Python is a popular programming language for data science.",
        "Natural language processing helps computers understand text.",
        "SPLADE is a sparse retrieval method for information retrieval.",
    ]

    # Add documents (simple!)
    print("\nIndexing documents...")
    for i, text in enumerate(docs_text):
        index.add_text(
            doc_id=f"doc_{i}", text=text, metadata={"source": "demo", "index": str(i)}, model=model
        )

    print(f"Indexed {len(index)} documents")

    # Create retriever
    print("\nCreating retriever...")
    retriever = SpladeIndex.retriever("../nanomarco_index", mode="memory")

    # Search (simple!)
    query = "how long keep financial records"
    print(f"\nQuery: {query}")

    results = retriever.search_text(query=query, model=model, top_k=3, return_text=True)

    profiler.disable()

    # Display results
    print("\nTop results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result.score:.3f}")
        print(f"   Doc ID: {result.doc_id}")
        print(f"   Text: {result.text}")

    # Cleanup
    shutil.rmtree("./demo_index")
    print("\n✓ Demo complete!")


if __name__ == "__main__":
    main()

    # Print stats
    print("\n" + "=" * 80)
    print("PROFILING RESULTS - TOP 30 BY CUMULATIVE TIME")
    print("=" * 80)
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(30)

    print("\n" + "=" * 80)
    print("PROFILING RESULTS - TOP 30 BY TIME")
    print("=" * 80)
    stats.sort_stats("time")
    stats.print_stats(30)
