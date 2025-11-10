"""
Basic RAG example with SPLADE-easy
"""

import shutil

from sentence_transformers import SentenceTransformer

import splade_easy as se


def main():
    # Load SPLADE model
    print("Loading SPLADE model...")
    model = SentenceTransformer("naver/splade-v3")

    # Sample documents
    docs_text = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Python is a popular programming language for data science.",
        "Natural language processing helps computers understand text.",
        "SPLADE is a sparse retrieval method for information retrieval.",
    ]

    # Create index and writer
    index = se.Index("./demo_index")

    with index.writer() as writer:
        writer.set_model(model)

        print("\nIndexing documents...")
        for i, text in enumerate(docs_text):
            writer.insert(
                doc_id=f"doc_{i}", text=text, metadata={"source": "demo", "index": str(i)}
            )

        print(f"Indexed {len(index)} documents")

    # Search using reader
    with index.reader() as reader:
        query = "what is machine learning?"
        print(f"\nQuery: {query}")

        results = reader.search_text(query, model=model, top_k=3, return_text=True)

        # Display results
        print("\nTop results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result.score:.3f}")
            print(f"   Doc ID: {result.doc_id}")
            print(f"   Text: {result.text}")

        # Show stats
        print(f"\nIndex stats: {index.stats}")

    # Cleanup
    shutil.rmtree("./demo_index")
    print("\n✓ Demo complete!")


if __name__ == "__main__":
    main()
