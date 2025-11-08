# SPLADE-Easy

A lightweight, portable SPLADE index for small-scale document retrieval and RAG applications.

---

## Overview

SPLADE-Easy provides a simple interface for creating and querying sparse lexical indexes produced by [SPLADE](https://huggingface.co/naver/splade-v3) models. It is designed for small to medium datasets (up to about one million documents) where efficient, interpretable retrieval is needed without the overhead of a full vector database.

---

## Installation

```bash
uv sync
```

Requires **Python 3.11+**.

---

## Quick Start

```python
from sentence_transformers import SentenceTransformer
from splade_easy import SpladeIndex

model = SentenceTransformer("naver/splade-v3")

# Create or open an index
index = SpladeIndex("./my_index")

# Add documents
index.add_text(
    doc_id="doc1",
    text="Machine learning is a subset of artificial intelligence.",
    metadata={"source": "wiki"},
    model=model,
)

# Search
retriever = SpladeIndex.retriever("./my_index", mode="memory")
results = retriever.search_text("What is machine learning?", model=model, top_k=3)

for r in results:
    print(r.doc_id, r.score)
```

---

## Core Features

* Simple API for SPLADE-based indexing and retrieval
* Two modes:

  * **disk**: minimal memory footprint
  * **memory**: faster in-RAM search
* Parallel shard search
* Soft deletes with compaction
* Content-addressed shards for deterministic indexing
* Atomic resharding and recovery-safe writes

---

## Command Line Tools

### Ingest a dataset

```bash
uv run ingest-dataset your_config.yaml
```

### Reshard or resize an index

```bash
uv run reshard ./my_index --target-size-mb 64
```

---

## Maintenance

```python
index.delete("doc1")      # mark as deleted
index.compact()           # remove deleted documents
stats = index.stats()     # get index statistics
```

---

## Design Highlights

| Component              | Purpose                               |
| ---------------------- | ------------------------------------- |
| **FlatBuffers**        | Fast zero-copy deserialization        |
| **Append-only shards** | Reliable write pattern for durability |
| **SHA-256 filenames**  | Deterministic and portable storage    |
| **Numba acceleration** | Efficient scoring on CPU              |
| **Atomic writes**      | Prevents partial shard corruption     |

---

## Development

```bash
git clone https://github.com/yourusername/splade-easy
cd splade-easy

# Install dev environment
make install

# Run tests
make test

# Run tests with coverage
make test-cov

# Lint and format
make lint
make format

# Run all pre-commit hooks
make pre-commit

# Clean

MIT License © 2025

```
