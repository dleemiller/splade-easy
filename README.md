# splade-easy

A lightweight SPLADE index for small-scale document retrieval and RAG applications.

## Installation
```bash
pip install splade-easy
```

## Quick Start
```python
from sentence_transformers import SentenceTransformer
from splade_easy import SpladeIndex

# Load SPLADE model
model = SentenceTransformer('naver/splade-cocondenser-ensembledistil')

# Create index and add documents
index = SpladeIndex('./my_index')
index.add_text(
    doc_id='doc_1',
    text='Machine learning is a subset of artificial intelligence',
    metadata={'source': 'wiki'},
    model=model
)

# Search
retriever = SpladeIndex.retriever('./my_index', mode='memory')
results = retriever.search_text(
    query="What is machine learning?",
    model=model,
    top_k=5
)

for result in results:
    print(f"{result.doc_id}: {result.score}")
```

## Features

- Simple API for SPLADE-based retrieval
- Efficient FlatBuffers serialization
- Two modes: disk (low memory) or memory (fast)
- Parallel search across shards
- Soft deletes with compaction
- Built for small to medium datasets (up to 1M documents)

## Core API

### Writing Documents
```python
from splade_easy import SpladeIndex, Document

index = SpladeIndex('./my_index', shard_size_mb=32)

# Simple: encode and add in one step
index.add_text(doc_id, text, metadata, model)

# Batch operations
index.add_texts(doc_ids, texts, metadatas, model)

# Advanced: manual control
import numpy as np
doc = Document(
    doc_id='doc_1',
    text='...',
    metadata={},
    token_ids=np.array([...], dtype=np.uint32),
    weights=np.array([...], dtype=np.float32)
)
index.add(doc)
```

### Searching
```python
# Create retriever
retriever = SpladeIndex.retriever('./my_index', mode='disk')  # or 'memory'

# Simple search
results = retriever.search_text(
    query="your query",
    model=model,
    top_k=10,
    return_text=True,
    num_workers=4  # parallel search
)

# Advanced: use pre-encoded vectors
results = retriever.search(
    query_tokens=tokens,
    query_weights=weights,
    top_k=10
)
```

### Maintenance
```python
# Delete documents (soft delete)
index.delete('doc_id')

# Compact (remove deleted documents)
index.compact()

# Statistics
stats = index.stats()
# Returns: {'num_docs': ..., 'num_shards': ..., 'deleted_docs': ..., 'total_size_mb': ...}
```

## Design

### Architecture

- **Append-only shards**: Documents written to 32MB shards (configurable)
- **Soft deletes**: Deleted documents marked, removed during compaction
- **Sequential scan**: Simple, effective for datasets under 1M documents
- **FlatBuffers**: Fast zero-copy deserialization

### Modes

**Disk mode** (default):
- Scans from disk each query
- Low memory usage
- 200-500ms per query
- Use for: Infrequent searches, large indexes

**Memory mode**:
- Preloads all shards into RAM
- 50-100ms per query
- ~50MB RAM per 10K documents
- Use for: Frequent searches, real-time applications

### Parallel Search

Search automatically parallelizes across shards:
```python
results = retriever.search_text(
    query="...",
    model=model,
    num_workers=4  # search 4 shards in parallel
)
```

## SPLADE Models

Compatible with sentence-transformers SPLADE models:

- `naver/splade-cocondenser-ensembledistil` (recommended)
- `prithivida/Splade_PP_en_v1`

## Examples

See `examples/basic_rag.py` for a complete example.

## Development
```bash
# Setup
git clone https://github.com/yourusername/splade-easy
cd splade-easy
uv sync

# Run tests
make test

# Run with coverage
make test-cov

# Lint and format
make lint
make format
```

## Performance

Typical performance on modern hardware:

- **Indexing**: ~1000 documents/second
- **Disk search**: 200-500ms per query
- **Memory search**: 50-100ms per query
- **Storage**: ~500KB per 1000 documents (with typical SPLADE sparsity)

Performance scales linearly with document count for datasets under 1M documents.

## License

MIT
