# splade-easy

Fast, simple sparse retrieval for small corpora using **inference-free SPLADE** models. Inspired by [bm25s](https://github.com/xhluca/bm25s).

The query path is transformer-free: just a fast tokenizer + a dense IDF lookup + an inverted-index scoring loop. Document encoding runs offline once, on whatever hardware is convenient.

Target: indexes up to ~100k documents.

## Install

```bash
uv add splade-easy                  # query-only (numpy + tokenizers)
uv add 'splade-easy[encode]'        # also pulls torch + sentence-transformers for doc encoding
uv add 'splade-easy[encode,eval]'   # adds NanoBEIR datasets loader
```

Requires Python 3.11+.

## Quickstart

```python
import splade_easy as se

corpus = [
    "A cat is a small carnivorous mammal.",
    "Dogs are loyal companions and pack animals.",
    "Pythons are non-venomous constrictor snakes.",
]

# Offline — encode docs (slow, runs the SPLADE doc model)
sparse_docs = se.encode_corpus(corpus)

# Build index
retriever = se.SpladeRetriever()
retriever.index(sparse_docs)
retriever.save("./idx", corpus=corpus)

# Online — no transformer is loaded
retriever = se.SpladeRetriever.load("./idx", load_corpus=True)
results, scores = retriever.retrieve("what is a feline?", k=2, return_docs=True)
```

## Default model

`opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill` is auto-paired with its bundled `idf.json` for query weighting. Override either side via the `model=` and `query_weights=` arguments.

## On-disk layout

```
idx/
  params.json           # model id, sizes, version, dtypes
  indptr.npy            # CSC inverted index
  indices.npy
  data.npy
  query_weights.npy     # dense (vocab_size,) IDF lookup
  tokenizer/            # HF fast tokenizer files
  corpus.jsonl          # optional
```

## Performance

Query latency, single thread, k=10, synthetic SPLADE-like corpus (vocab 30522, 100-250 nnz/doc):

| corpus  | backend | mean ms/q | p95 ms/q | speedup |
|---------|---------|-----------|----------|---------|
| 10k     | numpy   | 0.171     | 0.262    | 1.0x    |
| 10k     | cython  | **0.027** | 0.037    | 6.9x    |
| 50k     | numpy   | 0.709     | 1.138    | 1.0x    |
| 50k     | cython  | **0.103** | 0.150    | 7.0x    |

Reproduce: `uv run python benchmarks/bench_query.py --docs 50000 --queries 1000`.

## Quality (NanoBEIR)

Auto-eval with the default model on `zeta-alpha-ai/Nano*` subsets:

| dataset  | NDCG@10 | Recall@10 | MRR@10 |
|----------|---------|-----------|--------|
| scifact  | 0.743   | 0.930     | 0.687  |
| nq       | 0.707   | 0.850     | 0.669  |
| fiqa     | 0.482   | 0.553     | 0.545  |
| nfcorpus | 0.347   | 0.143     | 0.499  |

Reproduce: `uv run splade-eval-nanobeir --datasets scifact,nq,fiqa,nfcorpus`.

## Development

```bash
uv sync --extra encode --extra eval --group dev
uv run pytest                                  # 51 tests, ~0.2s
uv run cython-lint src/splade_easy/_scoring.pyx
uv run python benchmarks/bench_query.py
```

Cython hot path is in `src/splade_easy/_scoring.pyx`. A pure-numpy reference at `src/splade_easy/_scoring_py.py` is used in tests for parity checks and as a fallback when the C extension isn't built. The build is driven by `setup.py` + `setuptools.build_meta`.

## Status

Alpha.

## License

MIT.
