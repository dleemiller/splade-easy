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

## Using a different model

The default doc encoder is `opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill`. Its bundled `idf.json` is fetched automatically and used for query-side weighting. To use a different model, pass `model="..."` to both `encode_corpus()` and `SpladeRetriever`:

```python
import splade_easy as se

MODEL = "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"

sparse_docs = se.encode_corpus(corpus, model=MODEL)
retriever = se.SpladeRetriever(model=MODEL)
retriever.index(sparse_docs)
retriever.save("./idx", corpus=corpus)
```

Known inference-free SPLADE models. The registry in `splade_easy/models.py` is additive — any HF model id can be passed; the listing here documents which we've verified.

| model id | notes |
|---|---|
| `opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill` | Default. English, BERT-base, ~67M params. |
| `opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte` | English, GTE-base custom backbone, ~137M. Higher NDCG than distill; same query latency. |
| `opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill` | English, predecessor of v3-distill. |
| `opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1` | Multilingual, XLM-R backbone. |

The `[encode]` extra pins `transformers<5` because transformers 5.x has interop bugs with several of these models' custom modeling code (gte produces all-NaN encodings on multi-doc batches, position_ids buffers come back uninitialized, MLM-head weight tying gets skipped). sentence-transformers 5.5 supports the 4.x line, so this is a clean pin within its compatibility matrix.

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

splade-easy with two doc encoders vs [bm25s](https://github.com/xhluca/bm25s) on NanoBEIR (English subsets). Quality is full end-to-end retrieval; latency is `retrieve(text, k=10)` measured per query, single CPU thread.

| dataset  | model           | NDCG@10 | Recall@10 | MRR@10 | p50 ms/q | p95 ms/q |
|----------|-----------------|--------:|----------:|-------:|---------:|---------:|
| scifact  | bm25s           |   0.710 |     0.830 |  0.678 |    0.097 |    0.131 |
| scifact  | splade-distill  |   0.743 |   **0.930** |  0.687 |    0.048 |    0.072 |
| scifact  | splade-gte      | **0.798** |     0.910 | **0.770** |  **0.048** |  **0.074** |
| nq       | bm25s           |   0.501 |     0.760 |  0.427 |    0.093 |    0.177 |
| nq       | splade-distill  |   0.707 |     0.850 |  0.668 |  **0.037** |  **0.044** |
| nq       | splade-gte      | **0.732** | **0.860** | **0.711** |    0.037 |    0.047 |
| fiqa     | bm25s           |   0.437 |     0.564 |  0.491 |    0.086 |    0.123 |
| fiqa     | splade-distill  |   0.482 |     0.553 |  0.545 |    0.044 |    0.067 |
| fiqa     | splade-gte      | **0.568** | **0.653** | **0.633** |  **0.043** |  **0.062** |
| nfcorpus | bm25s           |   0.325 |     0.115 |  0.500 |    0.089 |    0.120 |
| nfcorpus | splade-distill  |   0.347 |     0.143 |  0.499 |  **0.025** |    0.041 |
| nfcorpus | splade-gte      | **0.373** | **0.150** | **0.558** |    0.025 |  **0.038** |

splade-gte beats both bm25s and splade-distill on every NDCG@10 number, with the same query-time latency as splade-distill — the bigger encoder only costs at offline encode time, not query time. bm25s uses English stopwords + Porter stemming (its README's recommended setup). Reproduce:

```bash
uv run python benchmarks/bench_compare.py
```

Doc encoding for splade-easy runs offline once (`encode_corpus()`) and is the slow part — minutes on CPU for ~5k docs, seconds on a GPU. The retrieval times above are what users actually pay at query time.

## Development

```bash
uv sync --extra encode --extra eval --group dev
uv run pytest                                  # 51 tests, ~0.2s
uv run cython-lint src/splade_easy/_scoring.pyx
uv run python benchmarks/bench_query.py
```

Cython hot path is in `src/splade_easy/_scoring.pyx`. A pure-numpy reference at `src/splade_easy/_scoring_py.py` is used in tests for parity checks and as a fallback when the C extension isn't built. The build is driven by `setup.py` + `setuptools.build_meta`.

## License

MIT.
