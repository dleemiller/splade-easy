"""SpladeRetriever — the main public class.

Index-time: needs sparse doc embeddings + tokenizer + IDF weights (fetched from HF).
Query-time: tokenize + IDF lookup + score+topk over the CSC inverted index. No torch.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import overload

import numpy as np

from . import models, sparse
from .tokenizer import QueryTokenizer, apply_idf

try:
    from . import _scoring as _SCORING  # type: ignore[attr-defined]
except ImportError:
    from . import _scoring_py as _SCORING


_VERSION = "0.2.0"


class SpladeRetriever:
    """Inverted-index SPLADE retriever. Build with `index()`, persist with `save()`/`load()`, query with `retrieve()`."""

    def __init__(self, model: str | None = None):
        self.model_id: str = model or models.DEFAULT_MODEL
        # Internal CSC arrays (term-major):
        self._indptr: np.ndarray | None = None
        self._indices: np.ndarray | None = None
        self._data: np.ndarray | None = None
        self._query_weights: np.ndarray | None = None
        self._tokenizer: QueryTokenizer | None = None
        self._n_docs: int = 0
        self._vocab_size: int = 0
        self._corpus: list | None = None

    # ---- build ----

    def index(self, sparse_docs: sparse.SparseCorpus) -> None:
        """Build the inverted index from sparse doc embeddings."""
        indptr_c, indices_c, data_c = sparse.csr_to_csc(sparse_docs)
        self._indptr = indptr_c
        self._indices = indices_c
        self._data = data_c
        self._n_docs = sparse_docs.n_docs
        self._vocab_size = sparse_docs.vocab_size

        from .encoder import fetch_query_weights, fetch_tokenizer

        self._tokenizer = fetch_tokenizer(self.model_id)
        self._query_weights = fetch_query_weights(self.model_id, self._tokenizer, self._vocab_size)

    # ---- persist ----

    def save(self, path: str | Path, corpus: Sequence | None = None) -> None:
        if self._indptr is None:
            raise RuntimeError("Nothing to save — call index() first")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        np.save(path / "indptr.npy", self._indptr)
        np.save(path / "indices.npy", self._indices)
        np.save(path / "data.npy", self._data)
        np.save(path / "query_weights.npy", self._query_weights)

        tok_dir = path / "tokenizer"
        tok_dir.mkdir(exist_ok=True)
        assert self._tokenizer is not None
        self._tokenizer.save(tok_dir / "tokenizer.json")

        params = {
            "model_id": self.model_id,
            "n_docs": self._n_docs,
            "vocab_size": self._vocab_size,
            "splade_easy_version": _VERSION,
            "dtype_data": str(self._data.dtype),
            "dtype_indices": str(self._indices.dtype),
            "dtype_indptr": str(self._indptr.dtype),
        }
        (path / "params.json").write_text(json.dumps(params, indent=2))

        if corpus is not None:
            with (path / "corpus.jsonl").open("w") as f:
                for item in corpus:
                    if isinstance(item, str):
                        f.write(json.dumps({"text": item}) + "\n")
                    else:
                        f.write(json.dumps(item) + "\n")

    @classmethod
    def load(
        cls,
        path: str | Path,
        mmap: bool = True,
        load_corpus: bool = False,
    ) -> SpladeRetriever:
        path = Path(path)
        params = json.loads((path / "params.json").read_text())

        inst = cls(model=params["model_id"])
        mmap_mode = "r" if mmap else None
        inst._indptr = np.load(path / "indptr.npy", mmap_mode=mmap_mode)
        inst._indices = np.load(path / "indices.npy", mmap_mode=mmap_mode)
        inst._data = np.load(path / "data.npy", mmap_mode=mmap_mode)
        inst._query_weights = np.load(path / "query_weights.npy")
        inst._n_docs = int(params["n_docs"])
        inst._vocab_size = int(params["vocab_size"])

        inst._tokenizer = QueryTokenizer.from_file(path / "tokenizer" / "tokenizer.json")

        if load_corpus:
            corpus_path = path / "corpus.jsonl"
            if corpus_path.exists():
                with corpus_path.open() as f:
                    inst._corpus = [json.loads(line) for line in f if line.strip()]

        return inst

    # ---- query ----

    @overload
    def retrieve(
        self, queries: str, k: int = ..., return_docs: bool = ...
    ) -> tuple[np.ndarray, np.ndarray]: ...
    @overload
    def retrieve(
        self, queries: list[str], k: int = ..., return_docs: bool = ...
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def retrieve(
        self,
        queries: str | list[str],
        k: int = 10,
        return_docs: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve top-k docs for one or more queries.

        Returns (results, scores). For a single query, both are 1D of length min(k, n_docs).
        For a batch, both are 2D shape (n_queries, min(k, n_docs)).
        `results` contains corpus indices, or corpus items if `return_docs=True` and corpus was loaded.
        """
        if self._indptr is None or self._tokenizer is None or self._query_weights is None:
            raise RuntimeError("Retriever not initialized — call index() or load() first")

        single = isinstance(queries, str)
        queries_list = [queries] if single else list(queries)

        token_lists = self._tokenizer.encode_batch(queries_list)
        q_ids_list: list[np.ndarray] = []
        q_weights_list: list[np.ndarray] = []
        for tids in token_lists:
            ids, ws = apply_idf(tids, self._query_weights)
            q_ids_list.append(ids)
            q_weights_list.append(ws)

        k_eff = min(k, self._n_docs)
        results, scores = _SCORING.score_topk_batch(
            self._indptr,
            self._indices,
            self._data,
            q_ids_list,
            q_weights_list,
            self._n_docs,
            k_eff,
        )

        if return_docs and self._corpus is not None:
            doc_results = np.empty(results.shape, dtype=object)
            for i in range(results.shape[0]):
                for j in range(results.shape[1]):
                    doc_results[i, j] = self._corpus[int(results[i, j])]
            results = doc_results

        if single:
            return results[0], scores[0]
        return results, scores
