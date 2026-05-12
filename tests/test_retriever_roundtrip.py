"""End-to-end retriever roundtrip with a mocked tokenizer + IDF (no HF network)."""
from __future__ import annotations

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from splade_easy import SpladeRetriever, sparse
from splade_easy.tokenizer import QueryTokenizer


@pytest.fixture
def tiny_setup(tmp_path):
    """Build a 4-doc index with a hand-written tokenizer and IDF — no network."""
    vocab = {"[UNK]": 0, "cat": 1, "dog": 2, "fish": 3, "the": 4}
    vocab_size = len(vocab)
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok_path = tmp_path / "_tok.json"
    tok.save(str(tok_path))

    # docs:
    # doc 0: cat (1.0), the (0.2)
    # doc 1: dog (1.0), the (0.2)
    # doc 2: cat (0.5), dog (0.5)
    # doc 3: fish (1.0)
    ids = [
        np.array([1, 4], dtype=np.int32),
        np.array([2, 4], dtype=np.int32),
        np.array([1, 2], dtype=np.int32),
        np.array([3], dtype=np.int32),
    ]
    ws = [
        np.array([1.0, 0.2], dtype=np.float32),
        np.array([1.0, 0.2], dtype=np.float32),
        np.array([0.5, 0.5], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
    ]
    sc = sparse.from_per_doc(ids, ws, vocab_size=vocab_size)

    # IDF: 'the' gets 0 (stopword), the rest get 1.0
    query_weights = np.array([0.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32)

    # Build retriever directly, skipping HF fetch
    r = SpladeRetriever(model="dummy/model")
    ip, idx, d = sparse.csr_to_csc(sc)
    r._indptr = ip
    r._indices = idx
    r._data = d
    r._n_docs = sc.n_docs
    r._vocab_size = sc.vocab_size
    r._query_weights = query_weights
    r._tokenizer = QueryTokenizer.from_file(tok_path)

    corpus = ["a cat sat", "a dog ran", "cat and dog", "fish swim"]
    return r, corpus, tmp_path


def test_single_query_returns_top1_correctly(tiny_setup):
    r, _, _ = tiny_setup
    ids, scores = r.retrieve("cat", k=1)
    # Doc 0 (cat=1.0) beats doc 2 (cat=0.5)
    assert ids.tolist() == [0]
    np.testing.assert_allclose(scores, [1.0])


def test_batch_query(tiny_setup):
    r, _, _ = tiny_setup
    ids, scores = r.retrieve(["cat", "dog"], k=2)
    assert ids.shape == (2, 2)
    # cat → doc 0 (1.0), doc 2 (0.5)
    assert ids[0].tolist() == [0, 2]
    # dog → doc 1 (1.0), doc 2 (0.5)
    assert ids[1].tolist() == [1, 2]


def test_query_with_dropped_stopword(tiny_setup):
    r, _, _ = tiny_setup
    # "the cat" — 'the' has zero IDF, only 'cat' contributes
    ids, scores = r.retrieve("the cat", k=1)
    assert ids.tolist() == [0]


def test_save_load_roundtrip(tiny_setup):
    r, corpus, tmp_path = tiny_setup
    idx_dir = tmp_path / "idx"
    r.save(idx_dir, corpus=corpus)

    # No-mmap reload
    r2 = SpladeRetriever.load(idx_dir, mmap=False, load_corpus=True)
    assert r2.model_id == "dummy/model"
    assert r2._n_docs == 4
    assert r2._corpus[2]["text"] == "cat and dog"

    ids, scores = r2.retrieve("cat", k=2)
    assert ids.tolist() == [0, 2]
    np.testing.assert_allclose(scores, [1.0, 0.5])


def test_save_load_mmap(tiny_setup):
    r, corpus, tmp_path = tiny_setup
    idx_dir = tmp_path / "idx"
    r.save(idx_dir, corpus=corpus)

    r2 = SpladeRetriever.load(idx_dir, mmap=True, load_corpus=False)
    # mmap_mode='r' yields memmap arrays
    assert isinstance(r2._indptr, np.memmap)
    ids, _ = r2.retrieve(["cat", "fish"], k=2)
    assert ids[0, 0] == 0
    assert ids[1, 0] == 3


def test_return_docs_returns_corpus_items(tiny_setup):
    r, corpus, tmp_path = tiny_setup
    idx_dir = tmp_path / "idx"
    r.save(idx_dir, corpus=corpus)

    r2 = SpladeRetriever.load(idx_dir, load_corpus=True)
    results, _ = r2.retrieve("cat", k=2, return_docs=True)
    # Returned items are JSON-deserialized dicts (since corpus was strings, wrapped in {"text": ...})
    assert results[0] == {"text": "a cat sat"}
    assert results[1] == {"text": "cat and dog"}
