"""Tests for query tokenization wrapper and IDF application."""

from __future__ import annotations

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from splade_easy.tokenizer import QueryTokenizer, apply_idf


@pytest.fixture
def tiny_tokenizer(tmp_path):
    vocab = {"[UNK]": 0, "cat": 1, "dog": 2, "fish": 3, "the": 4, "small": 5}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    return QueryTokenizer.from_file(path)


def test_encode_returns_unique_token_ids(tiny_tokenizer):
    ids = tiny_tokenizer.encode("the cat the dog cat")
    # unique: cat (1), dog (2), the (4)
    assert ids.tolist() == [1, 2, 4]
    assert ids.dtype == np.int32


def test_encode_batch(tiny_tokenizer):
    out = tiny_tokenizer.encode_batch(["cat fish", "dog dog"])
    assert out[0].tolist() == [1, 3]
    assert out[1].tolist() == [2]


def test_token_to_id(tiny_tokenizer):
    assert tiny_tokenizer.token_to_id("cat") == 1
    assert tiny_tokenizer.token_to_id("nope") is None


def test_apply_idf_drops_zero_weights():
    idf = np.array([0.0, 1.5, 0.0, 2.0, 0.5, 0.0], dtype=np.float32)
    ids = np.array([1, 2, 3], dtype=np.int32)  # token 2 has zero IDF
    out_ids, out_ws = apply_idf(ids, idf)
    assert out_ids.tolist() == [1, 3]
    np.testing.assert_allclose(out_ws, [1.5, 2.0])


def test_apply_idf_empty():
    idf = np.zeros(5, dtype=np.float32)
    out_ids, out_ws = apply_idf(np.array([], dtype=np.int32), idf)
    assert out_ids.size == 0
    assert out_ws.size == 0
