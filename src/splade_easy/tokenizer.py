"""Query-side tokenization + IDF lookup. No torch dependency."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


class QueryTokenizer:
    """Thin wrapper over a HF fast tokenizer for query-only use."""

    def __init__(self, tokenizer: Tokenizer):
        self._tok = tokenizer

    @classmethod
    def from_file(cls, path: str | Path) -> "QueryTokenizer":
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: str | Path) -> None:
        self._tok.save(str(path))

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def token_to_id(self, token: str) -> int | None:
        return self._tok.token_to_id(token)

    def encode(self, text: str) -> np.ndarray:
        """Return unique non-special token IDs for one query."""
        enc = self._tok.encode(text, add_special_tokens=False)
        return np.unique(np.asarray(enc.ids, dtype=np.int32))

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        encs = self._tok.encode_batch(texts, add_special_tokens=False)
        return [np.unique(np.asarray(e.ids, dtype=np.int32)) for e in encs]


def apply_idf(token_ids: np.ndarray, idf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply per-token IDF weights. Drops zero-weighted tokens (OOV / stopwords)."""
    if len(token_ids) == 0:
        return token_ids.astype(np.int32, copy=False), np.zeros(0, dtype=np.float32)
    weights = idf[token_ids]
    mask = weights > 0
    return token_ids[mask].astype(np.int32, copy=False), weights[mask].astype(np.float32, copy=False)
