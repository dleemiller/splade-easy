"""Offline document encoding (uses sentence-transformers) + query-side IDF fetch (no torch)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import models, sparse
from .tokenizer import QueryTokenizer


def encode_corpus(
    corpus: Sequence[str],
    model: str | None = None,
    batch_size: int = 32,
    device: str | None = None,
    show_progress: bool = True,
) -> sparse.SparseCorpus:
    """Encode a list of texts to a sparse corpus using a SPLADE document encoder."""
    model_id = model or models.DEFAULT_MODEL

    # Lazy import: torch + sentence-transformers are heavy extras
    from sentence_transformers import SparseEncoder

    enc = SparseEncoder(model_id, device=device)
    embs = enc.encode_document(
        list(corpus),
        batch_size=batch_size,
        convert_to_tensor=False,
        show_progress_bar=show_progress,
    )

    vocab_size = _vocab_size(enc)

    token_ids_list: list[np.ndarray] = []
    weights_list: list[np.ndarray] = []
    for emb in embs:
        coo = emb.coalesce()
        idx = coo.indices()
        # 1D sparse vector: indices shape is (1, nnz); flatten to (nnz,)
        if idx.ndim == 2:
            idx = idx[0]
        token_ids_list.append(idx.cpu().numpy().astype(np.int32))
        weights_list.append(coo.values().cpu().numpy().astype(np.float32))

    return sparse.from_per_doc(token_ids_list, weights_list, vocab_size)


def _vocab_size(encoder) -> int:
    """Try several routes to determine the encoder's vocab size."""
    tok = getattr(encoder, "tokenizer", None)
    if tok is not None:
        for attr in ("vocab_size", "get_vocab_size"):
            v = getattr(tok, attr, None)
            if callable(v):
                try:
                    return int(v())
                except TypeError:
                    pass
            elif v is not None:
                return int(v)
        try:
            return len(tok)
        except (AttributeError, TypeError):
            pass
    # Walk modules for an HF config
    for m in encoder.modules():
        cfg = getattr(m, "config", None)
        if cfg is not None and hasattr(cfg, "vocab_size"):
            return int(cfg.vocab_size)
    raise RuntimeError("Could not determine vocab_size from encoder")


def fetch_tokenizer(model_id: str) -> QueryTokenizer:
    """Download tokenizer.json from a HF model repo and wrap it."""
    from huggingface_hub import hf_hub_download

    tok_path = hf_hub_download(model_id, "tokenizer.json")
    return QueryTokenizer.from_file(tok_path)


_IDF_CANDIDATE_PATHS = [
    "idf.json",
    "IDF.json",
    "0_IDF/idf.json",
    "1_IDF/idf.json",
    "2_IDF/idf.json",
    "3_IDF/idf.json",
    "query_idf/idf.json",
]


def fetch_query_weights(
    model_id: str,
    tokenizer: QueryTokenizer,
    vocab_size: int,
) -> np.ndarray:
    """Download per-token IDF weights from HF and materialize as a dense `(vocab_size,) float32` array."""
    idf_map = _load_idf_map(model_id)
    weights = np.zeros(vocab_size, dtype=np.float32)
    n_assigned = 0
    for key, w in idf_map.items():
        if isinstance(key, str):
            tid = tokenizer.token_to_id(key)
            if tid is None:
                continue
        else:
            tid = int(key)
        if 0 <= tid < vocab_size:
            weights[tid] = float(w)
            n_assigned += 1
    if n_assigned == 0:
        raise RuntimeError(f"IDF file for {model_id} had no usable entries")
    return weights


def _load_idf_map(model_id: str) -> dict:
    from huggingface_hub import hf_hub_download, list_repo_files
    from huggingface_hub.utils import EntryNotFoundError

    for path in _IDF_CANDIDATE_PATHS:
        try:
            local = hf_hub_download(model_id, path)
        except EntryNotFoundError:
            continue
        except Exception:
            continue
        with open(local) as f:
            return json.load(f)

    # Fallback: list repo files and pattern-match
    try:
        files = list_repo_files(model_id)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not list files in {model_id} while searching for an IDF JSON: {e}"
        ) from e
    for f in files:
        base = Path(f).name.lower()
        if "idf" in base and base.endswith(".json"):
            local = hf_hub_download(model_id, f)
            with open(local) as fp:
                return json.load(fp)
    raise FileNotFoundError(f"No IDF JSON file found in {model_id}")
