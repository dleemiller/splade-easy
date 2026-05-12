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
    trust_remote_code: bool | None = None,
    max_seq_length: int | None = None,
) -> sparse.SparseCorpus:
    """Encode a list of texts to a sparse corpus using a SPLADE document encoder.

    For known models, `trust_remote_code` defaults to whatever the registry says
    (e.g. True for the gte family). Pass it explicitly to override. Unknown
    models default to False (safe).
    """
    model_id = model or models.DEFAULT_MODEL
    spec = models.resolve(model_id)
    trc = spec.trust_remote_code if trust_remote_code is None else trust_remote_code

    # Lazy import: torch + sentence-transformers are heavy extras
    from sentence_transformers import SparseEncoder

    model_kwargs: dict = {}
    if spec.code_revision is not None:
        model_kwargs["code_revision"] = spec.code_revision

    # NOTE: pass an empty `model_kwargs={}` rather than `None` — SBERT's SparseEncoder
    # routes through a different code path when model_kwargs is None, which on the gte
    # backbone yields all-NaN outputs even with our position_ids / weight-tying patches
    # applied. An empty dict gives the same effective behavior as not passing it.
    enc = SparseEncoder(
        model_id,
        device=device,
        trust_remote_code=trc,
        model_kwargs=model_kwargs,
    )
    if max_seq_length is not None:
        enc.max_seq_length = max_seq_length

    _patch_custom_modeling(enc)
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


def _patch_custom_modeling(enc) -> None:
    """Repair two transformers/custom-code interop issues that surface on the gte backbone.

    1. `position_ids` registered with `persistent=False` comes back filled with garbage
       values after the model loads. Reset any 1D `position_ids` buffer to `arange(n)`.
       Safe no-op for already-correct buffers.

    2. MLM-head output weights are expected to be tied to the input word embeddings
       (the checkpoint deliberately omits them) but the custom `NewConfig` lacks
       `tie_word_embeddings`, so transformers' auto-tie skips them and the head is
       left at random init. Copy the input embedding weight into the head's
       parameter storage via `.data.copy_()` — assigning the Parameter object
       directly gets cloned away by something downstream in SBERT's setup, while
       writing the storage in-place survives.
    """
    import torch

    # Pass 1: fix any 1D `position_ids` buffer in place.
    for m in enc.modules():
        pi = getattr(m, "position_ids", None)
        if isinstance(pi, torch.Tensor) and pi.dim() == 1:
            pi.copy_(torch.arange(pi.size(0), dtype=pi.dtype, device=pi.device))

    # Pass 2: copy input-embedding weights into output-embedding weights for
    # modules that expose both. We use `.data.copy_()` (in-place) rather than
    # `.weight =` (parameter reassignment) because downstream code in SBERT
    # appears to clone reassigned parameters away.
    for m in enc.modules():
        get_in = getattr(m, "get_input_embeddings", None)
        get_out = getattr(m, "get_output_embeddings", None)
        if not (callable(get_in) and callable(get_out)):
            continue
        try:
            inp = get_in()
            outp = get_out()
        except Exception:
            continue
        if (
            inp is not None
            and outp is not None
            and hasattr(inp, "weight")
            and hasattr(outp, "weight")
            and inp.weight.shape == outp.weight.shape
        ):
            with torch.no_grad():
                outp.weight.data.copy_(inp.weight.data)


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
