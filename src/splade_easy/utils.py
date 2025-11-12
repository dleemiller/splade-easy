# src/splade_easy/utils.py

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def get_shard_paths(index_dir: Path, metadata: dict, *, strict: bool = False) -> list[Path]:
    """
    Get shard paths, preferring content-addressed over legacy.

    If strict=True, raise FileNotFoundError when expected shards are missing.
    Otherwise, log a warning and return only existing shards.
    """
    shard_hashes = metadata.get("shard_hashes", [])
    paths = [index_dir / f"{h}.fb" for h in shard_hashes]
    existing = [p for p in paths if p.exists()]
    missing = [p for p in paths if not p.exists()]

    if missing:
        msg = f"{len(missing)} shard(s) listed in metadata but missing on disk"
        logger.warning(msg)
        logger.debug("Missing shards: %s", ", ".join(p.name for p in missing))
        if strict:
            raise FileNotFoundError(f"{msg}: {', '.join(p.name for p in missing)}")

    return existing


def hash_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def extract_model_id(model: Any) -> str:
    """Extract model ID from sentence-transformers model."""
    try:
        # Check model_card_data (sentence-transformers v3+)
        if hasattr(model, "model_card_data") and model.model_card_data:
            mcd = model.model_card_data
            # Try model_id first
            if hasattr(mcd, "model_id") and mcd.model_id:
                return mcd.model_id
            # Fall back to base_model
            if hasattr(mcd, "base_model") and mcd.base_model:
                return mcd.base_model

        # Try legacy _model_card_text
        if hasattr(model, "_model_card_text") and model._model_card_text:
            lines = model._model_card_text.split("\n")
            for line in lines:
                if "model_id:" in line.lower():
                    return line.split(":", 1)[1].strip()

        # Try _model_name or model_name attributes
        if hasattr(model, "_model_name"):
            return model._model_name
        if hasattr(model, "model_name"):
            return model.model_name
    except Exception:
        pass

    return "unknown"


def extract_splade_vectors(encoding: Any, threshold: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract token IDs and weights from SPLADE encoding.

    For DENSE vectors, only keeps values above threshold (default: any positive value).
    For SPARSE format, returns as-is.
    """
    # Format 1: Dict with 'indices' and 'values' (already sparse)
    if isinstance(encoding, dict) and "indices" in encoding and "values" in encoding:
        return (
            np.array(encoding["indices"], dtype=np.uint32),
            np.array(encoding["values"], dtype=np.float32),
        )

    # Format 2: Sparse tensor (already sparse)
    if hasattr(encoding, "indices") and hasattr(encoding, "values"):
        return (
            np.array(encoding.indices, dtype=np.uint32),
            np.array(encoding.values, dtype=np.float32),
        )

    # Format 3 & 4: Dense array/tensor - need to sparsify
    encoding_np = None

    if isinstance(encoding, np.ndarray):
        encoding_np = encoding
    else:
        # Try torch tensor
        try:
            import torch

            if isinstance(encoding, torch.Tensor):
                encoding_np = encoding.cpu().numpy()
        except ImportError:
            pass

    if encoding_np is not None:
        if encoding_np.ndim == 2:
            encoding_np = encoding_np[0]  # Take first batch

        # For SPLADE, only positive activations matter (ReLU output)
        # Use threshold to filter to truly sparse representation
        mask = encoding_np > threshold
        indices = np.where(mask)[0]
        values = encoding_np[indices]

        return (indices.astype(np.uint32), values.astype(np.float32))

    raise ValueError(
        f"Unsupported encoding format: {type(encoding)}. "
        "Expected dict with 'indices'/'values', sparse tensor, or dense array."
    )
