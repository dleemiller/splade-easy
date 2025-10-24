# src/splade_easy/utils.py

import numpy as np


def extract_model_id(model) -> str:
    """Extract model ID from sentence-transformers model."""
    try:
        if hasattr(model, "model_card_data") and model.model_card_data:
            if hasattr(model.model_card_data, "model_id"):
                return model.model_card_data.model_id
            if isinstance(model.model_card_data, dict) and "model_id" in model.model_card_data:
                return model.model_card_data["model_id"]

        if hasattr(model, "_model_card_text") and model._model_card_text:
            lines = model._model_card_text.split("\n")
            for line in lines:
                if "model_id:" in line.lower():
                    return line.split(":", 1)[1].strip()

        if hasattr(model, "_model_name"):
            return model._model_name
        if hasattr(model, "model_name"):
            return model.model_name
    except Exception:
        pass

    return "unknown"


def extract_splade_vectors(encoding, threshold: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
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
