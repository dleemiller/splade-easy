# src/splade_easy/utils.py

import numpy as np


def extract_splade_vectors(encoding) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract token IDs and weights from sentence-transformers SPLADE encoding.

    Args:
        encoding: Output from model.encode() with SPLADE model

    Returns:
        (token_ids, weights) as numpy arrays
    """
    # Handle different return formats from sentence-transformers

    # Format 1: Dict with 'indices' and 'values'
    if isinstance(encoding, dict) and "indices" in encoding and "values" in encoding:
        return (
            np.array(encoding["indices"], dtype=np.uint32),
            np.array(encoding["values"], dtype=np.float32),
        )

    # Format 2: Sparse tensor or array-like
    if hasattr(encoding, "indices") and hasattr(encoding, "values"):
        return (
            np.array(encoding.indices, dtype=np.uint32),
            np.array(encoding.values, dtype=np.float32),
        )

    # Format 3: Dense array (convert to sparse by taking non-zero)
    if isinstance(encoding, np.ndarray):
        nonzero_indices = np.nonzero(encoding)[0]
        nonzero_values = encoding[nonzero_indices]
        return (nonzero_indices.astype(np.uint32), nonzero_values.astype(np.float32))

    # Format 4: Torch tensor
    try:
        import torch

        if isinstance(encoding, torch.Tensor):
            encoding_np = encoding.cpu().numpy()
            if encoding_np.ndim == 2:
                encoding_np = encoding_np[0]  # Take first batch
            nonzero_indices = np.nonzero(encoding_np)[0]
            nonzero_values = encoding_np[nonzero_indices]
            return (nonzero_indices.astype(np.uint32), nonzero_values.astype(np.float32))
    except ImportError:
        pass

    raise ValueError(
        f"Unsupported encoding format: {type(encoding)}. "
        "Expected dict with 'indices'/'values', sparse tensor, or dense array."
    )
