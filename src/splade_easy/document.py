from dataclasses import dataclass

import numpy as np


@dataclass
class Document:
    doc_id: str
    text: str | bytes
    metadata: dict[str, str]
    token_ids: np.ndarray
    weights: np.ndarray
