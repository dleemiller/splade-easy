"""splade-easy: simple inference-free SPLADE retrieval."""

from __future__ import annotations

from .retriever import SpladeRetriever
from .sparse import SparseCorpus

__version__ = "0.2.0"

__all__ = ["SpladeRetriever", "SparseCorpus", "encode_corpus", "__version__"]


def encode_corpus(*args, **kwargs):
    """Encode a list of texts to a sparse corpus. Lazy import — only pulls torch on first call."""
    from .encoder import encode_corpus as _enc

    return _enc(*args, **kwargs)
