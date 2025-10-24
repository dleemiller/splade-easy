# src/splade_easy/__init__.py

from .index import Document, SpladeIndex
from .retriever import SearchResult
from .utils import extract_splade_vectors

__version__ = "0.1.0"
__all__ = ["SpladeIndex", "Document", "SearchResult", "extract_splade_vectors"]
