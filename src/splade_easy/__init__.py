# src/splade_easy/__init__.py

from .document import Document
from .index import Index
from .splade_index import SpladeIndex
from .retriever import SearchResult
from .utils import extract_splade_vectors

__version__ = "0.1.0"
__all__ = ["SpladeIndex","Index", "Document", "SearchResult", "extract_splade_vectors"]
