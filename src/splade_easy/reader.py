import heapq
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .index import Index
from .scoring import compute_splade_score, ensure_sorted_splade_vector
from .utils import extract_model_id, extract_splade_vectors


@dataclass
class SearchResult:
    doc_id: str
    score: float
    metadata: dict
    text: Optional[str] = None


class IndexReader:
    def __init__(self, index: Index):
        self.index = index
        # Auto-load if memory mode
        if self.index.memory:
            self.index.load()
    
    def load(self):
        """Load index into memory (optional, auto-called for memory mode)"""
        self.index.open()
        return self

    # Search API - unified method
    def search(
        self,
        query: Optional[str] = None,
        query_tokens: Optional[np.ndarray] = None,
        query_weights: Optional[np.ndarray] = None,
        model=None,
        top_k: int = 10,
        return_text: bool = False,
    ) -> list[SearchResult]:
        """Unified search method - handles both text and vector queries"""
        # Handle text query
        if query is not None:
            if model is None:
                raise ValueError("Model required for text search")

            # Model validation
            index_model_id = self.index.manifest.model_id
            query_model_id = extract_model_id(model)
            if query_model_id != "unknown" and index_model_id != query_model_id:
                raise ValueError(
                    f"Model mismatch! Index: {index_model_id}, Query: {query_model_id}"
                )

            # Encode query
            encoding = model.encode(query)
            query_tokens, query_weights = extract_splade_vectors(encoding)

        # Handle vector query
        elif query_tokens is not None and query_weights is not None:
            pass  # Use provided vectors
        else:
            raise ValueError("Must provide either query text or query_tokens/weights")

        # Ensure query vectors are sorted for optimal scoring
        query_tokens, query_weights = ensure_sorted_splade_vector(
            query_tokens, query_weights, deduplicate=True
        )

        # Search all shards and merge results
        heap, idx = [], 0
        for shard_path in self.index.iter_shards():
            for doc in self.index.iter_docs(shard_path, load_text=return_text):
                # Skip deleted documents
                if self._is_deleted(doc["doc_id"]):
                    continue

                score = compute_splade_score(
                    doc_tokens=doc["token_ids"],
                    doc_weights=doc["weights"],
                    query_tokens=query_tokens,
                    query_weights=query_weights,
                )

                if score <= 0:
                    continue

                result = SearchResult(
                    doc_id=doc["doc_id"],
                    score=score,
                    metadata=doc["metadata"],
                    text=doc.get("text"),
                )

                if len(heap) < top_k:
                    heapq.heappush(heap, (score, idx, result))
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, (score, idx, result))
                idx += 1

        return [r for _, __, r in sorted(heap, key=lambda x: x[0], reverse=True)]

    # Convenience methods for specific search types
    def search_text(self, query: str, model, top_k=10, return_text=False):
        """Search using text query"""
        return self.search(query=query, model=model, top_k=top_k, return_text=return_text)

    def search_vectors(
        self, query_tokens: np.ndarray, query_weights: np.ndarray, top_k=10, return_text=False
    ):
        """Search using pre-encoded SPLADE vectors"""
        return self.search(
            query_tokens=query_tokens,
            query_weights=query_weights,
            top_k=top_k,
            return_text=return_text,
        )

    def get(self, doc_id: str) -> Optional[dict]:
        """Get document by ID"""
        if self._is_deleted(doc_id):
            return None
        return self.index.get(doc_id)

    def get_batch(self, doc_ids: list[str]) -> list[Optional[dict]]:
        """Get multiple documents by ID"""
        return [self.get(doc_id) for doc_id in doc_ids]

    # Internal helpers
    def _is_deleted(self, doc_id: str) -> bool:
        """Check if document is deleted"""
        deleted_ids = self.index._load_deleted_ids()
        return doc_id in deleted_ids
