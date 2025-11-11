from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchResult:
    doc_id: str
    score: float
    metadata: dict
    text: Optional[str] = None
