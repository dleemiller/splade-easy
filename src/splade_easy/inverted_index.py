import logging
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Posting:
    """A single posting in the inverted index"""

    doc_offset: int  # Offset in shard file for zero-copy reading
    weight: float  # Token weight in document


class InvertedIndexBuilder:
    """Builds inverted index during indexing"""

    def __init__(self):
        self._index: dict[int, list[tuple[int, float]]] = {}

    def add_document(self, doc_offset: int, token_ids: np.ndarray, weights: np.ndarray):
        for token_id, weight in zip(token_ids, weights):
            if token_id not in self._index:
                self._index[token_id] = []
            self._index[token_id].append((doc_offset, weight))

    def finalize(self) -> dict[int, list[Posting]]:
        result = {}
        for token_id, postings in self._index.items():
            postings.sort(key=lambda x: x[0])
            result[token_id] = [Posting(offset, weight) for offset, weight in postings]
        return result


class InvertedIndexWriter:
    """Writes inverted index to disk in compact binary format"""

    def __init__(self, path: str):
        self.path = Path(path)
        self._buffer = bytearray()

    def write(self, index: dict[int, list[Posting]]):
        self._buffer.extend(struct.pack("I", len(index)))

        for token_id, postings in sorted(index.items()):
            self._buffer.extend(struct.pack("I", token_id))
            self._buffer.extend(struct.pack("I", len(postings)))

            for posting in postings:
                self._buffer.extend(struct.pack("I", posting.doc_offset))
                self._buffer.extend(struct.pack("f", posting.weight))

        with open(self.path, "wb") as f:
            f.write(self._buffer)

    def close(self):
        pass


class InvertedIndexReader:
    """
    Memory-mapped reader for inverted index files (.inv).

    Provides low-level access to on-disk posting lists for efficient lookup.
    Used by NumbaCache to build optimized in-memory structures for search.

    File format:
        [num_tokens: u32]
        For each token:
            [token_id: u32]
            [num_postings: u32]
            [postings: (doc_offset: u32, weight: f32) × num_postings]

    Note: This is a read-only interface. All search logic is in NumbaCache.
    """

    def __init__(self, path: str, shard_path: Path):
        """
        Initialize memory-mapped reader for an inverted index file.

        Args:
            path: Path to .inv file
            shard_path: Path to corresponding .fb shard file (for reference)
        """
        self.path = Path(path)
        self.shard_path = shard_path
        self._mmap = None
        self._token_map: dict[int, tuple[int, int]] = {}  # token_id → (offset, num_postings)

        if self.path.exists():
            self._load()

    def _load(self):
        """Load index header and build token map (postings loaded on-demand)."""
        with open(self.path, "rb") as f:
            self._mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            num_tokens = struct.unpack_from("I", self._mmap, 0)[0]
            offset = 4

            for _ in range(num_tokens):
                token_id = struct.unpack_from("I", self._mmap, offset)[0]
                offset += 4
                num_postings = struct.unpack_from("I", self._mmap, offset)[0]
                offset += 4
                self._token_map[token_id] = (offset, num_postings)
                offset += num_postings * 8  # Each posting is 8 bytes (u32 + f32)

    def get_postings_location(self, token_id: int) -> tuple[int, int] | None:
        """
        Get the location of posting list for a token.

        Args:
            token_id: Token ID to lookup

        Returns:
            (offset, num_postings) if token exists, None otherwise
        """
        return self._token_map.get(token_id)

    def read_posting_at(self, offset: int) -> tuple[int, float]:
        """
        Read a single posting at the given byte offset.

        Args:
            offset: Byte offset in mmap

        Returns:
            (doc_offset, weight) tuple
        """
        doc_offset = struct.unpack_from("I", self._mmap, offset)[0]
        weight = struct.unpack_from("f", self._mmap, offset + 4)[0]
        return doc_offset, weight

    def close(self):
        """Close memory-mapped file."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
