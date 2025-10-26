# src/splade_easy/shard.py

import struct
from collections.abc import Iterator
from pathlib import Path

import flatbuffers
import numpy as np

# Import generated FlatBuffers code
from .SpladeEasy import Document, KeyValue


class ShardWriter:
    """Writes documents to a shard file."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.f = open(path, "ab")  # noqa: SIM115 - file must stay open for appending
        self._size = 0

    def append(
        self, doc_id: str, text: str, metadata: dict, token_ids: np.ndarray, weights: np.ndarray
    ) -> None:
        """Append a document to the shard."""
        builder = flatbuffers.Builder(1024)

        # Build token_ids and weights arrays
        Document.DocumentStartTokenIdsVector(builder, len(token_ids))
        for tid in reversed(token_ids):
            builder.PrependUint32(int(tid))
        token_ids_vec = builder.EndVector()

        Document.DocumentStartWeightsVector(builder, len(weights))
        for w in reversed(weights):
            builder.PrependFloat32(float(w))
        weights_vec = builder.EndVector()

        # Build metadata
        meta_offsets = []
        for k, v in metadata.items():
            key_off = builder.CreateString(k)
            val_off = builder.CreateString(str(v))
            KeyValue.KeyValueStart(builder)
            KeyValue.KeyValueAddKey(builder, key_off)
            KeyValue.KeyValueAddValue(builder, val_off)
            meta_offsets.append(KeyValue.KeyValueEnd(builder))

        Document.DocumentStartMetadataVector(builder, len(meta_offsets))
        for off in reversed(meta_offsets):
            builder.PrependUOffsetTRelative(off)
        metadata_vec = builder.EndVector()

        # Build document
        doc_id_off = builder.CreateString(doc_id)
        text_off = builder.CreateString(text)

        Document.DocumentStart(builder)
        Document.DocumentAddDocId(builder, doc_id_off)
        Document.DocumentAddText(builder, text_off)
        Document.DocumentAddMetadata(builder, metadata_vec)
        Document.DocumentAddTokenIds(builder, token_ids_vec)
        Document.DocumentAddWeights(builder, weights_vec)
        doc_off = Document.DocumentEnd(builder)

        builder.Finish(doc_off)

        # Write length-prefixed message
        buf = bytes(builder.Output())
        self.f.write(struct.pack("I", len(buf)))
        self.f.write(buf)
        self._size += 4 + len(buf)

    def size(self) -> int:
        """Current shard size in bytes."""
        return self._size

    def close(self) -> None:
        self.f.close()


class ShardReader:
    """Reads documents from a shard file."""

    def __init__(self, path: str):
        self.path = Path(path)

        # Check if file exists and is not empty
        if not self.path.exists():
            self.data = None
            self.size = 0
            return

        file_size = self.path.stat().st_size
        if file_size == 0:
            self.data = None
            self.size = 0
        else:
            self.data = np.memmap(path, dtype="uint8", mode="r")
            self.size = len(self.data)

    def scan(self, load_text: bool = True) -> Iterator[dict]:
        """Scan all documents in the shard."""
        # Handle empty or missing shard
        if self.data is None or self.size == 0:
            return

        offset = 0

        while offset < self.size:
            # Read message length
            if offset + 4 > self.size:
                break

            msg_len = struct.unpack_from("I", self.data, offset)[0]
            offset += 4

            if offset + msg_len > self.size:
                break

            # Parse FlatBuffer
            doc = Document.Document.GetRootAs(self.data[offset:], 0)

            # Extract metadata
            metadata = {}
            for i in range(doc.MetadataLength()):
                kv = doc.Metadata(i)
                metadata[kv.Key().decode()] = kv.Value().decode()

            # Extract sparse vector - FAST!
            token_ids = doc.TokenIdsAsNumpy()
            weights = doc.WeightsAsNumpy()

            yield {
                "doc_id": doc.DocId().decode(),
                "text": doc.Text().decode() if load_text else None,
                "metadata": metadata,
                "token_ids": token_ids,
                "weights": weights,
            }

            offset += msg_len

    def close(self) -> None:
        if self.data is not None:
            del self.data
