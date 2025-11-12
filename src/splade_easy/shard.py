import struct
from collections.abc import Iterator
from pathlib import Path

import flatbuffers
import numpy as np

from .SpladeEasy import Document, KeyValue


class ShardWriter:
    """Writes documents to a shard file with inverted index support."""

    def __init__(self, path: str, initial_buffer_size: int = 32768, write_batch_size: int = 100):
        if write_batch_size <= 0:
            raise ValueError(f"write_batch_size must be positive, got {write_batch_size}")
        self.path = Path(path)
        self.f = open(path, "ab")  # noqa: SIM115 - file kept open for writer's lifetime, closed in close()
        self._size = 0
        self.initial_buffer_size = initial_buffer_size
        self.write_batch_size = write_batch_size
        self.write_buffer = []
        self._inverted_index_builder = None

    def enable_inverted_index(self):
        """Enable building inverted index while writing"""
        from .inverted_index import InvertedIndexBuilder

        self._inverted_index_builder = InvertedIndexBuilder()

    def append(
        self,
        doc_id: str,
        text: str | bytes,
        metadata: dict,
        token_ids: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        builder = flatbuffers.Builder(self.initial_buffer_size)

        # Vectors (zero-copy from numpy)
        token_ids_vec = builder.CreateNumpyVector(token_ids.astype(np.uint32, copy=False))
        weights_vec = builder.CreateNumpyVector(weights.astype(np.float32, copy=False))

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

        doc_id_off = builder.CreateString(doc_id)

        if isinstance(text, str):
            text = text.encode("utf-8")
        text_arr = (
            np.frombuffer(text, dtype=np.uint8) if text is not None else np.zeros(0, np.uint8)
        )
        text_vec = builder.CreateNumpyVector(text_arr)

        Document.DocumentStart(builder)
        Document.DocumentAddDocId(builder, doc_id_off)
        Document.DocumentAddText(builder, text_vec)
        Document.DocumentAddMetadata(builder, metadata_vec)
        Document.DocumentAddTokenIds(builder, token_ids_vec)
        Document.DocumentAddWeights(builder, weights_vec)
        doc_off = Document.DocumentEnd(builder)

        builder.Finish(doc_off)

        buf = bytes(builder.Output())

        # Record offset for inverted index
        doc_offset = self._size + 4

        self.write_buffer.append((len(buf), buf))
        self._size += 4 + len(buf)

        # Update inverted index
        if self._inverted_index_builder is not None:
            self._inverted_index_builder.add_document(doc_offset, token_ids, weights)

        if len(self.write_buffer) >= self.write_batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self.write_buffer:
            return
        data = b"".join(struct.pack("I", size) + buf for size, buf in self.write_buffer)
        self.f.write(data)
        self.write_buffer.clear()

    def size(self) -> int:
        return self._size

    def close(self) -> None:
        self._flush()
        self.f.close()

    def get_inverted_index(self):
        """Get the built inverted index (call after all documents are added)"""
        if self._inverted_index_builder is None:
            return None
        return self._inverted_index_builder.finalize()


class ShardReader:
    """Reads documents from a shard file."""

    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            self.data = None
        else:
            file_size = self.path.stat().st_size
            if file_size == 0:
                self.data = None
            else:
                self.data = np.memmap(path, dtype="uint8", mode="r")
        self.size = 0 if self.data is None else len(self.data)

    def scan(
        self, load_text: bool | str = True, *, want_positions: bool = False, light: bool = False
    ) -> Iterator[dict]:
        """
        load_text: False | "bytes" | True
        want_positions: include ('_pos': (offset, msg_len)) for each record
        light: if True, skip building token_ids/weights/metadata to minimize allocs
        """
        if self.data is None or self.size == 0:
            return
        offset = 0
        while offset < self.size:
            if offset + 4 > self.size:
                break
            msg_len = struct.unpack_from("I", self.data, offset)[0]
            rec_offset = offset + 4
            offset += 4
            if offset + msg_len > self.size:
                break

            doc = Document.Document.GetRootAs(self.data[rec_offset:], 0)

            # doc_id (tiny, ok to decode)
            did = doc.DocId().decode()

            # optionally skip heavy fields
            if light:
                token_ids = None
                weights = None
                metadata = None
            else:
                metadata = {}
                for i in range(doc.MetadataLength()):
                    kv = doc.Metadata(i)
                    metadata[kv.Key().decode()] = kv.Value().decode()

                token_ids = doc.TokenIdsAsNumpy()
                weights = doc.WeightsAsNumpy()

            text_value = None
            if load_text:
                text_np = doc.TextAsNumpy()  # zero-copy numpy view over memmap
                if isinstance(load_text, str) and load_text == "bytes":
                    text_value = memoryview(text_np) if text_np is not None else memoryview(b"")
                else:
                    text_value = (
                        (memoryview(text_np).tobytes().decode("utf-8"))
                        if text_np is not None
                        else ""
                    )

            rec = {
                "doc_id": did,
                "text": text_value,
                "metadata": metadata,
                "token_ids": token_ids,
                "weights": weights,
            }
            if want_positions:
                rec["_pos"] = (rec_offset, msg_len)

            yield rec
            offset = rec_offset + msg_len

    def read_text_at(self, rec_offset: int) -> memoryview | None:
        """Return a zero-copy memoryview over the text bytes for the record starting at rec_offset."""
        doc = Document.Document.GetRootAs(self.data[rec_offset:], 0)
        text_np = doc.TextAsNumpy()
        return memoryview(text_np) if text_np is not None else None

    def read_at_offset(self, rec_offset: int) -> dict | None:
        """Load full document at specific offset"""
        if self.data is None or rec_offset >= self.size:
            return None

        try:
            if rec_offset < 4:
                return None
            msg_len = struct.unpack_from("I", self.data, rec_offset - 4)[0]

            if rec_offset + msg_len > self.size:
                return None

            doc = Document.Document.GetRootAs(self.data[rec_offset:], 0)

            did = doc.DocId().decode()

            metadata = {}
            for i in range(doc.MetadataLength()):
                kv = doc.Metadata(i)
                metadata[kv.Key().decode()] = kv.Value().decode()

            token_ids = doc.TokenIdsAsNumpy()
            weights = doc.WeightsAsNumpy()

            return {
                "doc_id": did,
                "text": None,
                "metadata": metadata,
                "token_ids": token_ids,
                "weights": weights,
            }
        except Exception:
            return None

    def close(self) -> None:
        if self.data is not None:
            del self.data
