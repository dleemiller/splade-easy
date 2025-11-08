import hashlib
import logging
from pathlib import Path

import yaml
from datasets import load_dataset
from rich.logging import RichHandler
from rich.progress import track
from sentence_transformers import SentenceTransformer

from splade_easy.index import SpladeIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_time=False)],
)
logger = logging.getLogger(__name__)


class DatasetIngest:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.model = SentenceTransformer(
            self.config.get("model", "naver/splade-cocondenser-ensembledistil")
        )
        self.index_dir = self.config.get("index", {}).get("path", "./index")
        self.shard_size_mb = self.config.get("index", {}).get("shard_size_mb", 32)

    def _make_doc_id(self, row: dict, index: int) -> str:
        id_col = self.config.get("id_column")
        if id_col and id_col in row:
            return str(row[id_col])

        content = "".join(str(row.get(col, "")) for col in self.config["text_columns"])
        return f"doc_{index}_{hashlib.sha256(content.encode()).hexdigest()[:16]}"

    def _make_text(self, row: dict) -> str:
        parts = [str(row[col]) for col in self.config["text_columns"] if col in row and row[col]]
        return self.config.get("separator", " ").join(parts)

    def _make_metadata(self, row: dict) -> dict:
        meta_cols = self.config.get("metadata_columns", [])
        return {col: str(row[col]) for col in meta_cols if col in row and row[col] is not None}

    def ingest(self, batch_size: int = 100, max_docs: int = None, resume: bool = False):
        ds_config = self.config["dataset"]
        logger.info(f"Loading dataset: {ds_config['name']}")

        if ds_config["subset"] is not None:
            ds = load_dataset(
                ds_config["name"], ds_config["subset"], split=ds_config.get("split", "train")
            )
        else:
            ds = load_dataset(ds_config["name"], split=ds_config.get("split", "train"))
        if max_docs:
            ds = ds.select(range(min(max_docs, len(ds))))

        logger.info(f"Processing {len(ds)} documents")

        if resume and Path(self.index_dir).exists():
            logger.info("Resuming existing index")
            index = SpladeIndex(self.index_dir, shard_size_mb=self.shard_size_mb)
        else:
            logger.info("Creating new index")
            index = SpladeIndex(self.index_dir, shard_size_mb=self.shard_size_mb)

        doc_ids, texts, metadatas = [], [], []

        for i, row in track(enumerate(ds), total=len(ds), description="Ingesting"):
            doc_ids.append(self._make_doc_id(row, i))
            texts.append(self._make_text(row))
            metadatas.append(self._make_metadata(row))

            if len(doc_ids) >= batch_size:
                index.add_texts(doc_ids, texts, metadatas, self.model)
                doc_ids, texts, metadatas = [], [], []

        if doc_ids:
            index.add_texts(doc_ids, texts, metadatas, self.model)

        # Finalize any remaining shard
        index._finalize_current_shard()

        stats = index.stats()
        logger.info(
            f"Complete: {stats['num_docs']} docs, {stats['num_shards']} shards, {stats['total_size_mb']:.2f}MB"
        )

        return index


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-docs", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    DatasetIngest(args.config).ingest(args.batch_size, args.max_docs, args.resume)
