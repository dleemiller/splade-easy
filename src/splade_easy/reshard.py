#!/usr/bin/env python3
"""Reshard utility for SPLADE index with content-addressed shards."""

import argparse
import json
import logging
import os
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path

from splade_easy.shard import ShardReader, ShardWriter
from splade_easy.utils import hash_file

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class IndexResharder:
    """Context manager for safe resharding operations."""

    def __init__(self, index_dir: str, target_size_mb: int = 32, keep_originals: bool = False):
        self.index_dir = Path(index_dir)
        self.target_size_bytes = target_size_mb * 1024 * 1024
        self.keep_originals = keep_originals

        self.meta_path = self.index_dir / "metadata.json"
        self.deleted_path = self.index_dir / "deleted_ids.txt"
        self.temp_dir = self.index_dir / "_reshard_temp"

        self.new_shard_hashes = []
        self.success = False

    def __enter__(self):
        """Prepare for resharding."""
        # Ensure index metadata exists
        if not self.meta_path.exists():
            raise ValueError(f"Index not found: {self.index_dir}")

        with open(self.meta_path) as f:
            self.metadata = json.load(f)

        self.deleted_ids = set()
        if self.deleted_path.exists():
            with open(self.deleted_path) as f:
                self.deleted_ids = {line.strip() for line in f if line.strip()}

        # Find content-addressed shards (64-character hex filenames)
        all_fb_files = list(self.index_dir.glob("*.fb"))
        self.old_shards = sorted(
            [
                p
                for p in all_fb_files
                if len(p.stem) == 64 and all(c in "0123456789abcdef" for c in p.stem.lower())
            ]
        )

        if not self.old_shards:
            raise ValueError("No shards found")

        logger.info(f"Resharding {len(self.old_shards)} shards")
        logger.info(f"Target size: {self.target_size_bytes / (1024 * 1024):.0f} MB")

        # Ensure no stale temp shards from a previous interrupted run
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up and finalize."""
        if exc_type is not None:
            # Failed - clean up
            logger.error(f"Failed: {exc_val}")
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            for h in self.new_shard_hashes:
                (self.index_dir / f"{h}.fb").unlink(missing_ok=True)
            return False

        if not self.success:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            return True

        # Success - commit changes atomically
        try:
            temp_meta = self.index_dir / "_metadata.json.tmp"
            with open(temp_meta, "w") as f:
                json.dump(self.metadata, f, indent=2)
            temp_meta.replace(self.meta_path)

            self.deleted_path.unlink(missing_ok=True)

            # Remove old shards - BUT ONLY if they're not in the new shard list
            # This handles the case where resharding produces the same hash (no content change)
            new_shard_set = set(self.new_shard_hashes)

            if not self.keep_originals:
                for p in self.old_shards:
                    # Extract hash from filename (remove .fb extension)
                    old_hash = p.stem

                    # Only delete if this hash is NOT in the new shard list
                    if old_hash not in new_shard_set:
                        p.unlink()
            else:
                for p in self.old_shards:
                    old_hash = p.stem
                    # Only backup if not in new shard list
                    if old_hash not in new_shard_set:
                        p.rename(p.with_suffix(".fb.backup"))

            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info("✓ Complete")

        except Exception as e:
            logger.error(f"Finalization error: {e}")
            return False

        return True

    def reshard(self):
        """Perform resharding."""
        current_writer = None
        current_temp_path = None
        docs_written = 0

        for shard_path in self.old_shards:
            reader = ShardReader(str(shard_path))
            for doc in reader.scan(load_text=True):
                if doc["doc_id"] in self.deleted_ids:
                    continue

                if current_writer is None:
                    current_temp_path = self.temp_dir / f"temp_{len(self.new_shard_hashes)}.fb"
                    current_writer = ShardWriter(str(current_temp_path))

                current_writer.append(
                    doc["doc_id"],
                    doc["text"],
                    doc["metadata"],
                    doc["token_ids"],
                    doc["weights"],
                )
                docs_written += 1

                # Rotate shard if full
                if current_writer.size() >= self.target_size_bytes:
                    shard_hash = self._finalize_shard(current_writer, current_temp_path)
                    self.new_shard_hashes.append(shard_hash)
                    current_writer = None

            reader.close()

        if current_writer is not None:
            shard_hash = self._finalize_shard(current_writer, current_temp_path)
            self.new_shard_hashes.append(shard_hash)

        self.metadata["shard_hashes"] = self.new_shard_hashes
        self.metadata["num_shards"] = len(self.new_shard_hashes)
        self.metadata["num_docs"] = docs_written
        self.metadata["shard_size_mb"] = self.target_size_bytes // (1024 * 1024)

        # Bump commit_id for structural change
        self.metadata["commit_id"] = str(uuid.uuid4())
        self.metadata["updated_at"] = datetime.now().isoformat()

        self.success = True

        logger.info(
            f"{len(self.old_shards)} shards → {len(self.new_shard_hashes)} shards "
            f"({docs_written} docs)"
        )

        return {
            "old_shards": len(self.old_shards),
            "new_shards": len(self.new_shard_hashes),
            "docs_written": docs_written,
        }

    def _finalize_shard(self, writer, temp_path):
        """Close shard, hash it, and move to final location."""
        writer.close()
        shard_hash = hash_file(temp_path)
        final_path = self.index_dir / f"{shard_hash}.fb"
        # Same-filesystem atomic rename
        os.replace(temp_path, final_path)
        size_mb = final_path.stat().st_size / (1024 * 1024)
        logger.info(f"  {shard_hash[:16]}... ({size_mb:.1f} MB)")
        return shard_hash


def main():
    parser = argparse.ArgumentParser(description="Reshard SPLADE index")
    parser.add_argument("index_dir", help="Index directory")
    parser.add_argument("--shard-size", type=int, default=32, help="Target shard size in MB")
    parser.add_argument("--keep-originals", action="store_true", help="Keep original shards")
    args = parser.parse_args()

    try:
        with IndexResharder(args.index_dir, args.shard_size, args.keep_originals) as resharder:
            resharder.reshard()
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
