#!/usr/bin/env python3
"""Reshard utility for SPLADE index with content-addressed shards."""

import argparse
import json
import logging
import shutil
import sys
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
        if not self.meta_path.exists():
            raise ValueError(f"Index not found: {self.index_dir}")

        with open(self.meta_path) as f:
            self.metadata = json.load(f)

        # Load deleted IDs
        self.deleted_ids = set()
        if self.deleted_path.exists():
            with open(self.deleted_path) as f:
                self.deleted_ids = set(line.strip() for line in f if line.strip())

        # Find old shards (legacy or content-addressed)
        self.old_shards = sorted(self.index_dir.glob("shard_*.fb")) or sorted(
            self.index_dir.glob("[0-9a-f]" * 64 + ".fb")
        )

        if not self.old_shards:
            raise ValueError("No shards found")

        logger.info(f"Resharding {len(self.old_shards)} shards")
        logger.info(f"Target size: {self.target_size_bytes / (1024*1024):.0f} MB")

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
            # Update metadata
            temp_meta = self.index_dir / "_metadata.json.tmp"
            with open(temp_meta, "w") as f:
                json.dump(self.metadata, f, indent=2)
            temp_meta.replace(self.meta_path)

            # Clear deleted IDs
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

        # Read all non-deleted docs
        for shard_path in self.old_shards:
            reader = ShardReader(str(shard_path))
            for doc in reader.scan(load_text=True):
                if doc["doc_id"] in self.deleted_ids:
                    continue

                # Create new writer if needed
                if current_writer is None:
                    current_temp_path = self.temp_dir / f"temp_{len(self.new_shard_hashes)}.fb"
                    current_writer = ShardWriter(str(current_temp_path))

                # Write document
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

        # Finalize last shard
        if current_writer is not None:
            shard_hash = self._finalize_shard(current_writer, current_temp_path)
            self.new_shard_hashes.append(shard_hash)

        # Update metadata
        self.metadata["shard_hashes"] = self.new_shard_hashes
        self.metadata["num_shards"] = len(self.new_shard_hashes)
        self.metadata["num_docs"] = docs_written

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
        shutil.move(str(temp_path), str(final_path))
        size_mb = final_path.stat().st_size / (1024 * 1024)
        logger.info(f"  {shard_hash[:16]}... ({size_mb:.1f} MB)")
        return shard_hash


# def verify_integrity(index_dir: str) -> bool:
#     """Verify all shards match their hashes."""
#     index_dir = Path(index_dir)
#     meta_path = index_dir / "metadata.json"
#
#     with open(meta_path) as f:
#         metadata = json.load(f)
#
#     shard_hashes = metadata.get("shard_hashes", [])
#     if not shard_hashes:
#         logger.warning("No shard hashes in metadata (legacy index)")
#         return True
#
#     logger.info(f"Verifying {len(shard_hashes)} shards...")
#     all_valid = True
#
#     for i, h in enumerate(shard_hashes, 1):
#         path = index_dir / f"{h}.fb"
#         if not path.exists():
#             logger.error(f"[{i}] Missing: {h}")
#             all_valid = False
#             continue
#
#         actual = hash_file(path)
#         if actual != h:
#             logger.error(f"[{i}] Hash mismatch: {h[:16]}...")
#             all_valid = False
#         else:
#             size_mb = path.stat().st_size / (1024 * 1024)
#             logger.info(f"[{i}] ✓ {h[:16]}... ({size_mb:.1f} MB)")
#
#     return all_valid


def main():
    parser = argparse.ArgumentParser(description="Reshard SPLADE index")
    parser.add_argument("index_dir", help="Index directory")
    parser.add_argument("--shard-size", type=int, default=32, help="Target shard size in MB")
    parser.add_argument("--keep-originals", action="store_true", help="Keep original shards")
    # parser.add_argument("--verify", action="store_true", help="Verify integrity only")
    args = parser.parse_args()

    # if args.verify:
    #    valid = verify_integrity(args.index_dir)
    #    sys.exit(0 if valid else 1)

    try:
        with IndexResharder(args.index_dir, args.shard_size, args.keep_originals) as resharder:
            resharder.reshard()
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
