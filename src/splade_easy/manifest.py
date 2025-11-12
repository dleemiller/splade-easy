# manifest.py
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class Manifest:
    """
    Index manifest containing metadata and shard references.

    The manifest tracks the index structure, document count, and shard locations.
    It uses a commit_id to detect structural changes (resharding, compaction, etc.)
    for cache invalidation.

    Attributes:
        version: Manifest format version
        model_id: Identifier of the SPLADE model used for encoding
        shard_hashes: Content-addressed hashes of shard files
        num_docs: Total number of documents in the index
        shard_size_mb: Target size for new shards
        commit_id: UUID that changes on structural modifications
        created_at: ISO timestamp of index creation
        updated_at: ISO timestamp of last modification
    """

    version: str = "0.1.0"
    model_id: Optional[str] = None
    shard_hashes: list[str] = field(default_factory=list)
    num_docs: int = 0
    shard_size_mb: float = 32.0
    commit_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def path_for(cls, root: Path) -> Path:
        """
        Get the manifest file path for an index directory.

        Args:
            root: Index root directory

        Returns:
            Path to manifest.json
        """
        return root / "manifest.json"

    @classmethod
    def load(cls, root: Path) -> "Manifest":
        """
        Load manifest from index directory, creating new one if missing.

        Args:
            root: Index root directory

        Returns:
            Loaded or newly created manifest
        """
        manifest_path = cls.path_for(root)
        if not manifest_path.exists():
            manifest = cls()
            manifest.save(root)
            return manifest

        with open(manifest_path) as f:
            data = json.load(f)
        return cls(**data)

    def save(self, root: Path) -> None:
        """
        Save manifest atomically to disk.

        Uses a temporary file and atomic rename to ensure consistency.

        Args:
            root: Index root directory
        """
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.path_for(root)

        # Atomic write with temp file
        temp_path = manifest_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(asdict(self), f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # Atomic replace
        os.replace(temp_path, manifest_path)

    def bump_commit(self) -> None:
        """
        Increment commit ID to indicate structural changes.

        Call this when the index structure changes (new shard, deletion, resharding)
        to invalidate dependent caches.
        """
        self.commit_id = str(uuid.uuid4())
        self.updated_at = datetime.now().isoformat()

    def shard_paths(self, root: Path) -> list[Path]:
        """
        Get paths to all existing shard files.

        Args:
            root: Index root directory

        Returns:
            List of paths to .fb shard files that exist on disk
        """
        return [root / f"{h}.fb" for h in self.shard_hashes if (root / f"{h}.fb").exists()]
