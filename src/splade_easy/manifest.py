import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class Manifest:
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
        return root / "manifest.json"

    @classmethod
    def load(cls, root: Path) -> "Manifest":
        manifest_path = cls.path_for(root)
        if not manifest_path.exists():
            # Create new manifest if it doesn't exist
            manifest = cls()
            manifest.save(root)
            return manifest

        with open(manifest_path) as f:
            data = json.load(f)
        return cls(**data)

    def save(self, root: Path) -> None:
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
        self.commit_id = str(uuid.uuid4())
        self.updated_at = datetime.now().isoformat()

    def shard_paths(self, root: Path) -> list[Path]:
        return [root / f"{h}.fb" for h in self.shard_hashes if (root / f"{h}.fb").exists()]
