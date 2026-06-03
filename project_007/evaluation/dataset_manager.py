"""
PROJECT 007 — P2.5 Dataset Manager
Manages the video dataset directory structure with versioned manifests.
"""

import os
import json
import shutil
import time
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)


class DatasetManager:
    """
    Manages the video dataset structure for the P2.5 Evaluation framework.
    Includes versioned metadata manifest.
    """
    def __init__(self, root_dir="dataset"):
        self.root_dir = Path(root_dir)
        self.categories = [
            "normal",
            "camera_tamper",
            "intrusion",
            "interaction",
            "crowded",
            "occlusion",
            "lighting_change",
            "false_positive_cases",
            "calibration",
        ]
        self.metadata_dir = self.root_dir / "metadata"
        self._init_structure()

    def _init_structure(self):
        """Ensure the directory structure exists."""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        for cat in self.categories:
            (self.root_dir / cat).mkdir(parents=True, exist_ok=True)

        # Initialize manifest if it doesn't exist
        manifest_path = self.metadata_dir / "dataset_manifest.json"
        if not manifest_path.exists():
            self._save_manifest()

    def _load_manifest(self) -> dict:
        manifest_path = self.metadata_dir / "dataset_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return self._default_manifest()

    def _default_manifest(self) -> dict:
        return {
            "dataset_version": "1.0.0",
            "annotation_version": "1.0.0",
            "creation_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "last_modified": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "sample_counts": {cat: 0 for cat in self.categories},
            "total_videos": 0,
        }

    def _save_manifest(self):
        manifest = self._load_manifest() if (self.metadata_dir / "dataset_manifest.json").exists() else self._default_manifest()

        # Recount samples
        total = 0
        for cat in self.categories:
            cat_dir = self.root_dir / cat
            if cat_dir.exists():
                video_count = len([
                    f for f in cat_dir.iterdir()
                    if f.is_file() and f.suffix in (".mp4", ".avi", ".mkv", ".mov")
                ])
                manifest["sample_counts"][cat] = video_count
                total += video_count

        manifest["total_videos"] = total
        manifest["last_modified"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        with open(self.metadata_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)

    def add_video(self, video_path: str, category: str, expected_behavior: str, notes: str = "") -> str:
        """Copy a video into the dataset and generate its metadata JSON."""
        if category not in self.categories:
            raise ValueError(f"Unknown category {category}. Must be one of {self.categories}")

        src_path = Path(video_path)
        if not src_path.exists():
            raise FileNotFoundError(f"Source video {src_path} not found.")

        target_dir = self.root_dir / category
        video_id = f"{category}_{src_path.stem}"
        dest_video_path = target_dir / f"{video_id}{src_path.suffix}"

        shutil.copy2(src_path, dest_video_path)

        metadata = {
            "video_id": video_id,
            "category": category,
            "expected_behavior": expected_behavior,
            "notes": notes,
            "original_filename": src_path.name,
            "added_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        meta_path = target_dir / f"{video_id}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        self._save_manifest()
        logger.info(f"Added {video_id} to dataset/{category}")
        return video_id

    def list_videos(self, category=None):
        """List all videos, optionally filtered by category."""
        results = []
        dirs = [self.root_dir / category] if category else [self.root_dir / c for c in self.categories]

        for d in dirs:
            if not d.exists():
                continue
            for meta_file in d.glob("*_meta.json"):
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                video_files = list(d.glob(f"{meta['video_id']}.*"))
                video_files = [v for v in video_files if v.suffix not in (".json",)]
                if video_files:
                    meta["video_path"] = str(video_files[0])
                    results.append(meta)

        return results

    def get_manifest(self) -> dict:
        """Return the current dataset manifest."""
        self._save_manifest()  # Refresh counts
        return self._load_manifest()
