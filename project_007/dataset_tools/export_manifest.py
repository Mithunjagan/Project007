"""
PROJECT 007 — P3.0 Export Manifest
Generates and updates the dataset_manifest.json automatically.
"""

import json
import time
from pathlib import Path

from dataset_tools.session_manager import SessionManager
from dataset_tools.video_indexer import VideoIndexer
from utils.logger import get_logger

logger = get_logger(__name__)

SCENARIO_CATEGORIES = [
    "normal",
    "camera_tamper",
    "intrusion",
    "interaction",
    "crowded",
    "occlusion",
    "lighting_change",
]


def export_manifest(dataset_dir: str = "dataset") -> dict:
    """
    Generate a comprehensive dataset_manifest.json.
    """
    sm = SessionManager(dataset_dir)
    vi = VideoIndexer(dataset_dir)

    index = vi.index_all()
    category_counts = sm.get_category_counts()
    total_duration = sm.get_total_duration()

    # Count annotations
    anno_dir = Path(dataset_dir) / "annotations"
    annotation_count = 0
    if anno_dir.exists():
        annotation_count = len(list(anno_dir.glob("*.json")))

    manifest = {
        "dataset_version": "3.0.0",
        "annotation_version": "1.0.0",
        "creation_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "last_modified": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_videos": len(index),
        "total_duration_sec": round(total_duration, 2),
        "total_duration_hours": round(total_duration / 3600, 4),
        "total_annotations": annotation_count,
        "sample_counts": category_counts,
        "videos": [
            {
                "filename": v["filename"],
                "category": v["category"],
                "duration_sec": v["duration_sec"],
                "has_annotation": v["has_annotation"],
            }
            for v in index
        ],
    }

    out_dir = Path(dataset_dir) / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    logger.info(
        f"Manifest exported: {len(index)} videos, "
        f"{total_duration:.1f}s total, {annotation_count} annotations"
    )
    return manifest


if __name__ == "__main__":
    export_manifest()
