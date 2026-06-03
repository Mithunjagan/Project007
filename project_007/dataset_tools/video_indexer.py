"""
PROJECT 007 — P3.0 Video Indexer
Indexes all video files in the dataset with metadata extraction.
"""

import json
from pathlib import Path

import cv2

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


class VideoIndexer:
    """Indexes all videos in the dataset and extracts technical metadata."""

    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = Path(dataset_dir)

    def index_all(self) -> list:
        """Scan all category directories and index video files."""
        index = []
        for cat in SCENARIO_CATEGORIES:
            cat_dir = self.dataset_dir / cat
            if not cat_dir.exists():
                continue
            for video_file in sorted(cat_dir.glob("*.mp4")):
                entry = self._probe_video(video_file, cat)
                if entry:
                    index.append(entry)
        return index

    def _probe_video(self, video_path: Path, category: str) -> dict:
        """Extract technical metadata from a video file."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning(f"Cannot open {video_path}")
                return None

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()

            # Check for matching metadata
            meta_path = video_path.parent / f"{video_path.stem}_meta.json"
            has_metadata = meta_path.exists()

            # Check for annotation
            anno_dir = self.dataset_dir / "annotations"
            anno_path = anno_dir / f"{video_path.stem}.json"
            has_annotation = anno_path.exists()

            return {
                "filename": video_path.name,
                "path": str(video_path),
                "category": category,
                "frame_count": frame_count,
                "fps": round(fps, 2),
                "resolution": f"{width}x{height}",
                "duration_sec": round(duration, 2),
                "size_mb": round(video_path.stat().st_size / (1024 * 1024), 2),
                "has_metadata": has_metadata,
                "has_annotation": has_annotation,
            }
        except Exception as e:
            logger.warning(f"Error probing {video_path}: {e}")
            return None

    def save_index(self, output_path: str = "dataset/metadata/video_index.json"):
        """Save the full index to JSON."""
        index = self.index_all()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=4)
        logger.info(f"Video index saved: {len(index)} videos → {out}")
        return index
