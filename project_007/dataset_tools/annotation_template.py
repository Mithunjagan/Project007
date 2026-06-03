"""
PROJECT 007 — P3.0 Annotation Template
Defines the ground-truth annotation schema and provides utilities.
"""

import json
from pathlib import Path
from typing import List

from utils.logger import get_logger

logger = get_logger(__name__)

# Supported event labels
EVENT_LABELS = [
    "normal",
    "camera_rush",
    "proximity_intrusion",
    "camera_shake",
    "lens_occlusion",
    "high_energy_interaction",
]


def create_empty_annotation(video_id: str, total_frames: int = 0, fps: float = 0.0) -> dict:
    """Create an empty annotation template for a video."""
    return {
        "video_id": video_id,
        "total_frames": total_frames,
        "fps": fps,
        "annotator": "",
        "annotation_version": "1.0",
        "events": [],
    }


def add_event(annotation: dict, start_time: float, end_time: float, label: str) -> dict:
    """Add a labeled event to an annotation."""
    if label not in EVENT_LABELS:
        raise ValueError(f"Unknown label '{label}'. Must be one of {EVENT_LABELS}")
    if end_time <= start_time:
        raise ValueError(f"end_time ({end_time}) must be > start_time ({start_time})")

    fps = annotation.get("fps", 30.0) or 30.0
    annotation["events"].append({
        "start_time": round(start_time, 3),
        "end_time": round(end_time, 3),
        "start_frame": int(start_time * fps),
        "end_frame": int(end_time * fps),
        "label": label,
    })
    return annotation


def save_annotation(annotation: dict, output_dir: str = "dataset/annotations") -> str:
    """Save an annotation to disk."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{annotation['video_id']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=4)
    logger.info(f"Annotation saved: {path}")
    return str(path)


def load_annotation(video_id: str, annotations_dir: str = "dataset/annotations") -> dict:
    """Load an existing annotation."""
    path = Path(annotations_dir) / f"{video_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
