"""
PROJECT 007 — Dataset Import Utility
Imports existing video files into the dataset structure.

Usage:
    python -m dataset_tools.import_video <video_path> --category <category>
"""

import argparse
import json
import shutil
import time
import uuid
from pathlib import Path

import cv2

from dataset_tools.export_manifest import export_manifest
from dataset_tools.session_manager import SCENARIO_CATEGORIES
from utils.logger import get_logger

logger = get_logger(__name__)


def import_video(video_path: str, category: str, dataset_dir: str = "dataset") -> bool:
    """
    Import a video file into the dataset under the specified category.
    """
    video_file = Path("C:\Users\mith1\Downloads\fight_test.mp4")
    if not video_file.exists():
        logger.error(f"File not found: {video_path}")
        return False

    if category not in SCENARIO_CATEGORIES:
        logger.error(f"Invalid category: {category}. Must be one of {SCENARIO_CATEGORIES}")
        return False

    # Verify video can be opened
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        logger.error(f"Cannot open video file with OpenCV: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if total_frames <= 0:
        logger.error(f"Video has no frames or cannot be read properly: {video_path}")
        return False

    duration_sec = total_frames / fps
    resolution = f"{width}x{height}"

    # Generate session ID
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    session_id = f"session_{timestamp}_{short_uuid}"

    # Create directories
    target_dir = Path(dataset_dir) / category
    target_dir.mkdir(parents=True, exist_ok=True)

    # Copy video
    dest_video = target_dir / f"{session_id}.mp4"
    try:
        shutil.copy2(video_file, dest_video)
    except Exception as e:
        logger.error(f"Failed to copy video: {e}")
        return False

    # Generate metadata
    meta = {
        "session_id": session_id,
        "original_filename": video_file.name,
        "category": category,
        "import_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fps": fps,
        "resolution": resolution,
        "duration_seconds": round(duration_sec, 2),
        "total_frames": total_frames
    }

    meta_path = target_dir / f"{session_id}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    # Update manifest
    export_manifest(dataset_dir)

    # Print summary
    print("\nImported: SUCCESS")
    print(f"Category: {category}")
    print(f"Frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Duration: {duration_sec:.2f} sec")
    print(f"Saved To: {dest_video.as_posix()}\n")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import video into PROJECT 007 dataset.")
    parser.add_argument("video_path", help="Path to the video file to import.")
    parser.add_argument("--category", required=True, choices=SCENARIO_CATEGORIES, help="Scenario category for the video.")
    parser.add_argument("--dataset-dir", default="dataset", help="Dataset root directory.")
    
    args = parser.parse_args()
    
    import_video(args.video_path, args.category, args.dataset_dir)
