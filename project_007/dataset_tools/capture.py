"""
PROJECT 007 — P3.0 Dataset Capture
Record raw MP4 sessions from the webcam with metadata.

Usage:
    python -m dataset_tools.capture --category normal --operator user1
    python -m dataset_tools.capture --category intrusion --duration 60
"""

import argparse
import json
import time
import uuid
from pathlib import Path

import cv2

from config import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FALLBACK_WIDTH,
    CAMERA_FALLBACK_HEIGHT,
    TARGET_FPS,
)
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


def generate_session_id() -> str:
    """Generate a unique session ID: timestamp + short UUID."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    return f"session_{ts}_{short_id}"


def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam at index {CAMERA_INDEX}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if actual_w != CAMERA_WIDTH or actual_h != CAMERA_HEIGHT:
        logger.warning(
            f"Requested {CAMERA_WIDTH}x{CAMERA_HEIGHT}, got {actual_w}x{actual_h}. "
            f"Falling back to {CAMERA_FALLBACK_WIDTH}x{CAMERA_FALLBACK_HEIGHT}."
        )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FALLBACK_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FALLBACK_HEIGHT)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"Camera opened: {actual_w}x{actual_h}")
    return cap


def record_session(
    category: str,
    operator: str = "default",
    duration: int = 0,
    dataset_dir: str = "dataset",
) -> dict:
    """
    Record a raw MP4 session.

    Parameters
    ----------
    category : str
        Scenario category (must be in SCENARIO_CATEGORIES).
    operator : str
        Name of the person recording.
    duration : int
        Max recording duration in seconds. 0 = unlimited (press Q to stop).
    dataset_dir : str
        Root dataset directory.

    Returns
    -------
    dict : Session metadata.
    """
    if category not in SCENARIO_CATEGORIES:
        raise ValueError(f"Unknown category '{category}'. Must be one of {SCENARIO_CATEGORIES}")

    session_id = generate_session_id()
    output_dir = Path(dataset_dir) / category
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = open_camera()
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_path = output_dir / f"{session_id}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, TARGET_FPS, (frame_width, frame_height))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter for {video_path}")

    start_time = time.time()
    start_wall = time.strftime("%Y-%m-%dT%H:%M:%S")
    frame_count = 0

    logger.info(f"Recording session {session_id} [{category}] — press Q to stop")
    if duration > 0:
        logger.info(f"  Max duration: {duration}s")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame read failed")
                time.sleep(0.01)
                continue

            writer.write(frame)
            frame_count += 1

            # Show preview
            elapsed = time.time() - start_time
            preview = frame.copy()
            cv2.putText(
                preview,
                f"REC [{category}] {elapsed:.1f}s  F:{frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            cv2.imshow("PROJECT 007 - CAPTURE", preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                logger.info("Recording stopped by user.")
                break

            if duration > 0 and elapsed >= duration:
                logger.info(f"Duration limit ({duration}s) reached.")
                break

    finally:
        writer.release()
        cap.release()
        cv2.destroyAllWindows()

    end_time = time.time()
    actual_duration = round(end_time - start_time, 2)

    # Save metadata
    metadata = {
        "session_id": session_id,
        "start_time": start_wall,
        "duration": actual_duration,
        "resolution": f"{frame_width}x{frame_height}",
        "fps": TARGET_FPS,
        "operator": operator,
        "scenario_type": category,
        "total_frames": frame_count,
        "video_file": str(video_path),
        "original_filename": f"{session_id}.mp4",
    }

    meta_path = output_dir / f"{session_id}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Session saved: {video_path} ({frame_count} frames, {actual_duration}s)")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PROJECT 007 — Dataset Capture")
    parser.add_argument(
        "--category",
        default="normal",
        choices=SCENARIO_CATEGORIES,
        help="Scenario category",
    )
    parser.add_argument("--operator", default="default", help="Operator name")
    parser.add_argument("--duration", type=int, default=0, help="Max duration (0=unlimited)")
    parser.add_argument("--dataset-dir", default="dataset", help="Dataset root directory")
    args = parser.parse_args()

    record_session(
        category=args.category,
        operator=args.operator,
        duration=args.duration,
        dataset_dir=args.dataset_dir,
    )
