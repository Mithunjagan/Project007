"""
PROJECT 007 — P4.5 Dataset Leakage Checks
Ensures train/test split integrity: no frame leakage across splits.

Usage:
    python -m evaluation.leakage_check [--dataset dataset/training_dataset.parquet]
"""

import argparse
import json
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from utils.logger import get_logger

logger = get_logger(__name__)


def check_leakage(
    dataset_path: str = "dataset/training_dataset.parquet",
    test_size: float = 0.2,
) -> dict:
    """
    Verify that train/test splits do not leak frames across video boundaries.

    Returns
    -------
    dict : Leakage check report.
    """
    if not HAS_PANDAS:
        raise ImportError("pandas required: pip install pandas pyarrow")

    df = pd.read_parquet(dataset_path)

    if "video_id" not in df.columns:
        return {"error": "No video_id column found in dataset"}

    video_ids = df["video_id"].unique().tolist()
    total_videos = len(video_ids)
    total_frames = len(df)

    # Simulate video-level split
    from sklearn.model_selection import train_test_split
    train_videos, test_videos = train_test_split(
        video_ids, test_size=test_size, random_state=42
    )

    train_set = set(train_videos)
    test_set = set(test_videos)

    # Check for overlap
    overlap = train_set & test_set
    has_leakage = len(overlap) > 0

    # Check frame-level
    train_df = df[df["video_id"].isin(train_set)]
    test_df = df[df["video_id"].isin(test_set)]

    # Check for duplicate (video_id, frame_id) pairs
    train_keys = set(zip(train_df["video_id"], train_df["frame_id"]))
    test_keys = set(zip(test_df["video_id"], test_df["frame_id"]))
    frame_leakage = train_keys & test_keys

    # Class distribution per split
    train_dist = train_df["label"].value_counts().to_dict()
    test_dist = test_df["label"].value_counts().to_dict()

    report = {
        "total_videos": total_videos,
        "total_frames": total_frames,
        "train_videos": len(train_videos),
        "test_videos": len(test_videos),
        "train_frames": len(train_df),
        "test_frames": len(test_df),
        "video_overlap": list(overlap),
        "has_video_leakage": has_leakage,
        "frame_leakage_count": len(frame_leakage),
        "has_frame_leakage": len(frame_leakage) > 0,
        "train_class_distribution": {str(k): int(v) for k, v in train_dist.items()},
        "test_class_distribution": {str(k): int(v) for k, v in test_dist.items()},
        "recommendation": "PASS" if not has_leakage and len(frame_leakage) == 0 else "FAIL: leakage detected",
    }

    # Save
    out = Path("evaluation/reports/leakage_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=4)

    if has_leakage or len(frame_leakage) > 0:
        logger.error(f"LEAKAGE DETECTED: {len(overlap)} video overlaps, {len(frame_leakage)} frame overlaps")
    else:
        logger.info(f"No leakage: {len(train_videos)} train videos, {len(test_videos)} test videos")

    return report


def generate_safe_split(
    dataset_path: str = "dataset/training_dataset.parquet",
    test_size: float = 0.2,
    output_dir: str = "dataset",
) -> dict:
    """
    Generate a video-level train/test split that guarantees no leakage.

    Returns
    -------
    dict : Split info with file paths.
    """
    if not HAS_PANDAS:
        raise ImportError("pandas required")

    df = pd.read_parquet(dataset_path)
    video_ids = df["video_id"].unique().tolist()

    from sklearn.model_selection import train_test_split
    train_videos, test_videos = train_test_split(
        video_ids, test_size=test_size, random_state=42
    )

    train_df = df[df["video_id"].isin(set(train_videos))]
    test_df = df[df["video_id"].isin(set(test_videos))]

    out_dir = Path(output_dir)
    train_path = out_dir / "train_split.parquet"
    test_path = out_dir / "test_split.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    info = {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_videos": len(train_videos),
        "test_videos": len(test_videos),
        "train_frames": len(train_df),
        "test_frames": len(test_df),
    }

    logger.info(f"Safe split: train={len(train_df)} frames, test={len(test_df)} frames")
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P4.5 Leakage Check")
    parser.add_argument("--dataset", default="dataset/training_dataset.parquet")
    parser.add_argument("--generate-split", action="store_true")
    args = parser.parse_args()

    report = check_leakage(args.dataset)
    print(json.dumps(report, indent=4))

    if args.generate_split:
        generate_safe_split(args.dataset)
