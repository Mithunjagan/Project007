"""
PROJECT 007 — P4.0 Feature Export
Extracts features from ALL videos (annotated or not) and exports to parquet.

Usage:
    python -m training.export_features [--dataset-dir dataset]
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from pipeline.core import PipelineRunner
from training.feature_extractor import FeatureExtractor
from training.feature_schema import ALL_FEATURES
from dataset_tools.video_indexer import VideoIndexer
from utils.logger import get_logger

logger = get_logger(__name__)


def export_features(
    dataset_dir: str = "dataset",
    output_path: str = "dataset/features.parquet",
) -> int:
    """
    Extract features from all videos and export to parquet.

    Returns
    -------
    int : Number of feature rows exported.
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required: pip install pandas pyarrow")

    vi = VideoIndexer(dataset_dir)
    index = vi.index_all()

    if not index:
        logger.warning("No videos found.")
        return 0

    logger.info(f"Exporting features from {len(index)} videos")

    all_rows = []

    for i, video_info in enumerate(index):
        video_path = video_info["path"]
        video_id = Path(video_path).stem
        category = video_info["category"]
        logger.info(f"[{i + 1}/{len(index)}] {video_id} ({category})")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"  Cannot open {video_path}")
            continue

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        runner = PipelineRunner(
            frame_width, frame_height,
            sync_mode=True,
            enable_recording=False,
            log_telemetry=False,
        )
        extractor = FeatureExtractor()

        frame_count = 0
        now_mono = time.perf_counter()
        now_wall = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                dt = 1.0 / fps
                now_mono += dt
                now_wall += dt

                _, snap, fused = runner.step(
                    frame, frame_count, now_mono, now_wall
                )

                flow_metrics = {
                    "avg_flow_mag": snap.get("global_flow_magnitude", 0.0),
                    "instability_score": 0.0,
                }

                features = extractor.extract(
                    all_motion={},
                    pairwise_motion={},
                    flow_metrics=flow_metrics,
                    scene_stability=snap.get("scene_stability", 1.0),
                    occupancy_ratio=snap.get("occupancy_ratio", 0.0),
                    fused_evidence=fused,
                    raw_rules=[],
                    frame=frame,
                )

                features["frame_id"] = frame_count
                features["video_id"] = video_id
                features["category"] = category

                all_rows.append(features)

        finally:
            runner.cleanup()
            cap.release()

        logger.info(f"  {frame_count} frames processed")

    if not all_rows:
        logger.warning("No features extracted.")
        return 0

    df = pd.DataFrame(all_rows)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    logger.info(f"Features exported: {out} ({len(df)} rows)")
    return len(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P4.0 Feature Export")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output", default="dataset/features.parquet")
    args = parser.parse_args()

    count = export_features(args.dataset_dir, args.output)
    print(f"Exported {count} feature rows.")
