"""
PROJECT 007 — P4.0 Dataset Builder
Converts annotated video sessions into supervised training samples.

Usage:
    python -m training.dataset_builder [--dataset-dir dataset]
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
from training.feature_schema import (
    ALL_FEATURES,
    ANNOTATION_TO_CLASS,
    CLASS_LABELS,
)
from dataset_tools.annotation_template import load_annotation
from dataset_tools.video_indexer import VideoIndexer
from utils.logger import get_logger

logger = get_logger(__name__)


class DatasetBuilder:
    """
    Runs annotated videos through the pipeline in sync mode to extract
    labeled feature vectors for ML training.
    """

    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.vi = VideoIndexer(dataset_dir)

    def build(self, output_path: str = "dataset/training_dataset.parquet") -> int:
        """
        Build the training dataset from all annotated videos.

        Returns
        -------
        int : Number of samples generated.
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required: pip install pandas pyarrow")

        index = self.vi.index_all()
        annotated = [v for v in index if v["has_annotation"]]

        if not annotated:
            logger.warning("No annotated videos found. Cannot build training dataset.")
            return 0

        logger.info(f"Building training dataset from {len(annotated)} annotated videos")

        all_samples = []

        for i, video_info in enumerate(annotated):
            video_path = video_info["path"]
            video_id = Path(video_path).stem
            logger.info(f"[{i + 1}/{len(annotated)}] Processing {video_id}")

            annotation = load_annotation(video_id, str(self.dataset_dir / "annotations"))
            if not annotation:
                logger.warning(f"  No annotation found for {video_id}")
                continue

            samples = self._process_video(video_path, annotation)
            all_samples.extend(samples)
            logger.info(f"  Extracted {len(samples)} samples")

        if not all_samples:
            logger.warning("No samples generated.")
            return 0

        # Build DataFrame
        df = pd.DataFrame(all_samples)

        # Save
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)

        logger.info(f"Training dataset saved: {out} ({len(df)} samples)")
        logger.info(f"  Class distribution:")
        for cls_id, count in df["label"].value_counts().sort_index().items():
            cls_name = CLASS_LABELS.get(int(cls_id), "unknown")
            logger.info(f"    {cls_id} ({cls_name}): {count}")

        return len(df)

    def _process_video(self, video_path: str, annotation: dict) -> list:
        """Run a single video through the pipeline and extract labeled features."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open {video_path}")
            return []

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        runner = PipelineRunner(
            frame_width, frame_height,
            sync_mode=True,
            enable_recording=False,
            log_telemetry=False,
        )
        extractor = FeatureExtractor()

        events = annotation.get("events", [])
        samples = []
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

                annotated_frame, snap, fused = runner.step(
                    frame, frame_count, now_mono, now_wall
                )

                # Determine label for this frame
                label = 0  # normal
                for ev in events:
                    sf = ev.get("start_frame", 0)
                    ef = ev.get("end_frame", 0)
                    if sf <= frame_count <= ef:
                        anno_label = ev.get("label", "normal")
                        label = ANNOTATION_TO_CLASS.get(anno_label, 0)
                        break

                # Extract motion from snap
                all_motion = {}
                pairwise_motion = {}
                flow_metrics = {
                    "avg_flow_mag": snap.get("global_flow_magnitude", 0.0),
                    "instability_score": 0.0,
                }

                # Reconstruct fused evidence
                raw_rules = []  # Cannot reconstruct raw rules from snap

                features = extractor.extract(
                    all_motion=all_motion,
                    pairwise_motion=pairwise_motion,
                    flow_metrics=flow_metrics,
                    scene_stability=snap.get("scene_stability", 1.0),
                    occupancy_ratio=snap.get("occupancy_ratio", 0.0),
                    fused_evidence=fused,
                    raw_rules=raw_rules,
                    frame=frame,
                )

                # Add metadata
                features["label"] = label
                features["frame_id"] = frame_count
                features["video_id"] = Path(video_path).stem

                samples.append(features)

        finally:
            runner.cleanup()
            cap.release()

        return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P4.0 Dataset Builder")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output", default="dataset/training_dataset.parquet")
    args = parser.parse_args()

    builder = DatasetBuilder(args.dataset_dir)
    count = builder.build(args.output)
    print(f"Generated {count} training samples.")
