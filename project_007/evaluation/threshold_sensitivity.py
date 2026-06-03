"""
PROJECT 007 — P4.5 Threshold Sensitivity Study
Evaluates system performance across threshold variations.

Usage:
    python -m evaluation.threshold_sensitivity [--dataset-dir dataset]
"""

import argparse
import json
import time
from pathlib import Path
from copy import deepcopy

import cv2

from pipeline.core import PipelineRunner
from training.feature_schema import ANNOTATION_TO_CLASS
from dataset_tools.annotation_template import load_annotation
from dataset_tools.video_indexer import VideoIndexer
from utils.logger import get_logger

logger = get_logger(__name__)

# Thresholds to vary and their config names
THRESHOLD_PARAMS = [
    ("APPROACH_VELOCITY_THRESHOLD", "approach_velocity"),
    ("ARM_SWING_THRESHOLD", "arm_swing"),
    ("FALL_SCORE_THRESHOLD", "fall_score"),
    ("CONTACT_DISTANCE_THRESHOLD", "contact_distance"),
]

# Variation percentages
VARIATIONS = [-30, -20, -10, 0, 10, 20, 30]


def _evaluate_at_threshold(video_path, annotation, detection_threshold=0.35):
    """Run a single video and return TP/FP/TN/FN counts."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    runner = PipelineRunner(
        frame_width, frame_height,
        sync_mode=True, enable_recording=False, log_telemetry=False,
    )

    events = annotation.get("events", [])
    tp = fp = tn = fn = 0
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

            _, snap, fused = runner.step(frame, frame_count, now_mono, now_wall)

            gt_label = "normal"
            for ev in events:
                if ev.get("start_frame", 0) <= frame_count <= ev.get("end_frame", 0):
                    gt_label = ev.get("label", "normal")
                    break

            is_gt_anomaly = gt_label != "normal"

            det_score = max((fe.evidence_score for fe in fused), default=0.0)
            is_detected = det_score > detection_threshold

            if is_gt_anomaly and is_detected:
                tp += 1
            elif not is_gt_anomaly and is_detected:
                fp += 1
            elif not is_gt_anomaly and not is_detected:
                tn += 1
            else:
                fn += 1
    finally:
        runner.cleanup()
        cap.release()

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "frames": frame_count}


def run_sensitivity_study(
    dataset_dir: str = "dataset",
    output_path: str = "evaluation/reports/threshold_sensitivity.json",
) -> dict:
    """
    Evaluate performance across detection threshold variations.
    Tests thresholds at: baseline * (1 + variation%).
    """
    vi = VideoIndexer(dataset_dir)
    index = vi.index_all()
    annotated = [v for v in index if v["has_annotation"]]

    if not annotated:
        return {"error": "no_annotated_videos"}

    import config
    baseline_threshold = 0.35  # Detection threshold

    results = []

    for var_pct in VARIATIONS:
        threshold = baseline_threshold * (1 + var_pct / 100.0)
        logger.info(f"Testing threshold={threshold:.4f} (baseline {var_pct:+d}%)")

        total_tp = total_fp = total_tn = total_fn = 0

        for video_info in annotated:
            video_path = video_info["path"]
            video_id = Path(video_path).stem
            annotation = load_annotation(video_id, f"{dataset_dir}/annotations")
            if not annotation:
                continue

            r = _evaluate_at_threshold(video_path, annotation, threshold)
            if r:
                total_tp += r["tp"]
                total_fp += r["fp"]
                total_tn += r["tn"]
                total_fn += r["fn"]

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        entry = {
            "variation_pct": var_pct,
            "threshold": round(threshold, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": total_tp,
            "fp": total_fp,
            "tn": total_tn,
            "fn": total_fn,
        }
        results.append(entry)
        logger.info(f"  P={precision:.4f} R={recall:.4f} F1={f1:.4f}")

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "baseline_threshold": baseline_threshold,
        "variations_pct": VARIATIONS,
        "results": results,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=4)
    logger.info(f"Threshold sensitivity report saved to {out}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P4.5 Threshold Sensitivity")
    parser.add_argument("--dataset-dir", default="dataset")
    args = parser.parse_args()

    run_sensitivity_study(args.dataset_dir)
