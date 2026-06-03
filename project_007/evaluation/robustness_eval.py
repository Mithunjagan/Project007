"""
PROJECT 007 — P4.5 Robustness Evaluation
Measures system performance against perturbations:
lighting, blur, shake, low resolution, partial occlusion.

Usage:
    python -m evaluation.robustness_eval [--dataset-dir dataset]
"""

import argparse
import json
import time
import random
from pathlib import Path

import cv2
import numpy as np

from pipeline.core import PipelineRunner
from training.feature_schema import ANNOTATION_TO_CLASS
from dataset_tools.annotation_template import load_annotation
from dataset_tools.video_indexer import VideoIndexer
from utils.logger import get_logger

logger = get_logger(__name__)


def _apply_perturbation(frame: np.ndarray, perturbation: str) -> np.ndarray:
    """Apply a visual perturbation to a frame."""
    if perturbation == "dark":
        return (frame * 0.3).astype(np.uint8)
    elif perturbation == "bright":
        return np.clip(frame * 1.8, 0, 255).astype(np.uint8)
    elif perturbation == "motion_blur":
        kernel = np.zeros((15, 15))
        kernel[7, :] = 1.0 / 15
        return cv2.filter2D(frame, -1, kernel)
    elif perturbation == "camera_shake":
        dx, dy = random.randint(-10, 10), random.randint(-10, 10)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    elif perturbation == "low_resolution":
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    elif perturbation == "occlusion":
        result = frame.copy()
        h, w = result.shape[:2]
        # Black rectangle in center-right area
        x1, y1 = int(w * 0.6), int(h * 0.2)
        x2, y2 = int(w * 0.95), int(h * 0.8)
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 0), -1)
        return result
    else:
        return frame


def _evaluate_with_perturbation(video_path, annotation, perturbation):
    """Run a single video with perturbation and return metrics."""
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

            # Apply perturbation
            if perturbation != "none":
                frame = _apply_perturbation(frame, perturbation)

            _, snap, fused = runner.step(frame, frame_count, now_mono, now_wall)

            gt_label = "normal"
            for ev in events:
                if ev.get("start_frame", 0) <= frame_count <= ev.get("end_frame", 0):
                    gt_label = ev.get("label", "normal")
                    break

            is_gt_anomaly = gt_label != "normal"
            det_score = max((fe.evidence_score for fe in fused), default=0.0)
            is_detected = det_score > 0.35

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


def run_robustness_eval(
    dataset_dir: str = "dataset",
    output_path: str = "evaluation/reports/robustness_eval.json",
) -> dict:
    """
    Evaluate performance under each perturbation type.
    """
    perturbations = ["none", "dark", "bright", "motion_blur",
                     "camera_shake", "low_resolution", "occlusion"]

    vi = VideoIndexer(dataset_dir)
    index = vi.index_all()
    annotated = [v for v in index if v["has_annotation"]]

    if not annotated:
        return {"error": "no_annotated_videos"}

    results = {}

    for perturb in perturbations:
        logger.info(f"Evaluating perturbation: {perturb}")
        total_tp = total_fp = total_tn = total_fn = 0

        for video_info in annotated:
            video_path = video_info["path"]
            video_id = Path(video_path).stem
            annotation = load_annotation(video_id, f"{dataset_dir}/annotations")
            if not annotation:
                continue

            r = _evaluate_with_perturbation(video_path, annotation, perturb)
            if r:
                total_tp += r["tp"]
                total_fp += r["fp"]
                total_tn += r["tn"]
                total_fn += r["fn"]

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[perturb] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": total_tp, "fp": total_fp,
            "tn": total_tn, "fn": total_fn,
        }
        logger.info(f"  {perturb}: P={precision:.4f} R={recall:.4f} F1={f1:.4f}")

    # Compute degradation from baseline
    baseline_f1 = results.get("none", {}).get("f1", 0)
    for perturb, r in results.items():
        if perturb != "none" and baseline_f1 > 0:
            r["f1_degradation_pct"] = round(
                100 * (baseline_f1 - r["f1"]) / baseline_f1, 2
            )

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "perturbations": perturbations,
        "results": results,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=4)
    logger.info(f"Robustness evaluation saved to {out}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P4.5 Robustness Evaluation")
    parser.add_argument("--dataset-dir", default="dataset")
    args = parser.parse_args()

    run_robustness_eval(args.dataset_dir)
