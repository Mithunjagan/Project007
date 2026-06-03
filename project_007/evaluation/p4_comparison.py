"""
PROJECT 007 — P4.0 Benchmark Comparison
Compares: (A) Rules Only, (B) ML Only, (C) Hybrid Fusion.

Usage:
    python -m evaluation.p4_comparison [--dataset-dir dataset] [--headless]
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
from training.feature_schema import ALL_FEATURES, ANNOTATION_TO_CLASS, CLASS_LABELS
from fusion.ml_fusion import MLFusionEngine
from dataset_tools.annotation_template import load_annotation
from dataset_tools.video_indexer import VideoIndexer
from utils.logger import get_logger

logger = get_logger(__name__)


def _load_best_model():
    """Load the best available ML model."""
    model_paths = [
        "models/saved/xgboost.pkl",
        "models/saved/random_forest.pkl",
    ]
    for path in model_paths:
        if Path(path).exists():
            return path
    return None


def _evaluate_video(
    video_path: str,
    annotation: dict,
    mode: str,
    model_path: str = None,
    det_weight: float = 0.70,
    ml_weight: float = 0.30,
) -> dict:
    """
    Evaluate a single video in one of three modes:
    - 'rules_only': Deterministic rules only
    - 'ml_only': ML prediction only
    - 'hybrid': Combined fusion

    Returns per-frame metrics.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open {video_path}"}

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

    ml_fusion = None
    if mode in ("ml_only", "hybrid") and model_path:
        ml_fusion = MLFusionEngine(
            model_path=model_path,
            deterministic_weight=det_weight,
            ml_weight=ml_weight,
        )

    events = annotation.get("events", [])
    tp = fp = tn = fn = 0
    frame_count = 0
    now_mono = time.perf_counter()
    now_wall = time.time()
    latencies = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            dt = 1.0 / fps
            now_mono += dt
            now_wall += dt

            t_start = time.perf_counter()
            _, snap, fused = runner.step(frame, frame_count, now_mono, now_wall)
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000)

            # Ground truth
            gt_label = 0  # normal
            for ev in events:
                sf = ev.get("start_frame", 0)
                ef = ev.get("end_frame", 0)
                if sf <= frame_count <= ef:
                    gt_label = ANNOTATION_TO_CLASS.get(ev.get("label", "normal"), 0)
                    break

            is_gt_anomaly = gt_label != 0

            # Get detection result based on mode
            if mode == "rules_only":
                # Use deterministic evidence score
                det_score = 0.0
                for fe in fused:
                    det_score = max(det_score, fe.evidence_score)
                is_detected = det_score > 0.35  # SUSPICIOUS threshold

            elif mode == "ml_only":
                if ml_fusion and ml_fusion.is_active():
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
                    ml_result = ml_fusion.fuse(features, 0.0, "NORMAL")
                    is_detected = ml_result["ml_score"] > 0.5
                else:
                    is_detected = False

            elif mode == "hybrid":
                det_score = 0.0
                det_state = "NORMAL"
                for fe in fused:
                    if fe.evidence_score > det_score:
                        det_score = fe.evidence_score
                        det_state = fe.state

                if ml_fusion and ml_fusion.is_active():
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
                    fusion_result = ml_fusion.fuse(features, det_score, det_state)
                    is_detected = fusion_result["fused_score"] > 0.35
                else:
                    is_detected = det_score > 0.35

            else:
                is_detected = False

            # Confusion matrix
            if is_gt_anomaly and is_detected:
                tp += 1
            elif not is_gt_anomaly and is_detected:
                fp += 1
            elif not is_gt_anomaly and not is_detected:
                tn += 1
            elif is_gt_anomaly and not is_detected:
                fn += 1

    finally:
        runner.cleanup()
        cap.release()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    duration_sec = frame_count / fps
    hours = duration_sec / 3600 if duration_sec > 0 else 1
    fp_per_hour = fp / hours

    avg_latency = sum(latencies) / max(1, len(latencies))

    return {
        "frames": frame_count,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fp_per_hour": round(fp_per_hour, 2),
        "avg_latency_ms": round(avg_latency, 2),
    }


def run_comparison(
    dataset_dir: str = "dataset",
    output_path: str = "evaluation/reports/p4_comparison.json",
) -> dict:
    """
    Run the full 3-way comparison.
    """
    vi = VideoIndexer(dataset_dir)
    index = vi.index_all()
    annotated = [v for v in index if v["has_annotation"]]

    if not annotated:
        logger.warning("No annotated videos. Cannot run comparison.")
        return {"error": "no_annotated_videos"}

    model_path = _load_best_model()
    if model_path:
        logger.info(f"Using ML model: {model_path}")
    else:
        logger.warning("No trained ML model found. ML and hybrid modes will be limited.")

    modes = ["rules_only", "ml_only", "hybrid"]
    comparison = {mode: {
        "total_tp": 0, "total_fp": 0, "total_tn": 0, "total_fn": 0,
        "total_frames": 0, "total_duration_sec": 0.0,
        "latencies_ms": [],
    } for mode in modes}

    for i, video_info in enumerate(annotated):
        video_path = video_info["path"]
        video_id = Path(video_path).stem
        duration = video_info.get("duration_sec", 0)

        annotation = load_annotation(video_id, f"{dataset_dir}/annotations")
        if not annotation:
            continue

        logger.info(f"[{i + 1}/{len(annotated)}] {video_id}")

        for mode in modes:
            logger.info(f"  Mode: {mode}")
            result = _evaluate_video(
                video_path, annotation, mode,
                model_path=model_path,
            )

            if "error" in result:
                continue

            c = comparison[mode]
            c["total_tp"] += result["tp"]
            c["total_fp"] += result["fp"]
            c["total_tn"] += result["tn"]
            c["total_fn"] += result["fn"]
            c["total_frames"] += result["frames"]
            c["total_duration_sec"] += duration
            c["latencies_ms"].append(result["avg_latency_ms"])

    # Compute aggregate metrics
    report = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "videos": len(annotated)}

    for mode in modes:
        c = comparison[mode]
        tp, fp, tn, fn = c["total_tp"], c["total_fp"], c["total_tn"], c["total_fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        hours = c["total_duration_sec"] / 3600 if c["total_duration_sec"] > 0 else 1
        fp_per_hour = fp / hours
        avg_lat = sum(c["latencies_ms"]) / max(1, len(c["latencies_ms"]))

        report[mode] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "false_positives_per_hour": round(fp_per_hour, 2),
            "detection_latency_ms": round(avg_lat, 2),
            "total_frames": c["total_frames"],
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        }

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=4)

    logger.info("=" * 60)
    logger.info("  P4.0 COMPARISON RESULTS")
    logger.info("=" * 60)
    for mode in modes:
        m = report[mode]
        logger.info(f"  {mode:15s}  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  FP/h={m['false_positives_per_hour']:.1f}  Lat={m['detection_latency_ms']:.1f}ms")
    logger.info("=" * 60)
    logger.info(f"Report saved to {out}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P4.0 Benchmark Comparison")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output", default="evaluation/reports/p4_comparison.json")
    args = parser.parse_args()

    run_comparison(args.dataset_dir, args.output)
