"""
PROJECT 007 — P4.5 Error Analysis Engine
Automatically collects FP, FN, TP, TN with full context for each.

Usage:
    python -m evaluation.error_analysis [--dataset-dir dataset]
"""

import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import cv2
import numpy as np

from pipeline.core import PipelineRunner
from training.feature_extractor import FeatureExtractor
from training.feature_schema import ANNOTATION_TO_CLASS, CLASS_LABELS
from fusion.ml_fusion import MLFusionEngine
from dataset_tools.annotation_template import load_annotation
from dataset_tools.video_indexer import VideoIndexer
from utils.logger import get_logger

logger = get_logger(__name__)

DETECTION_THRESHOLD = 0.35


@dataclass
class ErrorEntry:
    """A single frame-level error record."""
    video_id: str
    frame_id: int
    timestamp_sec: float
    error_type: str  # TP, FP, TN, FN
    ground_truth: str
    prediction: str
    deterministic_score: float
    ml_score: float
    fused_score: float
    contributing_rules: list = field(default_factory=list)
    top_features: list = field(default_factory=list)
    state: str = "NORMAL"


class ErrorAnalysisEngine:
    """Runs annotated videos through the pipeline and classifies each frame."""

    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.vi = VideoIndexer(dataset_dir)
        self.errors: List[ErrorEntry] = []

    def analyze(self, model_path: Optional[str] = None) -> dict:
        """Run full error analysis on all annotated videos."""
        index = self.vi.index_all()
        annotated = [v for v in index if v["has_annotation"]]

        if not annotated:
            logger.warning("No annotated videos found.")
            return {"error": "no_annotated_videos"}

        logger.info(f"Error analysis: {len(annotated)} annotated videos")
        self.errors = []

        for i, video_info in enumerate(annotated):
            video_path = video_info["path"]
            video_id = Path(video_path).stem
            logger.info(f"[{i + 1}/{len(annotated)}] {video_id}")

            annotation = load_annotation(video_id, str(self.dataset_dir / "annotations"))
            if not annotation:
                continue

            self._analyze_video(video_path, annotation, model_path)

        return self._build_report()

    def _analyze_video(self, video_path: str, annotation: dict, model_path: Optional[str] = None):
        """Analyze a single video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        video_id = Path(video_path).stem

        runner = PipelineRunner(
            frame_width, frame_height,
            sync_mode=True, enable_recording=False, log_telemetry=False,
        )
        extractor = FeatureExtractor()
        ml_fusion = MLFusionEngine(model_path=model_path) if model_path else None

        events = annotation.get("events", [])
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

                # Ground truth
                gt_label = "normal"
                for ev in events:
                    if ev.get("start_frame", 0) <= frame_count <= ev.get("end_frame", 0):
                        gt_label = ev.get("label", "normal")
                        break

                is_gt_anomaly = gt_label != "normal"

                # Deterministic score
                det_score = 0.0
                det_state = "NORMAL"
                contributing_rules = []
                for fe in fused:
                    if fe.evidence_score > det_score:
                        det_score = fe.evidence_score
                        det_state = fe.state
                        contributing_rules = fe.contributing_rules

                # ML score
                ml_score = 0.0
                fused_score = det_score
                top_features = []

                if ml_fusion and ml_fusion.is_active():
                    flow_metrics = {
                        "avg_flow_mag": snap.get("global_flow_magnitude", 0.0),
                        "instability_score": 0.0,
                    }
                    features = extractor.extract(
                        all_motion={}, pairwise_motion={},
                        flow_metrics=flow_metrics,
                        scene_stability=snap.get("scene_stability", 1.0),
                        occupancy_ratio=snap.get("occupancy_ratio", 0.0),
                        fused_evidence=fused, raw_rules=[], frame=frame,
                    )
                    fusion_result = ml_fusion.fuse(features, det_score, det_state)
                    ml_score = fusion_result["ml_score"]
                    fused_score = fusion_result["fused_score"]
                    top_features = fusion_result.get("top_features", [])

                is_detected = fused_score > DETECTION_THRESHOLD

                # Classify
                if is_gt_anomaly and is_detected:
                    error_type = "TP"
                elif not is_gt_anomaly and is_detected:
                    error_type = "FP"
                elif not is_gt_anomaly and not is_detected:
                    error_type = "TN"
                else:
                    error_type = "FN"

                # Only store errors and a sample of correct predictions
                if error_type in ("FP", "FN") or (frame_count % 30 == 0):
                    self.errors.append(ErrorEntry(
                        video_id=video_id,
                        frame_id=frame_count,
                        timestamp_sec=round(frame_count / fps, 2),
                        error_type=error_type,
                        ground_truth=gt_label,
                        prediction="anomaly" if is_detected else "normal",
                        deterministic_score=round(det_score, 4),
                        ml_score=round(ml_score, 4),
                        fused_score=round(fused_score, 4),
                        contributing_rules=contributing_rules,
                        top_features=top_features,
                        state=det_state,
                    ))

        finally:
            runner.cleanup()
            cap.release()

    def _build_report(self) -> dict:
        """Build the error analysis report."""
        counts = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        fp_rules = {}
        fn_gt_labels = {}

        for e in self.errors:
            counts[e.error_type] += 1
            if e.error_type == "FP":
                for r in e.contributing_rules:
                    fp_rules[r] = fp_rules.get(r, 0) + 1
            elif e.error_type == "FN":
                fn_gt_labels[e.ground_truth] = fn_gt_labels.get(e.ground_truth, 0) + 1

        tp, fp, tn, fn = counts["TP"], counts["FP"], counts["TN"], counts["FN"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        # Sort FP rules by count
        fp_rules_sorted = dict(sorted(fp_rules.items(), key=lambda x: -x[1]))

        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_entries": len(self.errors),
            "counts": counts,
            "metrics": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "false_positive_rate": round(fpr, 4),
                "false_negative_rate": round(fnr, 4),
            },
            "false_positive_rule_attribution": fp_rules_sorted,
            "false_negative_gt_distribution": fn_gt_labels,
            "sample_errors": {
                "false_positives": [
                    asdict(e) for e in self.errors if e.error_type == "FP"
                ][:50],
                "false_negatives": [
                    asdict(e) for e in self.errors if e.error_type == "FN"
                ][:50],
            },
        }

        # Save
        out = Path("evaluation/reports/error_analysis.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=4)
        logger.info(f"Error analysis saved to {out}")

        return report

    def get_errors(self, error_type: str = None) -> List[ErrorEntry]:
        """Get error entries, optionally filtered."""
        if error_type:
            return [e for e in self.errors if e.error_type == error_type]
        return self.errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P4.5 Error Analysis")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--model", default=None, help="Path to ML model pickle")
    args = parser.parse_args()

    engine = ErrorAnalysisEngine(args.dataset_dir)
    report = engine.analyze(args.model)
    print(json.dumps(report.get("metrics", {}), indent=4))
