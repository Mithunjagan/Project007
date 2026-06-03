"""
PROJECT 007 — P4.5 Per-Class Metrics
Computes precision, recall, F1, and support for each class.

Usage:
    python -m evaluation.class_metrics [--dataset-dir dataset]
"""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import cv2

from pipeline.core import PipelineRunner
from training.feature_schema import ANNOTATION_TO_CLASS, CLASS_LABELS
from dataset_tools.annotation_template import load_annotation
from dataset_tools.video_indexer import VideoIndexer
from utils.logger import get_logger

logger = get_logger(__name__)

# Classes for multi-class evaluation
EVAL_CLASSES = [
    "normal",
    "camera_tamper",
    "intrusion",
    "camera_rush",
    "high_energy_interaction",
]


def _map_state_to_class(state: str, contributing_rules: list) -> str:
    """Map detector state + rules to a class name."""
    if state == "NORMAL":
        return "normal"
    # Check contributing rules
    rule_set = set(contributing_rules)
    if rule_set & {"DIRECTED_ARM_SWING", "SUSTAINED_CONTACT", "RAPID_APPROACH"}:
        return "high_energy_interaction"
    if rule_set & {"CAMERA_SHAKE", "LENS_OCCLUSION", "CAMERA_BLOCKAGE"}:
        return "camera_tamper"
    if rule_set & {"CAMERA_RUSH", "PROXIMITY_INTRUSION"}:
        if "CAMERA_RUSH" in rule_set:
            return "camera_rush"
        return "intrusion"
    # Fallback based on state
    if state in ("SUSPICIOUS", "HIGH_RISK", "CRITICAL"):
        return "intrusion"  # generic anomaly
    return "normal"


class ClassMetricsEngine:
    """Computes per-class precision, recall, F1 for all annotated videos."""

    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.vi = VideoIndexer(dataset_dir)

    def compute(self) -> dict:
        """Run per-class evaluation."""
        index = self.vi.index_all()
        annotated = [v for v in index if v["has_annotation"]]

        if not annotated:
            return {"error": "no_annotated_videos"}

        # Per-class confusion counts
        class_tp = defaultdict(int)
        class_fp = defaultdict(int)
        class_fn = defaultdict(int)
        class_support = defaultdict(int)
        confusion = defaultdict(lambda: defaultdict(int))  # [gt][pred] = count

        for i, video_info in enumerate(annotated):
            video_path = video_info["path"]
            video_id = Path(video_path).stem
            logger.info(f"[{i + 1}/{len(annotated)}] {video_id}")

            annotation = load_annotation(video_id, str(self.dataset_dir / "annotations"))
            if not annotation:
                continue

            self._process_video(video_path, annotation,
                                class_tp, class_fp, class_fn, class_support, confusion)

        # Compute metrics
        per_class = {}
        for cls in EVAL_CLASSES:
            tp = class_tp[cls]
            fp = class_fp[cls]
            fn = class_fn[cls]
            support = class_support[cls]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class[cls] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
                "tp": tp, "fp": fp, "fn": fn,
            }

        # Build confusion matrix
        cm = {}
        for gt_cls in EVAL_CLASSES:
            cm[gt_cls] = {pred_cls: confusion[gt_cls][pred_cls] for pred_cls in EVAL_CLASSES}

        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "classes": EVAL_CLASSES,
            "per_class": per_class,
            "confusion_matrix": cm,
        }

        out = Path("evaluation/reports/class_metrics.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=4)
        logger.info(f"Class metrics saved to {out}")

        # Print summary
        print(f"\n{'Class':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 70)
        for cls in EVAL_CLASSES:
            m = per_class[cls]
            print(f"{cls:<30} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10}")

        return report

    def _process_video(self, video_path, annotation, class_tp, class_fp, class_fn, class_support, confusion):
        """Process a single video for per-class metrics."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        runner = PipelineRunner(
            frame_width, frame_height,
            sync_mode=True, enable_recording=False, log_telemetry=False,
        )

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

                # Ground truth class
                gt_class = "normal"
                for ev in events:
                    if ev.get("start_frame", 0) <= frame_count <= ev.get("end_frame", 0):
                        gt_class = ev.get("label", "normal")
                        break
                class_support[gt_class] += 1

                # Predicted class
                det_state = "NORMAL"
                contributing_rules = []
                for fe in fused:
                    if fe.evidence_score > 0.35:
                        det_state = fe.state
                        contributing_rules = fe.contributing_rules
                        break

                pred_class = _map_state_to_class(det_state, contributing_rules)
                confusion[gt_class][pred_class] += 1

                # Per-class TP/FP/FN
                if gt_class == pred_class:
                    class_tp[gt_class] += 1
                else:
                    class_fn[gt_class] += 1
                    class_fp[pred_class] += 1

        finally:
            runner.cleanup()
            cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P4.5 Per-Class Metrics")
    parser.add_argument("--dataset-dir", default="dataset")
    args = parser.parse_args()

    engine = ClassMetricsEngine(args.dataset_dir)
    engine.compute()
