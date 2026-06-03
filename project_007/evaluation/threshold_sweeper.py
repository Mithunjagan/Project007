"""
PROJECT 007 — P2.5 Threshold Sweeper
Automated grid-search over rule weights and thresholds.

Outputs:
  evaluation/reports/precision_recall.csv
  evaluation/reports/threshold_results.json
  evaluation/reports/calibration_recommendations.json
"""

import csv
import json
import time
from pathlib import Path
from typing import Dict, List

from pipeline.core import PipelineRunner
from evaluation.dataset_manager import DatasetManager
from evaluation.annotation_loader import AnnotationLoader
from evaluation.metrics_engine import MetricsEngine
from evaluation.confusion_matrix import ConfusionMatrix
from utils.logger import get_logger

logger = get_logger(__name__)


class ThresholdSweeper:
    def __init__(self, dataset_dir="dataset", output_dir="evaluation/reports"):
        self.dataset_manager = DatasetManager(dataset_dir)
        self.annotation_loader = AnnotationLoader(f"{dataset_dir}/annotations")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_sweep(self, rule_weights_list: List[Dict[str, float]], videos: List[str] = None):
        """
        Evaluate multiple sets of rule weights over the dataset.
        All runs use sync_mode=True for determinism.
        """
        all_videos = self.dataset_manager.list_videos()
        if videos:
            all_videos = [
                v for v in all_videos
                if v["video_id"] in videos or v.get("original_filename") in videos
            ]

        if not all_videos:
            logger.warning("No videos found in dataset. Aborting sweep.")
            return []

        results = []
        csv_rows = []

        for idx, weights in enumerate(rule_weights_list):
            logger.info(f"━━━ Sweep {idx + 1}/{len(rule_weights_list)} ━━━")
            logger.info(f"  Weights: {weights}")

            metrics = MetricsEngine()
            cm = ConfusionMatrix()
            start_time = time.time()

            for video_meta in all_videos:
                self._evaluate_video(video_meta, weights, metrics, cm)

            elapsed = time.time() - start_time
            computed = metrics.compute_metrics()

            sweep_result = {
                "sweep_id": idx,
                "weights": weights,
                "metrics": computed,
                "confusion_matrix": cm.to_dict(),
                "time_taken_sec": round(elapsed, 2),
            }
            results.append(sweep_result)

            # CSV row
            csv_rows.append({
                "sweep_id": idx,
                "precision": computed["precision"],
                "recall": computed["recall"],
                "f1_score": computed["f1_score"],
                "false_positive_rate": computed["false_positive_rate"],
                "false_negative_rate": computed["false_negative_rate"],
                "fp_frames": computed["false_positives_frames"],
                "fn_frames": computed["false_negatives_frames"],
                "weights": json.dumps(weights),
            })

            logger.info(
                f"  P={computed['precision']:.4f}  R={computed['recall']:.4f}  "
                f"F1={computed['f1_score']:.4f}  FP={computed['false_positives_frames']}  "
                f"FN={computed['false_negatives_frames']}  ({elapsed:.1f}s)"
            )

        # ── Save outputs ──

        # 1. threshold_results.json
        with open(self.output_dir / "threshold_results.json", "w") as f:
            json.dump(results, f, indent=4)

        # 2. precision_recall.csv
        if csv_rows:
            csv_path = self.output_dir / "precision_recall.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(csv_rows)

        # 3. calibration_recommendations.json
        self._generate_recommendations(results)

        logger.info(f"Sweep complete. Reports saved to {self.output_dir}")
        return results

    def _evaluate_video(
        self, video_meta: Dict, weights: Dict,
        metrics: MetricsEngine, cm: ConfusionMatrix
    ):
        import cv2

        video_path = video_meta.get("video_path")
        if not video_path:
            return

        annotations = self.annotation_loader.get_events_for_video(
            video_meta.get("original_filename", "")
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        runner = PipelineRunner(
            frame_width, frame_height,
            sync_mode=True,
            enable_recording=False,
            log_telemetry=False,
        )
        # Override weights in the fusion engine
        runner.fusion_engine._accumulator._weights.update(weights)

        frame_count = 0
        now_mono = time.perf_counter()
        now_wall = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                now_mono += 1.0 / fps
                now_wall += 1.0 / fps

                _, _, fused = runner.step(frame, frame_count, now_mono, now_wall)

                expected_state = "NORMAL"
                for ann in annotations:
                    if ann.start_frame <= frame_count <= ann.end_frame:
                        expected_state = ann.label.upper()
                        break

                actual_state = "NORMAL"
                highest_score = -1
                for fe in fused:
                    if fe.evidence_score > highest_score:
                        highest_score = fe.evidence_score
                        actual_state = fe.state

                metrics.record_frame(frame_count, expected_state, actual_state, fused)
                cm.add_result(expected_state, actual_state)

        finally:
            runner.cleanup()
            cap.release()

    def _generate_recommendations(self, results: List[Dict]):
        """Generate calibration recommendations based on sweep results."""
        if not results:
            return

        # Find best F1
        best_f1 = -1
        best_sweep = None
        for res in results:
            f1 = res["metrics"]["f1_score"]
            if f1 > best_f1:
                best_f1 = f1
                best_sweep = res

        # Identify top false-positive rules across all sweeps
        all_fp_rules = {}
        for res in results:
            for rule, count in res["metrics"].get("false_positive_rule_attribution", {}).items():
                all_fp_rules[rule] = all_fp_rules.get(rule, 0) + count

        sorted_fp_rules = sorted(all_fp_rules.items(), key=lambda x: -x[1])

        recommendations = {
            "best_configuration": {
                "sweep_id": best_sweep["sweep_id"] if best_sweep else None,
                "weights": best_sweep["weights"] if best_sweep else {},
                "f1_score": round(best_f1, 4),
                "precision": best_sweep["metrics"]["precision"] if best_sweep else 0,
                "recall": best_sweep["metrics"]["recall"] if best_sweep else 0,
            },
            "top_false_positive_rules": [
                {"rule": rule, "total_fp_triggers": count}
                for rule, count in sorted_fp_rules[:5]
            ],
            "recommended_threshold_changes": [],
        }

        # Generate specific recommendations
        for rule, count in sorted_fp_rules[:3]:
            recommendations["recommended_threshold_changes"].append({
                "rule": rule,
                "current_fp_triggers": count,
                "suggestion": f"Consider increasing the weight/threshold for {rule} to reduce false positives.",
            })

        with open(self.output_dir / "calibration_recommendations.json", "w") as f:
            json.dump(recommendations, f, indent=4)


if __name__ == "__main__":
    sweeper = ThresholdSweeper()
    # Example sweep configurations
    configs = [
        {"CAMERA_SHAKE": 0.40, "PROXIMITY_INTRUSION": 0.35},
        {"CAMERA_SHAKE": 0.50, "PROXIMITY_INTRUSION": 0.40},
        {"CAMERA_SHAKE": 0.30, "PROXIMITY_INTRUSION": 0.25},
    ]
    sweeper.run_sweep(configs)
