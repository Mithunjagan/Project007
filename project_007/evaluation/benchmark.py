"""
PROJECT 007 — P3.0 Benchmark Suite
Runs the entire dataset through the replay engine and generates comprehensive metrics.

Usage:
    python -m evaluation.benchmark [--dataset-dir dataset] [--headless]
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List

from evaluation.replay_engine import ReplayEngine
from evaluation.metrics_engine import MetricsEngine
from evaluation.confusion_matrix import ConfusionMatrix
from evaluation.integrity import ReplayIntegrityValidator, validate_replays
from dataset_tools.session_manager import SessionManager
from dataset_tools.video_indexer import VideoIndexer
from dataset_tools.export_manifest import export_manifest
from utils.logger import get_logger

logger = get_logger(__name__)


def _hash_config() -> str:
    """Compute a SHA-256 hash of the current threshold configuration."""
    try:
        import config
        # Collect all relevant threshold values
        config_values = {
            "APPROACH_VELOCITY_THRESHOLD": config.APPROACH_VELOCITY_THRESHOLD,
            "ARM_SWING_THRESHOLD": config.ARM_SWING_THRESHOLD,
            "FALL_SCORE_THRESHOLD": config.FALL_SCORE_THRESHOLD,
            "CONTACT_DISTANCE_THRESHOLD": config.CONTACT_DISTANCE_THRESHOLD,
            "CROWD_DISPERSION_THRESHOLD": config.CROWD_DISPERSION_THRESHOLD,
            "DIRECTION_THRESHOLD": config.DIRECTION_THRESHOLD,
            "RULE_ACTIVE_MIN_FRAMES": config.RULE_ACTIVE_MIN_FRAMES,
            "PERSISTENCE_DECAY": config.PERSISTENCE_DECAY,
            "RISK_DECAY": config.RISK_DECAY,
            "MAX_RISK_SCORE": config.MAX_RISK_SCORE,
            "TAMPER_DARK_PIXEL_RATIO": config.TAMPER_DARK_PIXEL_RATIO,
            "TAMPER_SHAKE_MAGNITUDE": config.TAMPER_SHAKE_MAGNITUDE,
            "INTRUSION_AREA_GROWTH_RATE": config.INTRUSION_AREA_GROWTH_RATE,
            "INTRUSION_MAX_OCCUPANCY": config.INTRUSION_MAX_OCCUPANCY,
        }
        raw = json.dumps(config_values, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()
    except Exception as e:
        logger.warning(f"Could not hash config: {e}")
        return "unknown"


class BenchmarkSuite:
    """
    Runs the entire dataset through the replay engine.
    Generates precision, recall, F1, false positives/hour, latency, and rule attribution.
    """

    def __init__(self, dataset_dir: str = "dataset", output_dir: str = "evaluation/reports"):
        self.dataset_dir = dataset_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, headless: bool = True) -> dict:
        """
        Run the full benchmark suite over the dataset.

        Parameters
        ----------
        headless : bool
            If True, no OpenCV windows are shown (faster).

        Returns
        -------
        dict : Full benchmark results.
        """
        vi = VideoIndexer(self.dataset_dir)
        index = vi.index_all()

        if not index:
            logger.warning("No videos found in dataset. Cannot run benchmark.")
            return {"error": "no_videos"}

        logger.info(f"Benchmark starting: {len(index)} videos")

        # Aggregate metrics across all videos
        aggregate_metrics = MetricsEngine()
        aggregate_cm = ConfusionMatrix()
        aggregate_integrity = ReplayIntegrityValidator()

        per_video_results = []
        total_duration_sec = 0.0
        all_latency_reports = []

        start_time = time.time()

        for i, video_info in enumerate(index):
            video_path = video_info["path"]
            category = video_info["category"]
            duration = video_info.get("duration_sec", 0)
            total_duration_sec += duration

            logger.info(
                f"[{i + 1}/{len(index)}] {video_info['filename']} "
                f"({category}, {duration:.1f}s)"
            )

            try:
                engine = ReplayEngine(
                    dataset_dir=self.dataset_dir,
                    sync_mode=True,
                )
                result = engine.replay_video(
                    video_path,
                    category=category,
                    show_ui=not headless,
                )

                if result:
                    per_video_results.append(result)

                    # Merge latency
                    if result.get("latency"):
                        all_latency_reports.append(result["latency"])

            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                per_video_results.append({
                    "video": video_path,
                    "error": str(e),
                })

        elapsed = time.time() - start_time

        # Aggregate per-video metrics
        total_fp = 0
        total_fn = 0
        total_tp = 0
        total_tn = 0
        total_frames = 0
        all_rule_triggers = {}
        all_fp_attribution = {}

        for r in per_video_results:
            if "error" in r:
                continue
            m = r.get("metrics", {})
            total_tp += m.get("true_positives_frames", 0)
            total_fp += m.get("false_positives_frames", 0)
            total_tn += m.get("true_negatives_frames", 0)
            total_fn += m.get("false_negatives_frames", 0)
            total_frames += m.get("frames_analyzed", 0)

            for rule, count in m.get("rule_triggers", {}).items():
                all_rule_triggers[rule] = all_rule_triggers.get(rule, 0) + count

            for rule, count in m.get("false_positive_rule_attribution", {}).items():
                all_fp_attribution[rule] = all_fp_attribution.get(rule, 0) + count

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # False positives per hour
        hours = total_duration_sec / 3600 if total_duration_sec > 0 else 1
        fp_per_hour = total_fp / hours

        # Aggregate latency
        avg_latency = {}
        if all_latency_reports:
            for key in ["avg_frame_age_ms", "p95_frame_age_ms", "p99_frame_age_ms",
                         "avg_yolo_ms", "avg_pose_ms"]:
                vals = [r.get(key, 0) for r in all_latency_reports if r.get(key, 0) > 0]
                avg_latency[key] = round(sum(vals) / max(1, len(vals)), 2)

        # Sort rule attribution
        sorted_rule_triggers = dict(sorted(all_rule_triggers.items(), key=lambda x: -x[1]))
        sorted_fp_attribution = dict(sorted(all_fp_attribution.items(), key=lambda x: -x[1]))

        benchmark_result = {
            "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config_hash": _hash_config(),
            "dataset_dir": self.dataset_dir,
            "total_videos": len(index),
            "total_frames": total_frames,
            "total_duration_sec": round(total_duration_sec, 2),
            "total_duration_hours": round(hours, 4),
            "benchmark_elapsed_sec": round(elapsed, 2),
            "aggregate_metrics": {
                "true_positives": total_tp,
                "false_positives": total_fp,
                "true_negatives": total_tn,
                "false_negatives": total_fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "false_positives_per_hour": round(fp_per_hour, 2),
            },
            "latency": avg_latency,
            "rule_attribution": {
                "total_triggers": sorted_rule_triggers,
                "false_positive_attribution": sorted_fp_attribution,
            },
            "per_video_results": per_video_results,
        }

        # Save benchmark results
        out_path = self.output_dir / "benchmark_results.json"
        with open(out_path, "w") as f:
            json.dump(benchmark_result, f, indent=4)
        logger.info(f"Benchmark results saved to {out_path}")

        # Print summary
        logger.info("=" * 60)
        logger.info("  BENCHMARK SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Videos     : {len(index)}")
        logger.info(f"  Frames     : {total_frames}")
        logger.info(f"  Duration   : {hours:.2f} hours")
        logger.info(f"  Precision  : {precision:.4f}")
        logger.info(f"  Recall     : {recall:.4f}")
        logger.info(f"  F1         : {f1:.4f}")
        logger.info(f"  FP/hour    : {fp_per_hour:.2f}")
        logger.info(f"  Elapsed    : {elapsed:.1f}s")
        logger.info("=" * 60)

        return benchmark_result

    def generate_baseline_report(self, benchmark_result: dict = None) -> dict:
        """
        Generate a baseline report from the latest benchmark results.
        """
        if benchmark_result is None:
            results_path = self.output_dir / "benchmark_results.json"
            if results_path.exists():
                with open(results_path, "r") as f:
                    benchmark_result = json.load(f)
            else:
                logger.error("No benchmark results found. Run benchmark first.")
                return {}

        # Get dataset manifest
        manifest = export_manifest(self.dataset_dir)

        # Compute replay integrity checksum from per-video results
        integrity_checksums = []
        for r in benchmark_result.get("per_video_results", []):
            integrity = r.get("integrity", {})
            if integrity.get("checksum"):
                integrity_checksums.append(integrity["checksum"])

        combined_checksum = hashlib.sha256(
            json.dumps(integrity_checksums, sort_keys=True).encode()
        ).hexdigest()

        baseline = {
            "baseline_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dataset_version": manifest.get("dataset_version", "unknown"),
            "total_videos": manifest.get("total_videos", 0),
            "total_duration_hours": manifest.get("total_duration_hours", 0),
            "threshold_configuration_hash": benchmark_result.get("config_hash", "unknown"),
            "benchmark_metrics": benchmark_result.get("aggregate_metrics", {}),
            "latency_metrics": benchmark_result.get("latency", {}),
            "rule_attribution": benchmark_result.get("rule_attribution", {}),
            "replay_integrity_checksum": combined_checksum,
        }

        baseline_path = self.output_dir / "baseline_report.json"
        with open(baseline_path, "w") as f:
            json.dump(baseline, f, indent=4)

        logger.info(f"Baseline report saved to {baseline_path}")
        return baseline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PROJECT 007 — Benchmark Suite")
    parser.add_argument("--dataset-dir", default="dataset", help="Dataset root")
    parser.add_argument("--headless", action="store_true", help="No UI (faster)")
    parser.add_argument("--baseline", action="store_true", help="Also generate baseline report")
    args = parser.parse_args()

    suite = BenchmarkSuite(dataset_dir=args.dataset_dir)
    result = suite.run(headless=args.headless)

    if args.baseline:
        suite.generate_baseline_report(result)
