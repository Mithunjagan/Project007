"""
PROJECT 007 — P2.5 Replay Engine
Deterministic replay of recorded videos through the identical inference pipeline.

Usage:
    python -m evaluation.replay_engine <video_path> [category] [--async-replay]

Controls:
    Space  : Pause / Resume
    N      : Step one frame (paused)
    [      : Slower (min 0.25x)
    ]      : Faster (max 4x)
    Q      : Quit
"""

import os
import sys
import json
import time
import argparse
import cv2
from pathlib import Path

from pipeline.core import PipelineRunner
from evaluation.dataset_manager import DatasetManager
from evaluation.annotation_loader import AnnotationLoader
from evaluation.metrics_engine import MetricsEngine
from evaluation.confusion_matrix import ConfusionMatrix
from evaluation.integrity import ReplayIntegrityValidator

from utils.logger import get_logger

logger = get_logger(__name__)


class ReplayEngine:
    """
    Replays MP4 datasets through the exact inference pipeline.
    sync_mode=True by default for deterministic evaluation.
    """
    def __init__(self, dataset_dir="dataset", failures_dir="evaluation/failures", sync_mode=True):
        self.dataset_manager = DatasetManager(dataset_dir)
        self.annotation_loader = AnnotationLoader(f"{dataset_dir}/annotations")
        self.sync_mode = sync_mode

        self.failures_dir = Path(failures_dir)
        self.failures_dir.mkdir(parents=True, exist_ok=True)

        self.current_video_category = "normal"
        self.current_video_name = ""
        self.current_annotations = []

        self.playback_speed = 1.0
        self.is_paused = False

        # Per-run evaluation objects
        self.metrics_engine = MetricsEngine()
        self.confusion_matrix = ConfusionMatrix()
        self.integrity_validator = ReplayIntegrityValidator()

    def _get_expected_state(self, frame_count: int) -> str:
        """Map annotation events to system threat level states."""
        for ann in self.current_annotations:
            if ann.start_frame <= frame_count <= ann.end_frame:
                label = ann.label.lower()
                if label == "high_energy_interaction":
                    return "CRITICAL"
                elif label in ("camera_shake", "lens_occlusion"):
                    return "HIGH_RISK"
                elif label in ("camera_rush", "proximity_intrusion"):
                    return "SUSPICIOUS"
                else:
                    return "NORMAL"
        return "NORMAL"

    def _failure_callback(self, frame_count, frame, fe, raw_rules):
        """Called by PipelineRunner when state reaches CRITICAL."""
        expected_state = self._get_expected_state(frame_count)

        # If expected is normal but we got CRITICAL, it's a False Positive
        if expected_state == "NORMAL":
            # Delegate to PipelineRunner's capture method
            self._runner.capture_failure(
                frame_count, frame, fe, raw_rules, self.current_video_name
            )

    def replay_video(self, video_path: str, category: str = "normal", show_ui: bool = True):
        """
        Replay a single video through the full pipeline.

        Returns
        -------
        dict : Run summary including metrics, confusion matrix, latency, and integrity checksum.
        """
        self.current_video_name = Path(video_path).stem
        self.current_video_category = category
        self.current_annotations = self.annotation_loader.get_events_for_video(
            Path(video_path).name
        )

        # Reset per-run evaluation state
        self.metrics_engine.reset()
        self.confusion_matrix = ConfusionMatrix()
        self.integrity_validator.reset()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return {}

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        mode_str = "SYNC" if self.sync_mode else "ASYNC"
        logger.info(
            f"Replaying {video_path} ({total_frames} frames, {fps:.1f} FPS) "
            f"mode={mode_str} speed={self.playback_speed}x"
        )

        self._runner = PipelineRunner(
            frame_width, frame_height,
            sync_mode=self.sync_mode,
            enable_recording=False,
            log_telemetry=False,
            failures_dir=str(self.failures_dir),
        )
        self._runner.register_failure_callback(self._failure_callback)

        frame_count = 0
        now_mono = time.perf_counter()
        now_wall = time.time()
        prev_state = "NORMAL"

        try:
            while True:
                if not self.is_paused:
                    ret, frame = cap.read()
                    if not ret:
                        logger.info("End of video reached.")
                        break

                    frame_count += 1

                    # Simulate time advancing based on video FPS
                    time_delta = 1.0 / fps
                    now_mono += time_delta
                    now_wall += time_delta

                    annotated, snap, fused = self._runner.step(
                        frame, frame_count, now_mono, now_wall
                    )

                    # Determine expected and actual states
                    expected_state = self._get_expected_state(frame_count)

                    actual_state = "NORMAL"
                    highest_score = 0.0
                    for fe in fused:
                        if fe.evidence_score > highest_score:
                            highest_score = fe.evidence_score
                            actual_state = fe.state

                    # Record evaluation metrics
                    self.metrics_engine.record_frame(
                        frame_count, expected_state, actual_state, fused
                    )
                    self.confusion_matrix.add_result(expected_state, actual_state)
                    self.integrity_validator.record_frame(
                        frame_count, actual_state, highest_score
                    )

                    if actual_state in ["HIGH_RISK", "CRITICAL"]:
                        self.integrity_validator.record_event()

                    if actual_state != prev_state:
                        self.integrity_validator.record_state_transition(
                            frame_count, prev_state, actual_state
                        )
                        prev_state = actual_state

                    if show_ui:
                        # Draw Replay HUD
                        speed_text = f"REPLAY [{mode_str}]: {self.playback_speed}x"
                        if self.is_paused:
                            speed_text += " (PAUSED)"
                        cv2.putText(
                            annotated, speed_text,
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2,
                        )
                        progress = f"Frame {frame_count}/{total_frames}"
                        cv2.putText(
                            annotated, progress,
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), 1,
                        )
                        cv2.imshow("PROJECT 007 - REPLAY ENGINE", annotated)

                # Handle Keyboard Input
                if show_ui:
                    delay = int((1000 / fps) / self.playback_speed) if not self.is_paused else 50
                    key = cv2.waitKey(max(1, delay)) & 0xFF

                    if key == ord("q"):
                        break
                    elif key == ord(" "):
                        self.is_paused = not self.is_paused
                    elif key == ord("n") and self.is_paused:
                        ret, frame = cap.read()
                        if ret:
                            frame_count += 1
                            time_delta = 1.0 / fps
                            now_mono += time_delta
                            now_wall += time_delta
                            annotated, _, fused = self._runner.step(
                                frame, frame_count, now_mono, now_wall
                            )
                            cv2.putText(
                                annotated, "REPLAY: STEP",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 255), 2,
                            )
                            cv2.imshow("PROJECT 007 - REPLAY ENGINE", annotated)
                    elif key == ord("]"):
                        self.playback_speed = min(4.0, self.playback_speed * 2.0)
                        logger.info(f"Playback speed: {self.playback_speed}x")
                    elif key == ord("["):
                        self.playback_speed = max(0.25, self.playback_speed / 2.0)
                        logger.info(f"Playback speed: {self.playback_speed}x")
                else:
                    # Headless mode for threshold sweeping
                    pass

        finally:
            latency_report = self._runner.get_latency_report()
            self._runner.cleanup()
            cap.release()
            if show_ui:
                cv2.destroyAllWindows()

        # Build run summary
        metrics = self.metrics_engine.compute_metrics()
        self.integrity_validator.record_confusion(
            tp=metrics["true_positives_frames"],
            fp=metrics["false_positives_frames"],
            tn=metrics["true_negatives_frames"],
            fn=metrics["false_negatives_frames"],
        )

        run_summary = {
            "video": str(video_path),
            "category": category,
            "total_frames": frame_count,
            "sync_mode": self.sync_mode,
            "metrics": metrics,
            "confusion_matrix": self.confusion_matrix.to_dict(),
            "latency": latency_report,
            "integrity": self.integrity_validator.to_dict(),
        }

        logger.info(f"Replay complete: {frame_count} frames processed.")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  FP frames: {metrics['false_positives_frames']}")
        logger.info(f"  FN frames: {metrics['false_negatives_frames']}")
        logger.info(f"\n{self.confusion_matrix.generate_text_report()}")

        return run_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PROJECT 007 Replay Engine")
    parser.add_argument("video", help="Path to the video file to replay")
    parser.add_argument("category", nargs="?", default="normal", help="Dataset category (default: normal)")
    parser.add_argument("--async-replay", action="store_true", help="Use async mode (stress-test real-world queue behavior)")
    args = parser.parse_args()

    sync = not args.async_replay
    engine = ReplayEngine(sync_mode=sync)
    result = engine.replay_video(args.video, args.category)

    # Save run summary
    out_path = Path("evaluation/reports")
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "last_replay_summary.json", "w") as f:
        json.dump(result, f, indent=4)
    logger.info(f"Run summary saved to evaluation/reports/last_replay_summary.json")
