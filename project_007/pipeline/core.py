"""
PROJECT 007 — PipelineRunner (P2.5)
Reusable core pipeline that supports both live (async) and replay (sync) modes.

sync_mode=True  → Detection and Pose run inline on the calling thread.
                   No frame drops. Deterministic results. Used by ReplayEngine.
sync_mode=False → Original async threads (DetectionWorker + PoseExtractor).
                   Used by main.py for live camera.
"""

import os
import json
import time
import hashlib
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from config import (
    POSE_EVERY_N_FRAMES,
    FLOW_EVERY_N_FRAMES,
    MAX_RESULT_AGE_MS,
    HEARTBEAT_TIMEOUT_S,
    FAILURE_CLIP_ENABLED,
    FAILURE_CLIP_SECONDS_BEFORE,
    FAILURE_CLIP_SECONDS_AFTER,
    TARGET_FPS,
    DL_ENCODE_EVERY_N_FRAMES,
    DL_TEMPORAL_WINDOW,
    ENABLE_EGOMOTION,
)
from pipeline.models import FrameMeta
from pipeline.detector import PersonDetector
from pipeline.pose import PoseExtractor
from pipeline.buffer import TrackBuffer
from pipeline.rules import ProxyRuleEngine
from pipeline.fusion import FusionEngine
from pipeline.smoothing import EMASmoother
from pipeline.recorder import ClipRecorder
from pipeline.motion import MotionEngine
from pipeline.opticalflow import OpticalFlowWorker
from pipeline.scene import SceneDynamicsEngine
from pipeline.tamper import CameraTamperEngine
from pipeline.intrusion import IntrusionEngine
from telemetry.metrics import MetricsTracker
from telemetry.profiler import StageProfiler
from telemetry.recorder import TelemetryRecorder
from ui.display import DebugOverlay
from utils.fps import FPSCounter
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Async Detection Worker (used in live mode)
# ─────────────────────────────────────────────
class DetectionWorker:
    def __init__(self, detector: PersonDetector):
        self._detector = detector
        self._lock = threading.Lock()
        self._frame = None
        self._meta: FrameMeta | None = None
        self._frame_ready = threading.Event()
        self._detections = []
        self._heartbeat_ts: float = time.perf_counter()
        self._heartbeat_lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("DetectionWorker started (async YOLO thread)")

    def submit(self, frame, meta: FrameMeta) -> None:
        with self._lock:
            self._frame = frame
            self._meta = meta
        self._frame_ready.set()

    def get_detections(self):
        with self._lock:
            return list(self._detections)

    def get_heartbeat_age(self) -> float:
        with self._heartbeat_lock:
            return time.perf_counter() - self._heartbeat_ts

    def _loop(self) -> None:
        while self._running:
            with self._heartbeat_lock:
                self._heartbeat_ts = time.perf_counter()

            if not self._frame_ready.wait(timeout=0.1):
                continue
            self._frame_ready.clear()

            with self._lock:
                frame = self._frame
                meta = self._meta
                self._frame = None

            if frame is None:
                continue

            detections = self._detector.detect(frame, meta)

            with self._lock:
                self._detections = detections

    def stop(self) -> None:
        self._running = False
        self._frame_ready.set()
        self._thread.join(timeout=2.0)
        logger.info("DetectionWorker stopped")


def _is_stale(ts: float, now: float) -> bool:
    age_ms = (now - ts) * 1000.0
    return age_ms > MAX_RESULT_AGE_MS


# ─────────────────────────────────────────────
# Latency Percentile Tracker
# ─────────────────────────────────────────────
class LatencyTracker:
    """Collects latency samples and computes percentiles."""
    def __init__(self, max_samples=10000):
        self._frame_ages: list[float] = []
        self._yolo_lats: list[float] = []
        self._pose_lats: list[float] = []
        self._max = max_samples

    def add_frame_age(self, ms: float):
        if len(self._frame_ages) < self._max:
            self._frame_ages.append(ms)

    def add_yolo(self, ms: float):
        if len(self._yolo_lats) < self._max:
            self._yolo_lats.append(ms)

    def add_pose(self, ms: float):
        if len(self._pose_lats) < self._max:
            self._pose_lats.append(ms)

    def _percentile(self, data: list, p: float) -> float:
        if not data:
            return 0.0
        s = sorted(data)
        idx = int(len(s) * p / 100.0)
        idx = min(idx, len(s) - 1)
        return round(s[idx], 2)

    def report(self) -> dict:
        return {
            "avg_frame_age_ms": round(sum(self._frame_ages) / max(1, len(self._frame_ages)), 2),
            "p95_frame_age_ms": self._percentile(self._frame_ages, 95),
            "p99_frame_age_ms": self._percentile(self._frame_ages, 99),
            "avg_yolo_ms": round(sum(self._yolo_lats) / max(1, len(self._yolo_lats)), 2),
            "avg_pose_ms": round(sum(self._pose_lats) / max(1, len(self._pose_lats)), 2),
        }


# ─────────────────────────────────────────────
# PipelineRunner
# ─────────────────────────────────────────────
class PipelineRunner:
    """
    Encapsulates the core pipeline logic so it can be reused by main.py
    (live camera) and replay_engine.py (recorded datasets).

    Parameters
    ----------
    frame_width, frame_height : int
        Resolution of the input frames.
    sync_mode : bool
        If True, detection and pose run synchronously on the calling thread.
        Guarantees determinism and zero frame drops. Used for evaluation.
    enable_recording : bool
        If True, enables the ClipRecorder ring buffer.
    log_telemetry : bool
        If True, writes JSONL telemetry logs to disk.
    failures_dir : str
        Directory for automatic failure captures.
    """
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        sync_mode: bool = False,
        enable_recording: bool = True,
        log_telemetry: bool = True,
        failures_dir: str = "evaluation/failures",
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.sync_mode = sync_mode

        mode_str = "SYNC (deterministic)" if sync_mode else "ASYNC (live)"
        logger.info(f"Initialising PipelineRunner [{mode_str}] …")

        self.detector = PersonDetector()

        if sync_mode:
            self.det_worker = None
            logger.info("Detection: SYNCHRONOUS (inline on main thread)")
        else:
            self.det_worker = DetectionWorker(self.detector)

        self.pose_extractor = PoseExtractor()
        self.track_buffer = TrackBuffer()
        self.motion_engine = MotionEngine()
        self.rule_engine = ProxyRuleEngine()

        rule_weights = {
            "DIRECTED_ARM_SWING": 0.35, "RAPID_APPROACH": 0.30,
            "FALL_EVENT": 0.20, "SUSTAINED_CONTACT": 0.10, "CROWD_DISPERSION": 0.05,
            "CAMERA_SHAKE": 0.40, "LENS_OCCLUSION": 0.50, "CAMERA_BLOCKAGE": 0.30,
            "CAMERA_RUSH": 0.40, "PROXIMITY_INTRUSION": 0.35, "ABNORMAL_SINGLE_SUBJECT_ENERGY": 0.30,
        }
        self.fusion_engine = FusionEngine(rule_weights)
        self.smoother = EMASmoother(alpha=0.3)

        self.enable_recording = enable_recording
        self.recorder = ClipRecorder() if enable_recording else None

        self.opticalflow_worker = OpticalFlowWorker()
        self.scene_engine = SceneDynamicsEngine()
        self.tamper_engine = CameraTamperEngine()
        self.intrusion_engine = IntrusionEngine()

        self.overlay = DebugOverlay()
        self.fps_counter = FPSCounter()

        self.metrics = MetricsTracker()
        self.profiler = StageProfiler()

        self.log_telemetry = log_telemetry
        self.jsonl_recorder = TelemetryRecorder() if log_telemetry else None

        self.latest_keypoints: dict = {}
        self.latest_pose_results: dict = {}
        self.current_detections = []

        # Failure capture
        self.failures_dir = Path(failures_dir)
        self.failures_dir.mkdir(parents=True, exist_ok=True)
        self.failure_callbacks = []
        self._failure_frame_buffer = deque(
            maxlen=int(TARGET_FPS * FAILURE_CLIP_SECONDS_BEFORE)
        )

        # Latency tracking
        self.latency_tracker = LatencyTracker()

        # Sync-mode pose: accumulate results inline
        self._sync_pose_results: dict = {}

        # P5.1: Soft ReID
        from pipeline.reid import SoftReIDTracker
        self.reid_tracker = SoftReIDTracker()

        # P6.0-A: Egomotion Compensation
        self._egomotion_affine = None  # Current frame's 2x3 affine matrix
        self._latest_egomotion = {"translation_x": 0.0, "translation_y": 0.0, "rotation": 0.0, "stability_score": 1.0}
        self._ego_history = deque(maxlen=30)
        self._last_pose_frame_id = {}
        self._latest_motion = {}
        self._latest_pairwise_motion = {}
        if ENABLE_EGOMOTION:
            from pipeline.egomotion import EgomotionEstimator
            self.egomotion_estimator = EgomotionEstimator()
            logger.info("Egomotion Estimator: ENABLED")
        else:
            self.egomotion_estimator = None
            logger.info("Egomotion Estimator: DISABLED")

        # P5.0: Deep Learning Violence Detection
        self._frame_embeddings = deque(maxlen=DL_TEMPORAL_WINDOW)
        self._frame_motion_ctx = deque(maxlen=DL_TEMPORAL_WINDOW)
        self._frame_scene_ctx = deque(maxlen=DL_TEMPORAL_WINDOW)
        try:
            from pipeline.frame_encoder import FrameEncoder
            from models.violence_classifier import ViolenceClassifier
            self.frame_encoder = FrameEncoder()
            self.violence_classifier = ViolenceClassifier()
            dl_status = "LOADED" if self.violence_classifier.is_available() else "NO MODEL (legacy mode)"
            logger.info(f"DL Violence Classifier: {dl_status}")
        except Exception as e:
            self.frame_encoder = None
            self.violence_classifier = None
            logger.info(f"DL Violence Classifier: disabled ({e})")

    def register_failure_callback(self, callback):
        """Register an external callback: callback(frame_id, frame, fe, raw_rules)"""
        self.failure_callbacks.append(callback)

    # ─────────────────────────────────────────
    # Sync-mode helpers
    # ─────────────────────────────────────────
    def _sync_detect(self, frame, meta: FrameMeta):
        """Run detection synchronously (no thread, no queue)."""
        return self.detector.detect(frame, meta)

    def _sync_pose(self, track_id: int, frame, bbox, meta: FrameMeta):
        """Run pose extraction synchronously (inline)."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop_w, crop_h = x2 - x1, y2 - y1
        if crop_w < 10 or crop_h < 10:
            return

        crop = frame[y1:y2, x1:x2].copy()

        max_dim = max(crop_w, crop_h)
        from config import MAX_POSE_CROP
        if max_dim > MAX_POSE_CROP:
            scale = MAX_POSE_CROP / max_dim
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_LINEAR)

        pose_start = time.perf_counter()
        keypoints = self.pose_extractor._extract_keypoints(crop, (x1, y1, x2, y2))
        pose_end = time.perf_counter()

        if keypoints is not None:
            from pipeline.models import PoseResult
            result = PoseResult(
                track_id=track_id,
                keypoints=keypoints,
                meta=meta,
                queue_enter_ts=pose_start,
                pose_start_ts=pose_start,
                pose_end_ts=pose_end,
            )
            self._sync_pose_results[track_id] = result

    def _get_composed_affine(self, t_prev: int, t_curr: int) -> np.ndarray | None:
        """
        Compose frame-to-frame affines from t_prev + 1 to t_curr.
        """
        if t_prev >= t_curr:

            return None

        # Find all affines in history between t_prev + 1 and t_curr
        affines_to_compose = []
        for fid, A in self._ego_history:
            if t_prev < fid <= t_curr:
                affines_to_compose.append(A)

        if not affines_to_compose:
            return None

        # Compose using 3x3 homogeneous matrix multiplication
        H = np.eye(3, dtype=np.float32)
        has_compensation = False
        for A in affines_to_compose:
            if A is not None:
                H_i = np.eye(3, dtype=np.float32)
                H_i[:2, :] = A
                H = H_i @ H
                has_compensation = True

        if has_compensation:
            return H[:2, :]
        return None

    # ─────────────────────────────────────────
    # Main step
    # ─────────────────────────────────────────
    def step(self, frame, frame_count: int, now_mono: float, now_wall: float):
        meta = FrameMeta(frame_id=frame_count, capture_ts=now_mono, wall_ts=now_wall)

        # Failure clip buffer (always maintain for capture)
        self._failure_frame_buffer.append(frame.copy())

        # Background Optical Flow (every N frames)
        if frame_count % FLOW_EVERY_N_FRAMES == 0:
            self.opticalflow_worker.submit(frame)

        if self.recorder:
            self.recorder.update(frame, now_wall)

        # ── Detection ──
        if self.sync_mode:
            raw_detections = self._sync_detect(frame, meta)
        else:
            self.det_worker.submit(frame, meta)
            raw_detections = self.det_worker.get_detections()

        # Stale rejection (only relevant in async mode)
        fresh_detections = []
        stale_det_count = 0
        if self.sync_mode:
            fresh_detections = raw_detections
        else:
            for det in raw_detections:
                if det.meta and _is_stale(det.detect_end_ts, now_mono):
                    stale_det_count += 1
                else:
                    fresh_detections.append(det)

        if stale_det_count > 0:
            self.metrics.increment("stale_detections_rejected", stale_det_count)

        if fresh_detections:
            # Map raw IDs to persistent IDs using Soft ReID
            p_ids = self.reid_tracker.update(fresh_detections, frame, now_wall)
            for idx, det in enumerate(fresh_detections):
                det.track_id = p_ids[idx]
            self.current_detections = fresh_detections
        person_count = len(self.current_detections)

        # ── Pose ──
        if frame_count % POSE_EVERY_N_FRAMES == 0:
            if self.sync_mode:
                self._sync_pose_results.clear()
                for det in self.current_detections:
                    self._sync_pose(det.track_id, frame, det.bbox, meta)
            else:
                for det in self.current_detections:
                    self.pose_extractor.submit(det.track_id, frame, det.bbox, meta)

        # Collect pose results
        if self.sync_mode:
            raw_pose = dict(self._sync_pose_results)
            self._sync_pose_results.clear()
        else:
            raw_pose = self.pose_extractor.get_results()

        stale_pose_count = 0
        new_poses_received = {}
        for tid, pr in raw_pose.items():
            if not self.sync_mode and pr.meta and _is_stale(pr.pose_end_ts, now_mono):
                stale_pose_count += 1
            else:
                self.latest_pose_results[tid] = pr
                self.latest_keypoints[tid] = pr.keypoints
                new_poses_received[tid] = pr

        if stale_pose_count > 0:
            self.metrics.increment("stale_poses_rejected", stale_pose_count)

        # Store bbox in track buffer on every frame
        for det in self.current_detections:
            kps = self.latest_keypoints.get(det.track_id)
            if kps is not None:
                if det.track_id in new_poses_received:
                    self._last_pose_frame_id[det.track_id] = frame_count
                pose_fid = self._last_pose_frame_id.get(det.track_id, frame_count)
                self.track_buffer.update(det.track_id, kps, det.bbox, now_wall, frame_id=frame_count, pose_frame_id=pose_fid)

        active_ids = {det.track_id for det in self.current_detections}
        for tid in [k for k in self.latest_keypoints if k not in active_ids]:
            del self.latest_keypoints[tid]
        for tid in [k for k in self.latest_pose_results if k not in active_ids]:
            del self.latest_pose_results[tid]

        # Clean up cached features for inactive tracks
        for tid in list(self._latest_motion.keys()):
            if tid not in active_ids:
                del self._latest_motion[tid]
        for tid in list(self._last_pose_frame_id.keys()):
            if tid not in active_ids:
                del self._last_pose_frame_id[tid]
        for pair in list(self._latest_pairwise_motion.keys()):
            if pair[0] not in active_ids or pair[1] not in active_ids:
                del self._latest_pairwise_motion[pair]

        self.track_buffer.cleanup(current_time=now_wall)
        self.smoother.cleanup(active_ids)

        # P6.0-A: Compute egomotion affine matrix for camera motion compensation
        ego_affine = None
        if self.egomotion_estimator is not None:
            person_bboxes = [det.bbox for det in self.current_detections]
            ego_result = self.egomotion_estimator.update(frame, person_bboxes)
            self._latest_egomotion = {
                "translation_x": ego_result["translation_x"],
                "translation_y": ego_result["translation_y"],
                "rotation": ego_result["rotation"],
                "stability_score": ego_result["stability_score"],
                "is_compensating": ego_result["is_compensating"],
            }
            # Only apply compensation when motion exceeds noise floor
            if ego_result["is_compensating"]:
                self._egomotion_affine = ego_result["affine_matrix"]
                ego_affine = ego_result["affine_matrix"]
            else:
                self._egomotion_affine = None
        else:
            self._egomotion_affine = None

        # Record egomotion history for temporal composition
        self._ego_history.append((frame_count, ego_affine))

        # Extract raw motion and apply smoothing
        all_motion = {}
        for det in self.current_detections:
            history = self.track_buffer.get_history(det.track_id)
            pose_composed_affine = None
            if len(history) >= 2:
                # Composed affine covers pose interval
                t_prev = history[-2].get("pose_frame_id")
                t_curr = history[-1].get("pose_frame_id")
                if t_prev is not None and t_curr is not None:
                    pose_composed_affine = self._get_composed_affine(t_prev, t_curr)
            
            raw_m = self.motion_engine.compute(
                det.track_id, self.track_buffer, 
                egomotion_affine=self._egomotion_affine, 
                pose_composed_affine=pose_composed_affine
            )
            
            # Smooth the velocities and displacement on every frame
            smooth_vel = self.smoother.update(f"vel_{det.track_id}", raw_m["arm_velocity"])
            smooth_vec = self.smoother.update(f"vec_{det.track_id}", raw_m["arm_motion_vector"])
            smooth_disp = self.smoother.update(f"disp_{det.track_id}", raw_m["body_displacement"])

            raw_m_copy = dict(raw_m)
            raw_m_copy["arm_velocity"] = smooth_vel
            raw_m_copy["arm_motion_vector"] = smooth_vec
            raw_m_copy["body_displacement"] = smooth_disp

            all_motion[det.track_id] = raw_m_copy
            self._latest_motion[det.track_id] = raw_m_copy

        # Pairwise motion evaluation
        pairwise_motion = self.motion_engine.compute_pairwise(self.track_buffer, self._egomotion_affine)
        for pair, vals in pairwise_motion.items():
            self._latest_pairwise_motion[pair] = vals

        flow_metrics = self.opticalflow_worker.get_metrics()

        # Smooth flow metrics
        flow_metrics["avg_flow_mag"] = self.smoother.update(
            "flow_mag", flow_metrics.get("avg_flow_mag", 0.0)
        )

        scene_metrics = self.scene_engine.compute(
            self.current_detections, all_motion, flow_metrics,
            self.frame_width, self.frame_height
        )

        # Proxy Rules Evaluation
        raw_rules = self.rule_engine.evaluate(
            all_motion, pairwise_motion, scene_metrics.scene_stability_score,
            frame_count, now_wall
        )

        tamper_rules = self.tamper_engine.evaluate(
            frame, self.current_detections, flow_metrics, frame_count, now_wall
        )
        intrusion_rules = self.intrusion_engine.evaluate(
            self.current_detections, all_motion, self.track_buffer,
            self.frame_width, self.frame_height, frame_count, now_wall
        )

        raw_rules.extend(tamper_rules)
        raw_rules.extend(intrusion_rules)

        # P5.0: Deep Learning frame encoding
        dl_prediction = None
        if self.frame_encoder and frame_count % DL_ENCODE_EVERY_N_FRAMES == 0:
            try:
                embedding = self.frame_encoder.encode(frame)
                self._frame_embeddings.append(embedding)

                # Aggregate motion features for this frame
                avg_motion = {"arm_velocity": 0.0, "body_displacement": 0.0, "fall_score": 0.0, "approach_velocity": 0.0}
                if all_motion:
                    n = len(all_motion)
                    for m in all_motion.values():
                        avg_motion["arm_velocity"] += m.get("arm_velocity", 0.0) / n
                        avg_motion["body_displacement"] += m.get("body_displacement", 0.0) / n
                        avg_motion["fall_score"] += m.get("fall_score", 0.0) / n
                    if pairwise_motion:
                        for pm in pairwise_motion.values():
                            avg_motion["approach_velocity"] = max(avg_motion["approach_velocity"], pm.get("approach_velocity", 0.0))
                self._frame_motion_ctx.append(avg_motion)
                self._frame_scene_ctx.append({
                    "occupancy_ratio": scene_metrics.occupancy_ratio,
                    "scene_stability": scene_metrics.scene_stability_score,
                })

                # Run classifier when we have enough frames
                if self.violence_classifier and self.violence_classifier.is_available() and len(self._frame_embeddings) >= DL_TEMPORAL_WINDOW:
                    dl_prediction = self.violence_classifier.predict(
                        list(self._frame_embeddings),
                        list(self._frame_motion_ctx),
                        list(self._frame_scene_ctx),
                    )
            except Exception:
                pass  # DL failure must never crash the pipeline

        # P2: Contextual Fusion Engine
        fused_evidence = self.fusion_engine.update(
            raw_rules, scene_metrics.scene_stability_score, person_count,
            all_motion, pairwise_motion, flow_metrics, self.track_buffer, now_wall,
            dl_prediction=dl_prediction
        )

        # Event Qualification & Recording
        is_event = False
        is_recording = False
        for fe in fused_evidence:
            if fe.state in ["HIGH_RISK", "CRITICAL"]:
                is_event = True
                if self.recorder and fe.time_in_state > 2.0:
                    self.recorder.trigger(now_wall)

                # Trigger failure callbacks (external)
                if fe.state == "CRITICAL":
                    for cb in self.failure_callbacks:
                        cb(frame_count, frame, fe, raw_rules)

        if self.recorder:
            is_recording = self.recorder.is_recording

        # ── Telemetry ──
        frame_age_ms = 0.0
        det_latency_ms = 0.0

        if self.current_detections and self.current_detections[0].meta:
            frame_age_ms = (now_mono - self.current_detections[0].meta.capture_ts) * 1000.0
        else:
            frame_age_ms = (now_mono - meta.capture_ts) * 1000.0

        if self.current_detections and self.current_detections[0].detect_end_ts:
            det_latency_ms = (
                self.current_detections[0].detect_end_ts
                - self.current_detections[0].detect_start_ts
            ) * 1000.0

        pose_latency_ms = 0.0
        queue_wait_ms = 0.0
        if self.latest_pose_results:
            pose_lats = [
                pr.inference_ms for pr in self.latest_pose_results.values()
                if pr.inference_ms > 0
            ]
            q_waits = [
                pr.queue_wait_ms for pr in self.latest_pose_results.values()
                if pr.queue_wait_ms > 0
            ]
            if pose_lats:
                pose_latency_ms = sum(pose_lats) / len(pose_lats)
            if q_waits:
                queue_wait_ms = sum(q_waits) / len(q_waits)

        # Latency percentile tracking
        self.latency_tracker.add_frame_age(frame_age_ms)
        if det_latency_ms > 0:
            self.latency_tracker.add_yolo(det_latency_ms)
        if pose_latency_ms > 0:
            self.latency_tracker.add_pose(pose_latency_ms)

        warnings = []
        if not self.sync_mode:
            if self.det_worker and self.det_worker.get_heartbeat_age() > HEARTBEAT_TIMEOUT_S:
                warnings.append("DETECT THREAD STALLED")
            if self.pose_extractor.get_heartbeat_age() > HEARTBEAT_TIMEOUT_S:
                warnings.append("POSE THREAD STALLED")

        self.fps_counter.tick()
        self.metrics.update(
            fps=self.fps_counter.get(),
            frame_age_ms=frame_age_ms,
            detection_latency_ms=det_latency_ms,
            pose_latency_ms=pose_latency_ms,
            queue_depth=self.pose_extractor.get_queue_depth() if not self.sync_mode else 0,
            queue_wait_ms=queue_wait_ms,
            active_tracks=len(active_ids),
            dropped_crops=self.pose_extractor.get_dropped_crops() if not self.sync_mode else 0,
        )

        snap = self.metrics.snapshot()
        snap["person_count"] = person_count
        snap["warnings"] = warnings

        # P2 telemetry (serializable)
        snap["fused_evidence"] = [
            {
                "track_ids": list(fe.track_ids),
                "evidence_score": fe.evidence_score,
                "state": fe.state,
                "contributing_rules": fe.contributing_rules,
                "time_in_state": fe.time_in_state,
            }
            for fe in fused_evidence
        ]
        snap["scene_stability"] = scene_metrics.scene_stability_score
        snap["occupancy_ratio"] = scene_metrics.occupancy_ratio
        snap["global_flow_magnitude"] = flow_metrics.get("avg_flow_mag", 0.0)
        snap["egomotion"] = self._latest_egomotion
        snap["event_candidate"] = is_event
        snap["clip_recording"] = is_recording

        if self.jsonl_recorder:
            self.jsonl_recorder.record(frame_count, now_wall, snap)

        # Render Overlay
        with self.profiler.measure("overlay"):
            annotated = self.overlay.render(
                frame, self.current_detections, all_motion, snap,
                fused_evidence=fused_evidence,
                is_recording=is_recording,
            )

        return annotated, snap, fused_evidence

    # ─────────────────────────────────────────
    # Failure capture (called by ReplayEngine)
    # ─────────────────────────────────────────
    def capture_failure(self, frame_count: int, frame, fe, raw_rules, video_name: str = "unknown"):
        """
        Save a failure dump: screenshot + JSON state + optional clip.
        """
        failure_id = f"FP_{video_name}_f{frame_count}"
        json_path = self.failures_dir / f"{failure_id}.json"

        if json_path.exists():
            return  # Already captured this exact failure

        logger.warning(f"FALSE POSITIVE captured at frame {frame_count} → {failure_id}")

        # A. Screenshot
        cv2.imwrite(str(self.failures_dir / f"{failure_id}.jpg"), frame)

        # B. JSON state dump
        dump = {
            "video": video_name,
            "frame_id": frame_count,
            "timestamp": time.time(),
            "state": fe.state,
            "risk_score": fe.evidence_score,
            "track_ids": list(fe.track_ids),
            "contributing_rules": fe.contributing_rules,
            "active_rules": [r.rule_type for r in raw_rules],
            "time_in_state": fe.time_in_state,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(dump, f, indent=4)

        # C. Optional clip
        if FAILURE_CLIP_ENABLED and len(self._failure_frame_buffer) > 0:
            clip_path = str(self.failures_dir / f"{failure_id}.mp4")
            h, w = self._failure_frame_buffer[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(clip_path, fourcc, TARGET_FPS, (w, h))
            if writer.isOpened():
                for buf_frame in self._failure_frame_buffer:
                    writer.write(buf_frame)
                writer.release()
                logger.info(f"  Failure clip saved: {clip_path}")

    def get_latency_report(self) -> dict:
        return self.latency_tracker.report()

    def cleanup(self):
        logger.info("Cleaning up PipelineRunner resources …")
        if self.det_worker:
            self.det_worker.stop()
        self.pose_extractor.close()
        self.opticalflow_worker.close()
        if self.jsonl_recorder:
            self.jsonl_recorder.close()
        if self.recorder:
            self.recorder.close()
