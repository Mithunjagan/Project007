"""
PROJECT 007 — Pose Extraction
Asynchronous MediaPipe PoseLandmarker (Tasks API) on CPU with a
background worker thread.

P0.5 additions
--------------
* Drop-oldest queue backpressure  (newer frames > older frames)
* Queue wait time tracking  (queue_enter_ts → pose_start_ts)
* Dropped-crop counter  (thread-safe)
* Thread heartbeat  (main loop can detect stalled worker)
* Stress test mode  (configurable artificial delay)
* Results carry full FrameMeta + timing
"""

import os
import queue
import random
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

from config import (
    POSE_MODEL_URL,
    POSE_MODEL_PATH,
    POSE_MIN_DETECTION_CONFIDENCE,
    POSE_MIN_TRACKING_CONFIDENCE,
    MAX_POSE_CROP,
    POSE_QUEUE_MAX,
    ENABLE_STRESS_TEST,
    STRESS_DELAY_MIN,
    STRESS_DELAY_MAX,
)
from pipeline.models import FrameMeta, PoseResult
from utils.logger import get_logger

logger = get_logger(__name__)

# MediaPipe PoseLandmarker landmark indices
KEYPOINT_INDICES = {
    "left_shoulder":  11,  "right_shoulder": 12,
    "left_elbow":     13,  "right_elbow":    14,
    "left_wrist":     15,  "right_wrist":    16,
    "left_hip":       23,  "right_hip":      24,
    "left_knee":      25,  "right_knee":     26,
    "left_ankle":     27,  "right_ankle":    28,
}


def _ensure_model(model_path: str, model_url: str) -> str:
    """Download the PoseLandmarker .task model if it doesn't exist."""
    if os.path.isfile(model_path):
        return model_path
    logger.info(f"Downloading PoseLandmarker model → {model_path} …")
    try:
        urllib.request.urlretrieve(model_url, model_path)
        logger.info("Model download complete.")
    except Exception as e:
        logger.error(f"Failed to download pose model: {e}")
        raise
    return model_path


class PoseExtractor:
    """
    Asynchronous pose extraction with observability hooks.

    Queue policy: **drop oldest** — when the queue is full the oldest
    item is drained so the freshest crop always gets in.
    """

    def __init__(self):
        model_path = _ensure_model(POSE_MODEL_PATH, POSE_MODEL_URL)

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

        # Bounded queue — size controlled by config
        self._crop_queue: queue.Queue = queue.Queue(maxsize=POSE_QUEUE_MAX)

        # Thread-safe results
        self._results_lock = threading.Lock()
        self._results: dict[int, PoseResult] = {}

        # Counters (thread-safe via atomic-ish int ops + lock)
        self._dropped_crops: int = 0
        self._counter_lock = threading.Lock()

        # Heartbeat — updated every worker iteration (monotonic)
        self._heartbeat_ts: float = time.perf_counter()
        self._heartbeat_lock = threading.Lock()

        # Worker thread
        self._running = True
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        logger.info(
            f"PoseExtractor initialised  "
            f"(queue_max={POSE_QUEUE_MAX}, stress_test={ENABLE_STRESS_TEST})"
        )

    # ── public API ────────────────────────────────────

    def submit(self, track_id: int, frame, bbox, meta: FrameMeta = None) -> None:
        """
        Enqueue a crop job.  **Non-blocking.**

        Drop-oldest policy: if the queue is full, drain the oldest
        item and push the new one (fresher frames are more valuable).
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop_w, crop_h = x2 - x1, y2 - y1
        if crop_w < 10 or crop_h < 10:
            return

        crop = frame[y1:y2, x1:x2].copy()

        # Resize oversized crops
        max_dim = max(crop_w, crop_h)
        if max_dim > MAX_POSE_CROP:
            scale = MAX_POSE_CROP / max_dim
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_LINEAR)

        queue_enter_ts = time.perf_counter()
        item = (track_id, crop, (x1, y1, x2, y2), meta, queue_enter_ts)

        try:
            self._crop_queue.put_nowait(item)
        except queue.Full:
            # Drop-oldest: drain one old item, push the new one
            try:
                self._crop_queue.get_nowait()
            except queue.Empty:
                pass
            with self._counter_lock:
                self._dropped_crops += 1
            try:
                self._crop_queue.put_nowait(item)
            except queue.Full:
                pass  # extreme contention — give up silently

    def get_results(self) -> dict[int, PoseResult]:
        """Return all available PoseResults and clear the buffer."""
        with self._results_lock:
            results = dict(self._results)
            self._results.clear()
        return results

    def get_queue_depth(self) -> int:
        """Current number of items waiting in the crop queue."""
        return self._crop_queue.qsize()

    def get_dropped_crops(self) -> int:
        """Total number of crops dropped due to backpressure."""
        with self._counter_lock:
            return self._dropped_crops

    def get_heartbeat_age(self) -> float:
        """Seconds since the worker thread last reported alive."""
        with self._heartbeat_lock:
            return time.perf_counter() - self._heartbeat_ts

    # ── background worker ─────────────────────────────

    def _worker_loop(self) -> None:
        """Process crops from queue, update heartbeat every iteration."""
        while self._running:
            # Heartbeat update (outside queue wait)
            with self._heartbeat_lock:
                self._heartbeat_ts = time.perf_counter()

            try:
                item = self._crop_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            track_id, crop, bbox, meta, queue_enter_ts = item
            pose_start_ts = time.perf_counter()

            # Optional stress test delay
            if ENABLE_STRESS_TEST:
                time.sleep(random.uniform(STRESS_DELAY_MIN, STRESS_DELAY_MAX))

            try:
                keypoints = self._extract_keypoints(crop, bbox)
                if keypoints is not None:
                    pose_end_ts = time.perf_counter()
                    result = PoseResult(
                        track_id=track_id,
                        keypoints=keypoints,
                        meta=meta,
                        queue_enter_ts=queue_enter_ts,
                        pose_start_ts=pose_start_ts,
                        pose_end_ts=pose_end_ts,
                    )
                    with self._results_lock:
                        self._results[track_id] = result
            except Exception as e:
                logger.warning(f"Pose failed for track {track_id}: {e}")

    def _extract_keypoints(self, crop, bbox):
        """Run PoseLandmarker and map landmarks to frame coordinates."""
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)

        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        landmarks = result.pose_landmarks[0]
        x1, y1, x2, y2 = bbox
        bbox_w, bbox_h = x2 - x1, y2 - y1

        keypoints = {}
        for name, idx in KEYPOINT_INDICES.items():
            lm = landmarks[idx]
            keypoints[name] = {
                "x": x1 + lm.x * bbox_w,
                "y": y1 + lm.y * bbox_h,
                "visibility": lm.visibility,
            }
        return keypoints

    # ── cleanup ───────────────────────────────────────

    def close(self) -> None:
        """Stop the worker thread and release resources."""
        self._running = False
        self._worker.join(timeout=2.0)
        self._landmarker.close()
        logger.info("PoseExtractor closed")
