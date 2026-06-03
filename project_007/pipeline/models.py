"""
PROJECT 007 — Pipeline Data Models
Shared dataclasses carrying temporal metadata through every pipeline stage.

Every object carries frame_id + timestamps so downstream logic can
detect staleness and measure latency at each stage.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameMeta:
    """
    Temporal identity for a single captured frame.

    All timestamps use ``time.perf_counter()`` (monotonic) for latency
    math.  ``wall_ts`` is ``time.time()`` for human-readable logging.
    """
    frame_id: int
    capture_ts: float          # perf_counter at webcam read
    wall_ts: float             # time.time() for display / JSONL


@dataclass
class DetectionResult:
    """Single person detection with full timing metadata."""
    bbox: np.ndarray
    confidence: float
    track_id: int
    meta: FrameMeta
    detect_start_ts: float = 0.0   # perf_counter when YOLO started
    detect_end_ts: float = 0.0     # perf_counter when YOLO finished


@dataclass
class PoseResult:
    """Pose keypoints for one tracked person with queue + inference timing."""
    track_id: int
    keypoints: dict
    meta: FrameMeta
    queue_enter_ts: float = 0.0    # perf_counter when submitted to queue
    pose_start_ts: float = 0.0     # perf_counter when worker picked it up
    pose_end_ts: float = 0.0       # perf_counter when inference completed

    @property
    def queue_wait_ms(self) -> float:
        """How long this item waited inside the pose queue."""
        if self.pose_start_ts and self.queue_enter_ts:
            return (self.pose_start_ts - self.queue_enter_ts) * 1000.0
        return 0.0

    @property
    def inference_ms(self) -> float:
        """Pure MediaPipe inference duration."""
        if self.pose_end_ts and self.pose_start_ts:
            return (self.pose_end_ts - self.pose_start_ts) * 1000.0
        return 0.0


@dataclass
class MotionResult:
    """Motion features for one tracked person."""
    track_id: int
    arm_velocity: float = 0.0
    body_displacement: float = 0.0
    fall_score: float = 0.0
    meta: FrameMeta = field(default=None)
