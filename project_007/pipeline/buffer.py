"""
PROJECT 007 — Rolling Buffer System
Per-track keypoint history stored in bounded deques.
"""

import time
from collections import deque

from config import BUFFER_SIZE, STALE_TIMEOUT
from utils.logger import get_logger

logger = get_logger(__name__)


class TrackBuffer:
    """
    Maintains a rolling window of ``(keypoints, timestamp, center)``
    for every tracked person.

    * Each track ID maps to a ``deque(maxlen=BUFFER_SIZE)``.
    * Stale tracks (not seen for ``STALE_TIMEOUT`` seconds) are
      garbage-collected by :meth:`cleanup`.
    """

    def __init__(self):
        self._buffers: dict[int, deque] = {}   # track_id → deque
        self._last_seen: dict[int, float] = {} # track_id → timestamp
        logger.info(
            f"TrackBuffer initialised  "
            f"(size={BUFFER_SIZE}, stale_timeout={STALE_TIMEOUT}s)"
        )

    # ── public API ────────────────────────────────────

    def update(self, track_id: int, keypoints: dict, bbox: tuple[int, int, int, int], timestamp: float | None = None, frame_id: int | None = None, pose_frame_id: int | None = None) -> None:
        """Append a snapshot to the track's rolling history."""
        if timestamp is None:
            timestamp = time.time()

        center = self._compute_center(keypoints)

        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=BUFFER_SIZE)
            logger.info(f"New track registered: ID {track_id}")

        self._buffers[track_id].append({
            "keypoints": keypoints,
            "bbox": bbox,
            "timestamp": timestamp,
            "center": center,
            "frame_id": frame_id,
            "pose_frame_id": pose_frame_id,
        })
        self._last_seen[track_id] = timestamp

    def get_history(self, track_id: int) -> deque:
        """Return the rolling deque for *track_id* (empty deque if unknown)."""
        return self._buffers.get(track_id, deque())

    def get_active_ids(self) -> set[int]:
        """Return the set of currently active track IDs."""
        return set(self._buffers.keys())

    def cleanup(self, current_time: float | None = None) -> None:
        """Remove tracks that have not been updated within STALE_TIMEOUT."""
        if current_time is None:
            current_time = time.time()

        stale_ids = [
            tid
            for tid, last in self._last_seen.items()
            if (current_time - last) > STALE_TIMEOUT
        ]

        for tid in stale_ids:
            del self._buffers[tid]
            del self._last_seen[tid]
            logger.info(f"Stale track removed: ID {tid}")

    # ── internal ──────────────────────────────────────

    @staticmethod
    def _compute_center(keypoints: dict) -> tuple[float, float]:
        """
        Derive a centre position from the hip midpoint.

        Falls back to the average of all sufficiently-visible keypoints
        if hips are missing.
        """
        left_hip = keypoints.get("left_hip")
        right_hip = keypoints.get("right_hip")

        if left_hip and right_hip:
            return (
                (left_hip["x"] + right_hip["x"]) / 2.0,
                (left_hip["y"] + right_hip["y"]) / 2.0,
            )

        # Fallback: average all visible keypoints
        xs, ys = [], []
        for kp in keypoints.values():
            if isinstance(kp, dict) and kp.get("visibility", 0) > 0.3:
                xs.append(kp["x"])
                ys.append(kp["y"])

        if xs and ys:
            return (sum(xs) / len(xs), sum(ys) / len(ys))

        return (0.0, 0.0)
