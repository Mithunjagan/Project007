"""
PROJECT 007 — FPS Counter
Rolling-average FPS tracker using a fixed-size timestamp deque.
"""

import time
from collections import deque

from config import FPS_AVERAGE_WINDOW


class FPSCounter:
    """Tracks frames-per-second using a rolling average."""

    def __init__(self, window_size: int = FPS_AVERAGE_WINDOW):
        self._timestamps: deque = deque(maxlen=window_size)

    def tick(self) -> None:
        """Record the current frame timestamp."""
        self._timestamps.append(time.perf_counter())

    def get(self) -> float:
        """Return the current FPS as a float (0.0 if insufficient data)."""
        if len(self._timestamps) < 2:
            return 0.0

        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0

        return (len(self._timestamps) - 1) / elapsed
