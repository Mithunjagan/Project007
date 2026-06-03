"""
PROJECT 007 — Stage Profiler
Lightweight context-manager timer for pipeline stages.

Usage::

    profiler = StageProfiler()

    with profiler.measure("yolo"):
        detections = model.track(frame)

    print(profiler.get("yolo"))   # → 73.2  (ms, rolling avg)

Never blocks pipeline execution — pure ``perf_counter`` deltas.
"""

import threading
import time
from collections import deque
from contextlib import contextmanager

from config import FPS_AVERAGE_WINDOW
from utils.logger import get_logger

logger = get_logger(__name__)

_ROLLING_WINDOW = FPS_AVERAGE_WINDOW   # 30 samples


class StageProfiler:
    """
    Records per-stage durations in milliseconds using a rolling deque.

    Thread-safe: multiple threads can record to different stage names
    concurrently.
    """

    def __init__(self):
        self._timings: dict[str, deque] = {}
        self._lock = threading.Lock()
        logger.info("StageProfiler initialised")

    @contextmanager
    def measure(self, stage: str):
        """
        Context manager that times the enclosed block.

        .. code-block:: python

            with profiler.measure("yolo"):
                results = model(frame)
        """
        start = time.perf_counter()
        yield
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        with self._lock:
            if stage not in self._timings:
                self._timings[stage] = deque(maxlen=_ROLLING_WINDOW)
            self._timings[stage].append(elapsed_ms)

    def get(self, stage: str) -> float:
        """Return the rolling average duration in ms for *stage*."""
        with self._lock:
            dq = self._timings.get(stage)
            if not dq:
                return 0.0
            return round(sum(dq) / len(dq), 2)

    def get_all(self) -> dict[str, float]:
        """Return rolling averages for all recorded stages."""
        with self._lock:
            return {
                stage: round(sum(dq) / len(dq), 2) if dq else 0.0
                for stage, dq in self._timings.items()
            }
