"""
PROJECT 007 — Metrics Tracker
Thread-safe rolling metrics with system resource monitoring.

Lock regions are kept minimal — only shared-state writes and
snapshot reads are protected.  System polling (CPU / GPU) happens
at most once per second to avoid overhead.
"""

import threading
import time
from collections import deque

from config import FPS_AVERAGE_WINDOW
from utils.logger import get_logger

logger = get_logger(__name__)

_ROLLING_WINDOW = FPS_AVERAGE_WINDOW   # 30 samples


class MetricsTracker:
    """
    Central metrics store for the entire pipeline.

    Three kinds of metrics:

    * **rolling** — deque-based averages  (fps, latencies, queue depth …)
    * **counters** — monotonically increasing totals  (drops, rejects …)
    * **gauges** — point-in-time values  (cpu_util, gpu_util, vram …)
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Rolling averages (deque-based)
        self._rolling: dict[str, deque] = {
            "fps":                   deque(maxlen=_ROLLING_WINDOW),
            "detection_latency_ms":  deque(maxlen=_ROLLING_WINDOW),
            "pose_latency_ms":       deque(maxlen=_ROLLING_WINDOW),
            "frame_age_ms":          deque(maxlen=_ROLLING_WINDOW),
            "queue_depth":           deque(maxlen=_ROLLING_WINDOW),
            "queue_wait_ms":         deque(maxlen=_ROLLING_WINDOW),
        }

        # Monotonic counters
        self._counters: dict[str, int] = {
            "dropped_frames":           0,
            "dropped_crops":            0,
            "stale_detections_rejected": 0,
            "stale_poses_rejected":     0,
        }

        # Point-in-time gauges
        self._gauges: dict[str, float] = {
            "active_tracks": 0,
            "gpu_util":      0.0,
            "vram_gb":       0.0,
            "vram_total_gb": 0.0,
            "cpu_util":      0.0,
        }

        # System-resource polling rate limiter
        self._last_sys_poll: float = 0.0
        self._sys_poll_interval: float = 1.0   # seconds

        logger.info("MetricsTracker initialised")

    # ── writes ────────────────────────────────────────

    def update(self, **kwargs) -> None:
        """
        Update one or more metrics.

        * Rolling keys → append value
        * Counter keys → **add** value  (pass delta, not total)
        * Gauge keys → **set** value
        """
        with self._lock:
            for key, value in kwargs.items():
                if key in self._rolling:
                    self._rolling[key].append(value)
                elif key in self._counters:
                    self._counters[key] += int(value)
                elif key in self._gauges:
                    self._gauges[key] = value

    def increment(self, key: str, amount: int = 1) -> None:
        """Increment a counter by *amount*."""
        with self._lock:
            if key in self._counters:
                self._counters[key] += amount

    # ── reads ─────────────────────────────────────────

    def snapshot(self) -> dict:
        """
        Return a flat dict of all current metric values.

        Rolling metrics are averaged.  System resources are polled at
        most once per second (the call is inside the lock but the
        actual polling functions are fast / non-blocking).
        """
        with self._lock:
            now = time.perf_counter()
            if now - self._last_sys_poll > self._sys_poll_interval:
                self._poll_system_unsafe()      # already inside lock
                self._last_sys_poll = now

            result: dict = {}

            # Rolling → averages
            for key, dq in self._rolling.items():
                result[key] = round(sum(dq) / len(dq), 2) if dq else 0.0

            # Counters → totals
            result.update(self._counters)

            # Gauges → latest
            result.update(self._gauges)

            return result

    # ── system resource polling (called inside lock) ──

    def _poll_system_unsafe(self) -> None:
        """Sample CPU and GPU utilisation.  Called rarely."""
        # CPU
        try:
            import psutil
            self._gauges["cpu_util"] = psutil.cpu_percent(interval=None)
        except Exception:
            pass

        # GPU (only when CUDA is available)
        try:
            import torch
            if torch.cuda.is_available():
                # VRAM reserved by the caching allocator
                self._gauges["vram_gb"] = round(
                    torch.cuda.memory_reserved(0) / (1024 ** 3), 2
                )
                # Total GPU memory
                props = torch.cuda.get_device_properties(0)
                self._gauges["vram_total_gb"] = round(
                    props.total_memory / (1024 ** 3), 2
                )
                # Utilisation (PyTorch ≥ 2.0)
                if hasattr(torch.cuda, "utilization"):
                    self._gauges["gpu_util"] = float(
                        torch.cuda.utilization(0)
                    )
        except Exception:
            pass

