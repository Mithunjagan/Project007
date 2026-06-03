"""
PROJECT 007 — Telemetry Recorder
Structured JSONL logger with automatic file rotation.

Writes one JSON object per frame to ``logs/telemetry.jsonl``.
Rotates when the file exceeds ``TELEMETRY_MAX_FILE_MB``.
Buffered writes reduce I/O pressure on the main loop.
"""

import json
import os

from config import TELEMETRY_LOG_DIR, TELEMETRY_MAX_FILE_MB, TELEMETRY_FLUSH_INTERVAL
from utils.logger import get_logger

logger = get_logger(__name__)


class TelemetryRecorder:
    """
    Append-only JSONL writer with buffered flush and auto-rotation.

    Call :meth:`record` every frame.  Data is flushed to disk every
    ``TELEMETRY_FLUSH_INTERVAL`` frames to avoid per-frame I/O.
    """

    def __init__(self):
        os.makedirs(TELEMETRY_LOG_DIR, exist_ok=True)
        self._filepath = os.path.join(TELEMETRY_LOG_DIR, "telemetry.jsonl")
        self._buffer: list[str] = []
        self._flush_count: int = 0
        logger.info(f"TelemetryRecorder → {self._filepath}")

    # ── public API ────────────────────────────────────

    def record(self, frame_id: int, wall_ts: float, snapshot: dict) -> None:
        """
        Buffer a single telemetry record.

        Parameters
        ----------
        frame_id : int
            Monotonic frame counter.
        wall_ts : float
            ``time.time()`` for human-readable timestamps.
        snapshot : dict
            Flat metrics snapshot from ``MetricsTracker.snapshot()``.
        """
        entry = {
            "frame_id": frame_id,
            "timestamp": round(wall_ts, 4),
            **{k: round(v, 4) if isinstance(v, float) else v
               for k, v in snapshot.items()},
        }
        self._buffer.append(json.dumps(entry, separators=(",", ":")))
        self._flush_count += 1

        if self._flush_count >= TELEMETRY_FLUSH_INTERVAL:
            self.flush()

    def flush(self) -> None:
        """Write buffered records to disk."""
        if not self._buffer:
            return

        self._rotate_if_needed()

        try:
            with open(self._filepath, "a", encoding="utf-8") as f:
                f.write("\n".join(self._buffer) + "\n")
        except Exception as e:
            logger.warning(f"Telemetry write failed: {e}")

        self._buffer.clear()
        self._flush_count = 0

    def close(self) -> None:
        """Flush remaining records and release resources."""
        self.flush()
        logger.info("TelemetryRecorder closed")

    # ── internal ──────────────────────────────────────

    def _rotate_if_needed(self) -> None:
        """Rename the current file if it exceeds the size limit."""
        if not os.path.exists(self._filepath):
            return

        size_mb = os.path.getsize(self._filepath) / (1024 * 1024)
        if size_mb < TELEMETRY_MAX_FILE_MB:
            return

        # Find next available rotation index
        idx = 1
        while os.path.exists(f"{self._filepath}.{idx}"):
            idx += 1

        rotated = f"{self._filepath}.{idx}"
        os.rename(self._filepath, rotated)
        logger.info(f"Telemetry log rotated → {rotated}")
