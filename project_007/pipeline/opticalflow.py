"""
PROJECT 007 — Optical Flow Engine
Computes dense optical flow on a background worker thread.
Used for camera shake and global motion instability detection.
"""

import threading
import queue
import time
import cv2
import numpy as np

from config import FLOW_DOWNSCALE_WIDTH
from utils.logger import get_logger

logger = get_logger(__name__)


class OpticalFlowWorker:
    """
    Background worker that computes dense optical flow on downscaled frames.
    """
    def __init__(self):
        self._queue = queue.Queue(maxsize=2)
        self._result_lock = threading.Lock()
        
        # Results
        self._avg_flow_mag = 0.0
        self._dominant_direction = (0.0, 0.0)
        self._instability_score = 0.0
        
        self._prev_gray = None
        
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        
        logger.info(f"OpticalFlowWorker started (downscale_w={FLOW_DOWNSCALE_WIDTH})")

    def submit(self, frame: np.ndarray) -> None:
        """Non-blocking frame submission."""
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    def get_metrics(self) -> dict:
        """Returns thread-safe latest optical flow metrics."""
        with self._result_lock:
            return {
                "avg_flow_mag": self._avg_flow_mag,
                "dominant_direction": self._dominant_direction,
                "instability_score": self._instability_score
            }

    def _loop(self):
        while self._running:
            try:
                frame = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if frame is None:
                continue

            # Downscale and grayscale
            h, w = frame.shape[:2]
            scale = FLOW_DOWNSCALE_WIDTH / float(w)
            new_h = int(h * scale)
            
            small = cv2.resize(frame, (FLOW_DOWNSCALE_WIDTH, new_h))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if self._prev_gray is None:
                self._prev_gray = gray
                continue

            # Compute flow
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0
            )
            
            self._prev_gray = gray

            # Metrics
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            avg_mag = float(np.mean(mag))
            
            # Dominant direction (mean vector)
            mean_u = float(np.mean(flow[..., 0]))
            mean_v = float(np.mean(flow[..., 1]))
            
            # Instability: how chaotic is the field?
            # If standard deviation of magnitude is high compared to mean, it's chaotic.
            std_mag = float(np.std(mag))
            instability = std_mag / (avg_mag + 1e-5)

            with self._result_lock:
                self._avg_flow_mag = avg_mag
                self._dominant_direction = (mean_u, mean_v)
                self._instability_score = float(instability)

    def close(self):
        self._running = False
        self._thread.join(timeout=2.0)
        logger.info("OpticalFlowWorker stopped")
