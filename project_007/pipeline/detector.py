"""
PROJECT 007 — Person Detection
YOLOv8n person detection on CUDA with ByteTrack persistent tracking.

P0.5: accepts FrameMeta, returns DetectionResult with inference timestamps.
"""

import time

import torch
from ultralytics import YOLO

from config import (
    YOLO_MODEL,
    YOLO_DEVICE,
    YOLO_CONF_THRESHOLD,
    YOLO_PERSON_CLASS,
    YOLO_HALF,
    YOLO_TRACKER,
)
from pipeline.models import FrameMeta, DetectionResult
from utils.logger import get_logger

logger = get_logger(__name__)


class PersonDetector:
    """
    Wraps YOLOv8n for person-only detection with ByteTrack tracking.

    * Prefers CUDA with FP16 but falls back to CPU automatically.
    * ``detect()`` returns a list of ``DetectionResult`` dataclasses,
      each carrying the originating ``FrameMeta`` plus inference
      start/end timestamps (monotonic).
    * Empty / failed frames return ``[]`` — never raises.
    """

    def __init__(self):
        # Resolve actual device — fall back to CPU if CUDA is unavailable
        if YOLO_DEVICE == "cuda" and not torch.cuda.is_available():
            self._device = "cpu"
            self._half = False  # FP16 not supported on CPU
            logger.warning(
                "CUDA requested but not available. "
                "Falling back to CPU (FP16 disabled). "
                "Install CUDA-enabled PyTorch for GPU acceleration."
            )
        else:
            self._device = YOLO_DEVICE
            self._half = YOLO_HALF

        logger.info(f"Loading YOLO model: {YOLO_MODEL} on device={self._device}")
        self.model = YOLO(YOLO_MODEL)
        self.model.to(self._device)

        if self._half:
            logger.info("FP16 (half-precision) inference enabled")
        logger.info("PersonDetector initialised")

    # ─────────────────────────────────────────────────
    def detect(self, frame, meta: FrameMeta = None) -> list[DetectionResult]:
        """
        Run YOLO detection + ByteTrack on *frame*.

        Parameters
        ----------
        frame : np.ndarray
            BGR image.
        meta : FrameMeta, optional
            Temporal metadata for this frame.

        Returns
        -------
        list[DetectionResult]
            One entry per tracked person with bbox, confidence,
            track_id, and inference timing.
        """
        detect_start = time.perf_counter()

        try:
            results = self.model.track(
                frame,
                persist=True,
                tracker=YOLO_TRACKER,
                classes=[YOLO_PERSON_CLASS],
                conf=YOLO_CONF_THRESHOLD,
                half=self._half,
                verbose=False,
            )

            detect_end = time.perf_counter()
            detections: list[DetectionResult] = []

            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    bbox = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0].cpu().numpy())

                    # ByteTrack may not assign an ID on the very first frame
                    if box.id is None:
                        continue
                    track_id = int(box.id[0].cpu().numpy())

                    detections.append(
                        DetectionResult(
                            bbox=bbox,
                            confidence=confidence,
                            track_id=track_id,
                            meta=meta,
                            detect_start_ts=detect_start,
                            detect_end_ts=detect_end,
                        )
                    )

            return detections

        except Exception as e:
            logger.warning(f"Detection error: {e}")
            return []
