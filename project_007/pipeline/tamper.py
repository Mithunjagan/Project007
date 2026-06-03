"""
PROJECT 007 — Camera Tamper Engine
Detects physical interference: occlusion, blockage, and camera shake.
"""

import cv2
import numpy as np

from config import TAMPER_DARK_PIXEL_RATIO, TAMPER_SHAKE_MAGNITUDE
from pipeline.events import RuleEvent
from utils.logger import get_logger

logger = get_logger(__name__)


class CameraTamperEngine:
    """
    Evaluates frame-level statistics and optical flow to detect tampering.
    """

    def evaluate(self, frame: np.ndarray, detections: list, flow_metrics: dict, frame_id: int, timestamp: float) -> list[RuleEvent]:
        events = []
        h, w = frame.shape[:2]

        # 1. LENS_OCCLUSION (Sudden darkness / coverage)
        # Convert to grayscale and check dark pixels
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dark_pixels = cv2.countNonZero(cv2.inRange(gray, 0, 30))
        dark_ratio = float(dark_pixels) / (w * h)

        if dark_ratio > TAMPER_DARK_PIXEL_RATIO:
            events.append(
                RuleEvent(
                    rule_type="LENS_OCCLUSION",
                    confidence=min(1.0, dark_ratio),
                    uncertainty=0.1,
                    track_ids=frozenset(),
                    frame_id=frame_id,
                    timestamp=timestamp
                )
            )

        # 2. CAMERA_BLOCKAGE (Central obstruction)
        # Check if any detection dominates a large center region of the frame
        center_w_min, center_w_max = w * 0.15, w * 0.85
        center_h_min, center_h_max = h * 0.15, h * 0.85
        center_area = (center_w_max - center_w_min) * (center_h_max - center_h_min)

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            # Calculate intersection with center region
            ix1 = max(center_w_min, x1)
            iy1 = max(center_h_min, y1)
            ix2 = min(center_w_max, x2)
            iy2 = min(center_h_max, y2)

            if ix1 < ix2 and iy1 < iy2:
                intersect_area = (ix2 - ix1) * (iy2 - iy1)
                obstruction_ratio = intersect_area / center_area

                if obstruction_ratio > 0.95:  # 95% of the large center region covered
                    events.append(
                        RuleEvent(
                            rule_type="CAMERA_BLOCKAGE",
                            confidence=min(1.0, obstruction_ratio),
                            uncertainty=0.1,
                            track_ids=frozenset([det.track_id]),
                            frame_id=frame_id,
                            timestamp=timestamp
                        )
                    )

        # 3. CAMERA_SHAKE (High global optical flow)
        flow_mag = flow_metrics.get("avg_flow_mag", 0.0)
        flow_instability = flow_metrics.get("instability_score", 0.0)

        # High magnitude AND low instability means the whole frame moved uniformly (camera moved)
        # High magnitude AND high instability means chaotic movement
        if flow_mag > TAMPER_SHAKE_MAGNITUDE:
            events.append(
                RuleEvent(
                    rule_type="CAMERA_SHAKE",
                    confidence=min(1.0, flow_mag / (TAMPER_SHAKE_MAGNITUDE * 2.0)),
                    uncertainty=min(1.0, flow_instability),
                    track_ids=frozenset(),
                    frame_id=frame_id,
                    timestamp=timestamp
                )
            )

        return events
