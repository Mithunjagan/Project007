"""
PROJECT 007 — Intrusion Engine
Detects aggressive approach toward the camera and single-subject anomalies.
"""

from config import INTRUSION_AREA_GROWTH_RATE, INTRUSION_MAX_OCCUPANCY
from pipeline.events import RuleEvent
from utils.logger import get_logger

logger = get_logger(__name__)


class IntrusionEngine:
    """
    Evaluates geometric growth of subjects to detect camera-directed aggression.
    """

    def evaluate(self, detections: list, all_motion: dict, track_buffer, frame_width: int, frame_height: int, frame_id: int, timestamp: float) -> list[RuleEvent]:
        events = []
        frame_area = max(1, frame_width * frame_height)

        for det in detections:
            tid = det.track_id
            motion = all_motion.get(tid, {})
            history = track_buffer.get_history(tid)

            if len(history) < 2:
                continue

            current = history[-1]
            previous = history[-2]

            # Calculate BBox areas
            curr_w = current["bbox"][2] - current["bbox"][0]
            curr_h = current["bbox"][3] - current["bbox"][1]
            curr_area = max(1, curr_w * curr_h)

            prev_w = previous["bbox"][2] - previous["bbox"][0]
            prev_h = previous["bbox"][3] - previous["bbox"][1]
            prev_area = max(1, prev_w * prev_h)

            dt = current["timestamp"] - previous["timestamp"]
            if dt <= 0:
                dt = 0.04

            # 1. CAMERA_RUSH (Rapid BBox Growth)
            area_growth_ratio = curr_area / float(prev_area)
            # Subtract 1 so no-growth = 0, then normalize to per-second growth rate
            normalized_growth_rate = (area_growth_ratio - 1.0) / dt

            if normalized_growth_rate > INTRUSION_AREA_GROWTH_RATE:
                events.append(
                    RuleEvent(
                        rule_type="CAMERA_RUSH",
                        confidence=min(1.0, normalized_growth_rate / (INTRUSION_AREA_GROWTH_RATE * 2.0)),
                        uncertainty=motion.get("uncertainty", 0.5),
                        track_ids=frozenset([tid]),
                        frame_id=frame_id,
                        timestamp=timestamp
                    )
                )

            # 2. PROXIMITY_INTRUSION (Excessive Frame Occupancy)
            occupancy = curr_area / float(frame_area)
            if occupancy > INTRUSION_MAX_OCCUPANCY and motion.get("body_displacement", 0.0) > 0.5:
                events.append(
                    RuleEvent(
                        rule_type="PROXIMITY_INTRUSION",
                        confidence=min(1.0, occupancy),
                        uncertainty=motion.get("uncertainty", 0.5),
                        track_ids=frozenset([tid]),
                        frame_id=frame_id,
                        timestamp=timestamp
                    )
                )

            # 3. ABNORMAL_SINGLE_SUBJECT_ENERGY (High energy, no target)
            # If they are the only person (or very isolated) and have huge arm/body motion
            if len(detections) == 1:
                arm_vel = motion.get("arm_velocity", 0.0)
                if arm_vel > 4.0:  # Higher threshold since it's just one person
                    events.append(
                        RuleEvent(
                            rule_type="ABNORMAL_SINGLE_SUBJECT_ENERGY",
                            confidence=min(1.0, arm_vel / 8.0),
                            uncertainty=motion.get("uncertainty", 0.5),
                            track_ids=frozenset([tid]),
                            frame_id=frame_id,
                            timestamp=timestamp
                        )
                    )

        return events
