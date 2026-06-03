"""
PROJECT 007 — Scene Dynamics Engine
Evaluates scene-level anomaly metrics.
"""

from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SceneMetrics:
    occupancy_ratio: float
    crowd_density: int
    scene_motion_energy: float
    scene_stability_score: float  # [0.0 - 1.0], 1.0 is perfectly stable


class SceneDynamicsEngine:
    """
    Computes global scene-level metrics like occupancy and total motion energy.
    """
    
    def compute(self, detections: list, all_motion: dict, flow_metrics: dict, frame_width: int, frame_height: int) -> SceneMetrics:
        """
        Evaluate scene metrics for the current frame.
        """
        frame_area = max(1, frame_width * frame_height)
        
        total_bbox_area = 0
        scene_motion_energy = 0.0
        avg_uncertainty = 0.0
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            total_bbox_area += (w * h)
            
            motion = all_motion.get(det.track_id)
            if motion:
                scene_motion_energy += motion.get("body_displacement", 0.0)
                avg_uncertainty += motion.get("uncertainty", 0.0)

        if detections:
            avg_uncertainty /= len(detections)

        # Normalize occupancy (can exceed 1.0 if overlapping heavily, so we clamp)
        occupancy_ratio = min(1.0, float(total_bbox_area) / frame_area)
        
        # Stability based on flow instability and tracking uncertainty
        flow_instab = flow_metrics.get("instability_score", 0.0)
        # Normalize flow_instab to [0-1] roughly (e.g. instab of 5 is very bad)
        normalized_flow_instab = min(1.0, flow_instab / 5.0)
        
        stability_score = 1.0 - max(normalized_flow_instab, avg_uncertainty)
        stability_score = max(0.1, min(1.0, stability_score))  # 0.1 min bound
        
        return SceneMetrics(
            occupancy_ratio=occupancy_ratio,
            crowd_density=len(detections),
            scene_motion_energy=scene_motion_energy,
            scene_stability_score=float(stability_score)
        )
