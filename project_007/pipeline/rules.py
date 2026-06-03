"""
PROJECT 007 — Proxy Rule Engine
Evaluates deterministic behavioral rules based on motion features.
"""

import numpy as np

from config import (
    APPROACH_VELOCITY_THRESHOLD,
    ARM_SWING_THRESHOLD,
    FALL_SCORE_THRESHOLD,
    CONTACT_DISTANCE_THRESHOLD,
    CROWD_DISPERSION_THRESHOLD,
    DIRECTION_THRESHOLD,
)
from pipeline.events import RuleEvent
from utils.logger import get_logger

logger = get_logger(__name__)


class ProxyRuleEngine:
    """
    Evaluates proxy interaction dynamics without relying on ML classifiers.
    """

    def evaluate(
        self, all_motion: dict, pairwise_motion: dict, scene_stability: float, frame_id: int, timestamp: float
    ) -> list[RuleEvent]:
        """
        Evaluate all deterministic rules on the current frame's motion state.
        
        Parameters
        ----------
        all_motion : dict
            {track_id: {"arm_velocity": ..., "arm_motion_vector": ..., "fall_score": ..., "fall_score_delta": ..., "uncertainty": ...}}
        pairwise_motion : dict
            {(id_a, id_b): {"distance": ..., "approach_velocity": ..., "target_vec_a2b": ..., "target_vec_b2a": ...}}
        scene_stability : float
            Current scene stability [0.1 - 1.0]. Lower values increase thresholds.
        frame_id : int
        timestamp : float
        
        Returns
        -------
        list[RuleEvent]
            Active rule triggers for this frame.
        """
        events = []
        
        # Adaptive Threshold Scaling: Instability linearly increases thresholds by up to 50%
        stability_factor = 1.0 + (1.0 - scene_stability) * 0.5
        
        adaptive_arm_swing = ARM_SWING_THRESHOLD * stability_factor
        adaptive_approach = APPROACH_VELOCITY_THRESHOLD * stability_factor

        # Evaluate per-track rules
        for track_id, motion in all_motion.items():
            # 3. FALL_EVENT
            if motion["fall_score"] > FALL_SCORE_THRESHOLD and motion["fall_score_delta"] > 0.15:
                # Confidence proportional to fall score and delta
                conf = min(1.0, motion["fall_score"] * (1.0 + motion["fall_score_delta"]))
                events.append(
                    RuleEvent(
                        rule_type="FALL_EVENT",
                        confidence=float(conf),
                        uncertainty=motion["uncertainty"],
                        track_ids=frozenset([track_id]),
                        frame_id=frame_id,
                        timestamp=timestamp,
                    )
                )

        # Evaluate pairwise rules
        for (id_a, id_b), p_motion in pairwise_motion.items():
            # Only evaluate if both tracks have valid motion
            if id_a not in all_motion or id_b not in all_motion:
                continue

            dist = p_motion["distance"]
            app_vel = p_motion["approach_velocity"]
            unc_a = all_motion[id_a]["uncertainty"]
            unc_b = all_motion[id_b]["uncertainty"]
            joint_uncertainty = (unc_a + unc_b) / 2.0

            # 1. RAPID_APPROACH
            if app_vel > adaptive_approach:
                conf = min(1.0, app_vel / (adaptive_approach * 2.0))
                events.append(
                    RuleEvent(
                        rule_type="RAPID_APPROACH",
                        confidence=float(conf),
                        uncertainty=joint_uncertainty,
                        track_ids=frozenset([id_a, id_b]),
                        frame_id=frame_id,
                        timestamp=timestamp,
                    )
                )

            # 2. DIRECTED_ARM_SWING
            # Check A swinging towards B
            arm_vel_a = all_motion[id_a]["arm_velocity"]
            if dist < CONTACT_DISTANCE_THRESHOLD and arm_vel_a > adaptive_arm_swing:
                arm_vec_a = all_motion[id_a]["arm_motion_vector"]
                target_vec = p_motion["target_vec_a2b"]
                dot_prod = self._normalized_dot(arm_vec_a, target_vec)
                
                if dot_prod > DIRECTION_THRESHOLD:
                    conf = min(1.0, (arm_vel_a / adaptive_arm_swing) * dot_prod)
                    events.append(
                        RuleEvent(
                            rule_type="DIRECTED_ARM_SWING",
                            confidence=float(conf),
                            uncertainty=unc_a,
                            track_ids=frozenset([id_a, id_b]),
                            frame_id=frame_id,
                            timestamp=timestamp,
                        )
                    )

            # Check B swinging towards A
            arm_vel_b = all_motion[id_b]["arm_velocity"]
            if dist < CONTACT_DISTANCE_THRESHOLD and arm_vel_b > adaptive_arm_swing:
                arm_vec_b = all_motion[id_b]["arm_motion_vector"]
                target_vec = p_motion["target_vec_b2a"]
                dot_prod = self._normalized_dot(arm_vec_b, target_vec)
                
                if dot_prod > DIRECTION_THRESHOLD:
                    conf = min(1.0, (arm_vel_b / adaptive_arm_swing) * dot_prod)
                    events.append(
                        RuleEvent(
                            rule_type="DIRECTED_ARM_SWING",
                            confidence=float(conf),
                            uncertainty=unc_b,
                            track_ids=frozenset([id_b, id_a]),
                            frame_id=frame_id,
                            timestamp=timestamp,
                        )
                    )

            # 4. SUSTAINED_CONTACT (Single frame indicator, persistence filter will accumulate)
            if dist < CONTACT_DISTANCE_THRESHOLD and (arm_vel_a > 1.0 or arm_vel_b > 1.0):
                conf = min(1.0, 1.0 - (dist / CONTACT_DISTANCE_THRESHOLD))
                events.append(
                    RuleEvent(
                        rule_type="SUSTAINED_CONTACT",
                        confidence=float(conf),
                        uncertainty=joint_uncertainty,
                        track_ids=frozenset([id_a, id_b]),
                        frame_id=frame_id,
                        timestamp=timestamp,
                    )
                )

        # 5. CROWD_DISPERSION
        # Detect multiple subjects accelerating outward from a centroid
        if len(all_motion) >= 3:
            # Find dispersion center (crude average of active tracks)
            # We don't have centers in all_motion directly, but we can compute from pairwise or assume if dispersion is high.
            # A simpler dispersion proxy: check if most pairwise distances are increasing rapidly.
            separating_pairs = 0
            total_pairs = len(pairwise_motion)
            if total_pairs > 0:
                for pm in pairwise_motion.values():
                    if pm["approach_velocity"] < -CROWD_DISPERSION_THRESHOLD:
                        separating_pairs += 1
                
                if separating_pairs >= 3 and (separating_pairs / total_pairs) > 0.5:
                    events.append(
                        RuleEvent(
                            rule_type="CROWD_DISPERSION",
                            confidence=float(separating_pairs / total_pairs),
                            uncertainty=0.1,  # Generally confident if happening globally
                            track_ids=frozenset(all_motion.keys()),
                            frame_id=frame_id,
                            timestamp=timestamp,
                        )
                    )

        return events

    @staticmethod
    def _normalized_dot(vec1: tuple[float, float], vec2: tuple[float, float]) -> float:
        """Returns the dot product of two vectors after normalizing them."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        
        if n1 < 1e-5 or n2 < 1e-5:
            return 0.0
            
        v1_norm = v1 / n1
        v2_norm = v2 / n2
        return float(np.dot(v1_norm, v2_norm))
