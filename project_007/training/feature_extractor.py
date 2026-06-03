"""
PROJECT 007 — P4.0 Feature Extractor
Extracts per-frame feature vectors from the pipeline's motion, tracking,
scene, and risk data.
"""

import numpy as np
from collections import deque
from typing import Optional

from training.feature_schema import (
    ALL_FEATURES,
    MOTION_FEATURES,
    TRACKING_FEATURES,
    SCENE_FEATURES,
    RISK_FEATURES,
    PAIRWISE_FEATURES,
    TEMPORAL_FEATURES,
)
from utils.logger import get_logger

logger = get_logger(__name__)

TEMPORAL_WINDOW = 16  # Frames to aggregate for temporal features


class FeatureExtractor:
    """
    Extracts a fixed-length feature vector from a single frame's pipeline outputs.
    Also maintains a rolling window for temporal aggregation.
    """

    def __init__(self, window_size: int = TEMPORAL_WINDOW):
        self.window_size = window_size
        self._history = deque(maxlen=window_size)

    def reset(self):
        """Clear the temporal window."""
        self._history.clear()

    def extract(
        self,
        all_motion: dict,
        pairwise_motion: dict,
        flow_metrics: dict,
        scene_stability: float,
        occupancy_ratio: float,
        fused_evidence: list,
        raw_rules: list,
        frame: Optional[np.ndarray] = None,
        detections: Optional[list] = None,
    ) -> dict:
        """
        Extract a feature dict from the current frame's pipeline state.

        Returns
        -------
        dict : Feature name -> float value (all keys from ALL_FEATURES).
        """
        features = {}

        # ── Motion Features ──
        if all_motion:
            arm_vels = [m.get("arm_velocity", 0.0) for m in all_motion.values()]
            body_disps = [m.get("body_displacement", 0.0) for m in all_motion.values()]
            fall_scores = [m.get("fall_score", 0.0) for m in all_motion.values()]
            fall_deltas = [m.get("fall_score_delta", 0.0) for m in all_motion.values()]
            uncertainties = [m.get("uncertainty", 1.0) for m in all_motion.values()]

            features["arm_velocity"] = max(arm_vels) if arm_vels else 0.0
            features["body_displacement"] = max(body_disps) if body_disps else 0.0
            features["fall_score"] = max(fall_scores) if fall_scores else 0.0
            features["fall_score_delta"] = max(fall_deltas) if fall_deltas else 0.0
            features["uncertainty"] = min(uncertainties) if uncertainties else 1.0
        else:
            for f in MOTION_FEATURES:
                features[f] = 0.0
            features["uncertainty"] = 1.0

        # ── Tracking Features ──
        track_count = len(all_motion) if all_motion else 0
        features["track_count"] = float(track_count)

        nearest_dist = 999.0
        for pw in pairwise_motion.values():
            d = pw.get("distance", 999.0)
            if d < nearest_dist:
                nearest_dist = d
        features["nearest_person_distance"] = nearest_dist if nearest_dist < 999.0 else 0.0

        # Overlap duration: count fused evidence entries with time_in_state > 0
        overlap_dur = 0.0
        for fe in fused_evidence:
            if hasattr(fe, "time_in_state"):
                overlap_dur = max(overlap_dur, fe.time_in_state)
            elif isinstance(fe, dict):
                overlap_dur = max(overlap_dur, fe.get("time_in_state", 0.0))
        features["overlap_duration"] = overlap_dur

        # ── Scene Features ──
        features["optical_flow_magnitude"] = flow_metrics.get("avg_flow_mag", 0.0)
        features["optical_flow_instability"] = flow_metrics.get("instability_score", 0.0)

        brightness = 0.5  # default
        if frame is not None:
            gray = frame.mean() / 255.0 if frame.ndim == 3 else frame.mean() / 255.0
            brightness = float(gray)
        features["frame_brightness"] = brightness

        features["occlusion_ratio"] = occupancy_ratio
        features["scene_stability"] = scene_stability

        # ── Risk Features ──
        features["active_rules_count"] = float(len(raw_rules))

        rule_conf_sum = 0.0
        for r in raw_rules:
            if hasattr(r, "confidence"):
                rule_conf_sum += r.confidence
        features["rule_confidence_sum"] = rule_conf_sum

        max_risk = 0.0
        for fe in fused_evidence:
            score = fe.evidence_score if hasattr(fe, "evidence_score") else (
                fe.get("evidence_score", 0.0) if isinstance(fe, dict) else 0.0
            )
            max_risk = max(max_risk, score)
        features["current_risk_score"] = max_risk

        # ── Pairwise Features ──
        min_pair_dist = 0.0
        max_approach_vel = 0.0
        if pairwise_motion:
            dists = [pw.get("distance", 999.0) for pw in pairwise_motion.values()]
            approach_vels = [pw.get("approach_velocity", 0.0) for pw in pairwise_motion.values()]
            min_pair_dist = min(dists) if dists else 0.0
            max_approach_vel = max(approach_vels) if approach_vels else 0.0
        features["min_pair_distance"] = min_pair_dist
        features["max_approach_velocity"] = max_approach_vel

        # ── Temporal Features (rolling window) ──
        self._history.append(features.copy())
        temporal = self._compute_temporal()
        features.update(temporal)

        return features

    def _compute_temporal(self) -> dict:
        """Compute temporal aggregation features from the rolling window."""
        if len(self._history) < 2:
            return {f: 0.0 for f in TEMPORAL_FEATURES}

        arm_vels = [h["arm_velocity"] for h in self._history]
        body_disps = [h["body_displacement"] for h in self._history]
        fall_scores = [h["fall_score"] for h in self._history]
        risk_scores = [h["current_risk_score"] for h in self._history]
        rule_counts = [h["active_rules_count"] for h in self._history]

        return {
            "arm_velocity_mean": float(np.mean(arm_vels)),
            "arm_velocity_max": float(np.max(arm_vels)),
            "arm_velocity_std": float(np.std(arm_vels)),
            "body_displacement_mean": float(np.mean(body_disps)),
            "body_displacement_max": float(np.max(body_disps)),
            "fall_score_mean": float(np.mean(fall_scores)),
            "fall_score_max": float(np.max(fall_scores)),
            "risk_score_mean": float(np.mean(risk_scores)),
            "risk_score_max": float(np.max(risk_scores)),
            "rule_count_sum": float(np.sum(rule_counts)),
        }

    @staticmethod
    def feature_names() -> list:
        """Return ordered list of feature names."""
        return list(ALL_FEATURES)
