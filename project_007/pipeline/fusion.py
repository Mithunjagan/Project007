"""
PROJECT 007 — Contextual Threat Fusion Layer
Replaces instantaneous rule triggering with contextual multi-signal evidence fusion,
hysteresis state machines, and temporal accumulation.
"""

from dataclasses import dataclass
from typing import Optional

from pipeline.events import RuleEvent, SceneInteraction
from config import RISK_DECAY, MAX_RISK_SCORE
import uuid
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FusedEvidence:
    track_ids: frozenset
    evidence_score: float
    state: str  # NORMAL, SUSPICIOUS, HIGH_RISK, CRITICAL
    contributing_rules: list[str]
    time_in_state: float


class ConfidenceSuppressionLayer:
    """
    Suppresses rule confidence globally if tracking is unstable, lighting is poor,
    or other contextual invalidations occur.
    """
    def apply(self, events: list[RuleEvent], scene_stability: float, person_count: int) -> list[RuleEvent]:
        valid_events = []
        for e in events:
            # If scene stability is very bad, scale down confidence
            scaled_conf = e.confidence * scene_stability
            
            # Suppress interpersonal rules if only one person is detected
            interpersonal = ["DIRECTED_ARM_SWING", "RAPID_APPROACH", "SUSTAINED_CONTACT"]
            if e.rule_type in interpersonal and person_count < 2:
                scaled_conf = 0.0

            if scaled_conf > 0.1:  # Noise floor
                e.confidence = scaled_conf
                valid_events.append(e)
                
        return valid_events


class ContextGates:
    """
    Applies logic gates to specific rules before they enter the accumulator.
    """
    def filter(self, events: list[RuleEvent], all_motion: dict, pairwise_motion: dict, flow_metrics: dict) -> list[RuleEvent]:
        gated = []
        for e in events:
            if e.rule_type == "DIRECTED_ARM_SWING":
                # Only pass if target distance is actively decreasing
                ids = list(e.track_ids)
                if len(ids) == 2:
                    p_motion = pairwise_motion.get((ids[0], ids[1])) or pairwise_motion.get((ids[1], ids[0]))
                    if p_motion and p_motion["approach_velocity"] > 0:
                        gated.append(e)
            
            elif e.rule_type == "CAMERA_SHAKE":
                # Pass only if flow coherence (inverse of instability) is high enough
                instability = flow_metrics.get("instability_score", 0.0)
                if instability < 2.0:  # Fairly coherent global motion, not just noise
                    gated.append(e)
            
            else:
                # Other rules pass through freely for now
                gated.append(e)

        return gated


class InteractionManager:
    """
    P4.7: Manages SceneInteraction objects based on spatial and temporal proximity
    to preserve evidence across track ID fragmentation.
    """
    def __init__(self, weights: dict):
        self._weights = weights
        self._interactions: list[SceneInteraction] = []
        self._spatial_threshold = 200.0  # pixels
        self._temporal_threshold = 2.0   # seconds
        
        self.stats = {
            "interaction_merges": 0,
            "interaction_expirations": 0,
            "interaction_lifetimes": [],
            "track_churn_events": 0
        }

    def _get_group_centroid(self, track_ids, track_buffer) -> tuple[float, float]:
        centers = []
        for tid in track_ids:
            hist = track_buffer.get_history(tid)
            if hist:
                centers.append(hist[-1]["center"])
        if not centers:
            return None
        return (sum(c[0] for c in centers) / len(centers), sum(c[1] for c in centers) / len(centers))

    def update(self, events: list[RuleEvent], track_buffer, timestamp: float) -> list[SceneInteraction]:
        # Expire old global interaction if not updated recently
        if self._interactions:
            interaction = self._interactions[0]
            if (timestamp - interaction.last_seen_ts) > self._temporal_threshold:
                self.stats["interaction_expirations"] += 1
                self.stats["interaction_lifetimes"].append(timestamp - interaction.first_seen_ts)
                self._interactions = []

        if events:
            # Gather all track IDs and rules active in this frame
            all_tracks = set()
            for e in events:
                all_tracks.update(e.track_ids)
            
            active_rules = list(set([e.rule_type for e in events]))
            instant_score = sum(self._weights.get(e.rule_type, 0.0) * e.confidence for e in events)

            group_centroid = self._get_group_centroid(all_tracks, track_buffer)
            if not group_centroid:
                group_centroid = (200.0, 150.0)  # Default center fallback

            if self._interactions:
                interaction = self._interactions[0]
                self.stats["interaction_merges"] += 1
                
                interaction.last_seen_ts = timestamp
                alpha_c = 0.5
                interaction.centroid = (
                    alpha_c * group_centroid[0] + (1 - alpha_c) * interaction.centroid[0],
                    alpha_c * group_centroid[1] + (1 - alpha_c) * interaction.centroid[1]
                )
                
                # Check for track churn
                new_tracks = all_tracks - interaction.contributing_tracks
                if new_tracks:
                    self.stats["track_churn_events"] += 1
                
                interaction.contributing_tracks.update(all_tracks)
                interaction.active_rules = list(set(interaction.active_rules + active_rules))
                
                new_score = (interaction.risk_score * RISK_DECAY) + instant_score
                interaction.risk_score = min(MAX_RISK_SCORE, new_score)
            else:
                # Spawn new interaction representing the global scene state
                new_interaction = SceneInteraction(
                    interaction_id="global_scene",
                    centroid=group_centroid,
                    radius=self._spatial_threshold,
                    first_seen_ts=timestamp,
                    last_seen_ts=timestamp,
                    state_enter_ts=timestamp,
                    risk_score=min(MAX_RISK_SCORE, instant_score),
                    state="NORMAL",
                    contributing_tracks=all_tracks,
                    active_rules=active_rules
                )
                self._interactions.append(new_interaction)
        else:
            # If no rules triggered, decay the active interaction's risk score
            if self._interactions:
                interaction = self._interactions[0]
                interaction.risk_score *= RISK_DECAY

        # Evaluate State Machine for the active interaction
        if self._interactions:
            interaction = self._interactions[0]
            current_state = interaction.state
            dwell_time = timestamp - interaction.state_enter_ts
            score = interaction.risk_score

            new_state = current_state

            if current_state == "NORMAL":
                if score > 0.35 and dwell_time > 2.0:
                    new_state = "SUSPICIOUS"
            elif current_state == "SUSPICIOUS":
                if score < 0.20 and dwell_time > 3.0:
                    new_state = "NORMAL"
                elif score > 0.60 and dwell_time > 2.0:
                    new_state = "HIGH_RISK"
            elif current_state == "HIGH_RISK":
                if score < 0.35 and dwell_time > 5.0:
                    new_state = "SUSPICIOUS"
                elif score > 0.85 and dwell_time > 2.0 and len(interaction.active_rules) >= 3:
                    new_state = "CRITICAL"
            elif current_state == "CRITICAL":
                if score < 0.50 and dwell_time > 5.0:
                    new_state = "HIGH_RISK"

            if new_state != current_state:
                interaction.state = new_state
                interaction.state_enter_ts = timestamp

        return self._interactions


class FusionEngine:
    """
    Facade combining Suppression, Context, and Scene-Level Evidence Persistence.
    """
    def __init__(self, weights: dict):
        self._suppression = ConfidenceSuppressionLayer()
        self._gates = ContextGates()
        self.interaction_manager = InteractionManager(weights)

    def update(
        self, raw_events: list[RuleEvent], scene_stability: float, person_count: int,
        all_motion: dict, pairwise_motion: dict, flow_metrics: dict, track_buffer, timestamp: float,
        dl_prediction: dict = None
    ) -> list[FusedEvidence]:
        
        # 1. Suppression
        events = self._suppression.apply(raw_events, scene_stability, person_count)
        
        # 2. Context Gates
        events = self._gates.filter(events, all_motion, pairwise_motion, flow_metrics)

        # 3. Scene-Level Interaction Management
        interactions = self.interaction_manager.update(events, track_buffer, timestamp)

        # 4. Apply DL prediction boosting if available
        if dl_prediction and dl_prediction.get("confidence", 0) > 0.3:
            dl_state = dl_prediction.get("state", "NORMAL")
            dl_conf = dl_prediction["confidence"]
            state_scores = {"NORMAL": 0.0, "SUSPICIOUS": 0.4, "HIGH_RISK": 0.7, "CRITICAL": 1.0}
            dl_risk = state_scores.get(dl_state, 0.0) * dl_conf

            for interaction in interactions:
                # Blend DL risk into existing risk score
                from config import DL_FUSION_WEIGHT, RULE_FUSION_WEIGHT
                blended = (RULE_FUSION_WEIGHT * interaction.risk_score) + (DL_FUSION_WEIGHT * dl_risk)
                interaction.risk_score = min(MAX_RISK_SCORE, blended)

        # 5. Map back to FusedEvidence for backward compatibility
        fused = []
        for i in interactions:
            fused.append(FusedEvidence(
                track_ids=frozenset(i.contributing_tracks),
                evidence_score=i.risk_score,
                state=i.state,
                contributing_rules=i.active_rules,
                time_in_state=timestamp - i.state_enter_ts
            ))

        return fused
