"""
PROJECT 007 — Behavior Scoring Engine
Calculates a continuous risk score based on active proxy rules.
"""

from config import RISK_DECAY, MAX_RISK_SCORE
from pipeline.events import RuleEvent, BehaviorScore


class BehaviorScoringEngine:
    """
    Combines weighted rule outputs into a continuous intensity score.
    Never binary flips instantly.
    """

    def __init__(self):
        self._current_score = 0.0
        
        self._weights = {
            # P1 Interpersonal Rules
            "DIRECTED_ARM_SWING": 0.35,
            "RAPID_APPROACH": 0.30,
            "FALL_EVENT": 0.20,
            "SUSTAINED_CONTACT": 0.10,
            "CROWD_DISPERSION": 0.05,
            
            # P1.5 Camera Threat Rules
            "CAMERA_SHAKE": 0.40,
            "LENS_OCCLUSION": 0.50,
            "CAMERA_BLOCKAGE": 0.30,
            "CAMERA_RUSH": 0.40,
            "PROXIMITY_INTRUSION": 0.35,
            "ABNORMAL_SINGLE_SUBJECT_ENERGY": 0.30,
        }

    def update(self, active_rules: list[RuleEvent], timestamp: float) -> BehaviorScore:
        """
        Calculates the new behavior score based on active rules and decays the old score.
        """
        # Calculate instant score from current active rules
        instant_score = 0.0
        for rule in active_rules:
            weight = self._weights.get(rule.rule_type, 0.0)
            # Factor in confidence
            instant_score += weight * rule.confidence

        # Combine with decayed previous score
        self._current_score = (self._current_score * RISK_DECAY) + instant_score

        # Clamp
        self._current_score = max(0.0, min(MAX_RISK_SCORE, self._current_score))

        return BehaviorScore(
            score=float(self._current_score),
            timestamp=timestamp
        )
