"""
PROJECT 007 — Events & Rules Data Models
P1: Proxy Rule Outputs, Behavior Scores, and Event Candidates.
"""

from dataclasses import dataclass, field
import uuid


@dataclass
class RuleEvent:
    """Output from a single deterministic rule evaluation."""
    rule_type: str
    confidence: float
    uncertainty: float
    track_ids: frozenset[int]
    frame_id: int
    timestamp: float


@dataclass
class BehaviorScore:
    """Output from the BehaviorScoringEngine."""
    score: float
    timestamp: float


@dataclass
class EventCandidate:
    """
    Candidate anomalous interaction (HIGH-ENERGY INTERACTION).
    This is NOT a confirmed threat.
    """
    track_ids: list[int]
    active_rules: list[str]
    behavior_score: float
    confidence: float
    uncertainty: float
    first_seen_ts: float
    latest_seen_ts: float
    frame_id: int
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class SceneInteraction:
    """
    P4.7: Spatially and temporally continuous event footprint.
    Persists risk and state across track ID changes.
    """
    interaction_id: str
    centroid: tuple[float, float]
    radius: float
    first_seen_ts: float
    last_seen_ts: float
    state_enter_ts: float
    risk_score: float
    state: str  # NORMAL, SUSPICIOUS, HIGH_RISK, CRITICAL
    contributing_tracks: set[int]
    active_rules: list[str]
