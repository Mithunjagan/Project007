"""
PROJECT 007 — Replay Integrity Validator
Verifies deterministic replay by comparing two runs on the same video.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List

from utils.logger import get_logger

logger = get_logger(__name__)


class ReplayIntegrityValidator:
    """
    Validates that two sync-mode replay runs on the same video
    produce identical results.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.event_count = 0
        self.state_transitions = []
        self.confusion_counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        self.frame_states = []

    def record_frame(self, frame_id: int, state: str, evidence_score: float):
        self.frame_states.append({
            "frame_id": frame_id,
            "state": state,
            "evidence_score": round(evidence_score, 6)
        })

    def record_event(self):
        self.event_count += 1

    def record_state_transition(self, frame_id: int, from_state: str, to_state: str):
        self.state_transitions.append({
            "frame_id": frame_id,
            "from": from_state,
            "to": to_state
        })

    def record_confusion(self, tp: int, fp: int, tn: int, fn: int):
        self.confusion_counts = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

    def compute_checksum(self) -> str:
        """Produce a deterministic hash of the entire run."""
        payload = {
            "event_count": self.event_count,
            "state_transitions": self.state_transitions,
            "confusion_counts": self.confusion_counts,
            "frame_state_count": len(self.frame_states),
            # Hash states in blocks to keep payload manageable
            "frame_state_hash": hashlib.sha256(
                json.dumps(self.frame_states, sort_keys=True).encode()
            ).hexdigest()
        }
        raw = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> Dict:
        return {
            "event_count": self.event_count,
            "state_transitions": self.state_transitions,
            "confusion_counts": self.confusion_counts,
            "checksum": self.compute_checksum()
        }


def validate_replays(run1: Dict, run2: Dict) -> bool:
    """
    Compare two replay run summaries. Returns True if deterministic.
    """
    if run1["checksum"] != run2["checksum"]:
        logger.error(
            f"DETERMINISM VIOLATION: checksums differ!\n"
            f"  Run 1: {run1['checksum']}\n"
            f"  Run 2: {run2['checksum']}"
        )
        if run1["event_count"] != run2["event_count"]:
            logger.error(f"  Event counts: {run1['event_count']} vs {run2['event_count']}")
        if run1["confusion_counts"] != run2["confusion_counts"]:
            logger.error(f"  Confusion: {run1['confusion_counts']} vs {run2['confusion_counts']}")
        return False

    logger.info("Replay integrity validated: DETERMINISTIC ✓")
    return True
