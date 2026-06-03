"""
PROJECT 007 — P2.5 Metrics Engine
Computes precision, recall, rule contributions, and false-positive attribution.
"""

from collections import defaultdict
from typing import Dict


class MetricsEngine:
    def __init__(self):
        self.reset()

    def reset(self):
        self.frame_results = []
        self.rule_contributions = defaultdict(float)
        self.rule_triggers = defaultdict(int)
        # P2.5: False-positive rule attribution
        self.fp_rule_attribution = defaultdict(int)

    def record_frame(self, frame_id: int, expected_state: str, actual_state: str, fused_evidence: list):
        """Record the state for a single frame for later analysis."""
        self.frame_results.append({
            "frame_id": frame_id,
            "expected_state": expected_state.upper(),
            "actual_state": actual_state.upper(),
        })

        positive_states = ["HIGH_RISK", "CRITICAL"]
        is_expected_pos = expected_state.upper() in positive_states
        is_actual_pos = actual_state.upper() in positive_states

        # Track which rules contribute to elevated states
        if is_actual_pos:
            for fe in fused_evidence:
                if fe.state in positive_states:
                    for rule in fe.contributing_rules:
                        self.rule_triggers[rule] += 1
                        self.rule_contributions[rule] += fe.evidence_score

                    # P2.5: If this is a FALSE POSITIVE, attribute to specific rules
                    if not is_expected_pos:
                        for rule in fe.contributing_rules:
                            self.fp_rule_attribution[rule] += 1

    def compute_metrics(self) -> Dict:
        """Calculate precision, recall, rule contributions, and FP attribution."""
        tp = fp = tn = fn = 0

        positive_states = ["HIGH_RISK", "CRITICAL"]

        for res in self.frame_results:
            is_expected_pos = res["expected_state"] in positive_states
            is_actual_pos = res["actual_state"] in positive_states

            if is_expected_pos and is_actual_pos:
                tp += 1
            elif not is_expected_pos and is_actual_pos:
                fp += 1
            elif not is_expected_pos and not is_actual_pos:
                tn += 1
            elif is_expected_pos and not is_actual_pos:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        # Normalize rule contributions
        total_contribution = sum(self.rule_contributions.values())
        normalized_contributions = {}
        if total_contribution > 0:
            normalized_contributions = {
                rule: round(score / total_contribution, 3)
                for rule, score in sorted(
                    self.rule_contributions.items(), key=lambda x: -x[1]
                )
            }

        # Sort FP attribution by frequency (descending)
        sorted_fp_attribution = dict(
            sorted(self.fp_rule_attribution.items(), key=lambda x: -x[1])
        )

        return {
            "frames_analyzed": len(self.frame_results),
            "true_positives_frames": tp,
            "false_positives_frames": fp,
            "true_negatives_frames": tn,
            "false_negatives_frames": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "rule_contributions": normalized_contributions,
            "rule_triggers": dict(self.rule_triggers),
            "false_positive_rule_attribution": sorted_fp_attribution,
        }
