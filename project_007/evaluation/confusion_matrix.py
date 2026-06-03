import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

class ConfusionMatrix:
    def __init__(self):
        self.states = ["NORMAL", "SUSPICIOUS", "HIGH_RISK", "CRITICAL"]
        self.matrix = {expected: {actual: 0 for actual in self.states} for expected in self.states}

    def add_result(self, expected_state: str, actual_state: str):
        # Fallback to NORMAL if unexpected state
        exp = expected_state.upper() if expected_state.upper() in self.states else "NORMAL"
        act = actual_state.upper() if actual_state.upper() in self.states else "NORMAL"
        
        self.matrix[exp][act] += 1

    def generate_text_report(self) -> str:
        lines = []
        lines.append("=" * 65)
        lines.append(f"{'CONFUSION MATRIX':^65}")
        lines.append("=" * 65)
        
        # Header
        header = f"{'Expected \\ Actual':<20}" + "".join([f"{s:>11}" for s in self.states])
        lines.append(header)
        lines.append("-" * 65)
        
        for exp in self.states:
            row_str = f"{exp:<20}"
            for act in self.states:
                row_str += f"{self.matrix[exp][act]:>11}"
            lines.append(row_str)
            
        lines.append("=" * 65)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return self.matrix
