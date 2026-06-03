"""
PROJECT 007 — P2.5 Report Generator
Compiles evaluation reports with calibration recommendations and latency analysis.

Outputs:
  evaluation/reports/evaluation_report.json
"""

import json
from pathlib import Path
from typing import Dict, List

from utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    def __init__(self, output_dir="evaluation/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, sweep_results: List[Dict], latency_data: Dict = None):
        """
        Parse sweep results and generate evaluation_report.json.
        Includes performance metrics, rule analysis, and latency validation.
        """
        if not sweep_results:
            logger.error("No sweep results provided for report generation.")
            return

        # Find best F1
        best_f1 = -1
        best_sweep = None

        for res in sweep_results:
            metrics = res["metrics"]
            f1 = metrics.get("f1_score", 0)
            if f1 == 0:
                precision = metrics["precision"]
                recall = metrics["recall"]
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_sweep = res

        if not best_sweep:
            return

        best_metrics = best_sweep["metrics"]

        # Aggregate FP rule attribution across all sweeps
        all_fp_rules = {}
        for res in sweep_results:
            for rule, count in res["metrics"].get("false_positive_rule_attribution", {}).items():
                all_fp_rules[rule] = all_fp_rules.get(rule, 0) + count

        sorted_fp_rules = sorted(all_fp_rules.items(), key=lambda x: -x[1])

        report = {
            "evaluation_summary": {
                "total_configurations_tested": len(sweep_results),
                "best_configuration_id": best_sweep["sweep_id"],
                "best_weights": best_sweep["weights"],
                "best_f1_score": round(best_f1, 4),
            },
            "performance_metrics": {
                "true_positives": best_metrics["true_positives_frames"],
                "false_positives": best_metrics["false_positives_frames"],
                "true_negatives": best_metrics["true_negatives_frames"],
                "false_negatives": best_metrics["false_negatives_frames"],
                "precision": best_metrics["precision"],
                "recall": best_metrics["recall"],
                "f1_score": best_metrics.get("f1_score", round(best_f1, 4)),
                "false_positive_rate": best_metrics.get("false_positive_rate", 0),
                "false_negative_rate": best_metrics.get("false_negative_rate", 0),
            },
            "rule_analysis": {
                "top_contributing_rules": best_metrics.get("rule_contributions", {}),
                "raw_triggers": best_metrics.get("rule_triggers", {}),
                "false_positive_attribution": dict(sorted_fp_rules),
            },
            "confusion_matrix": best_sweep.get("confusion_matrix", {}),
        }

        # Add latency validation if available
        if latency_data:
            report["latency_validation"] = latency_data

        report_path = self.output_dir / "evaluation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        logger.info(f"Evaluation report generated at {report_path}")
        return report


if __name__ == "__main__":
    generator = ReportGenerator()
    sweep_path = Path("evaluation/reports/threshold_results.json")
    if sweep_path.exists():
        with open(sweep_path, "r") as f:
            data = json.load(f)
        generator.generate_report(data)
    else:
        logger.error(f"No sweep results found at {sweep_path}")
