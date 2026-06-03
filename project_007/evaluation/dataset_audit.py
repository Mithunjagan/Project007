"""
PROJECT 007 — P3.0 Dataset Audit
Generates quality reports about the dataset.

Usage:
    python -m evaluation.dataset_audit
"""

import json
from pathlib import Path
from collections import defaultdict

from dataset_tools.session_manager import SessionManager
from dataset_tools.video_indexer import VideoIndexer
from dataset_tools.annotation_template import load_annotation
from utils.logger import get_logger

logger = get_logger(__name__)

SCENARIO_CATEGORIES = [
    "normal",
    "camera_tamper",
    "intrusion",
    "interaction",
    "crowded",
    "occlusion",
    "lighting_change",
]


class DatasetAudit:
    """Generates quality reports about the dataset."""

    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.sm = SessionManager(dataset_dir)
        self.vi = VideoIndexer(dataset_dir)

    def run_audit(self) -> dict:
        """Run a full audit and return the report."""
        index = self.vi.index_all()

        # Category distribution
        category_dist = defaultdict(int)
        category_duration = defaultdict(float)
        durations = []

        for v in index:
            category_dist[v["category"]] += 1
            category_duration[v["category"]] += v["duration_sec"]
            durations.append(v["duration_sec"])

        total_duration = sum(durations)
        avg_duration = total_duration / len(durations) if durations else 0

        # Annotation coverage
        annotated_count = sum(1 for v in index if v["has_annotation"])
        unannotated = [v["filename"] for v in index if not v["has_annotation"]]

        # Class imbalance
        max_count = max(category_dist.values()) if category_dist else 1
        imbalance = {}
        for cat in SCENARIO_CATEGORIES:
            count = category_dist.get(cat, 0)
            ratio = count / max_count if max_count > 0 else 0
            imbalance[cat] = {
                "count": count,
                "ratio_to_max": round(ratio, 3),
                "duration_sec": round(category_duration.get(cat, 0), 2),
            }

        # Annotation event stats
        total_events = 0
        event_label_dist = defaultdict(int)
        anno_dir = self.dataset_dir / "annotations"
        if anno_dir.exists():
            for anno_file in anno_dir.glob("*.json"):
                try:
                    with open(anno_file, "r") as f:
                        anno = json.load(f)
                    for ev in anno.get("events", []):
                        total_events += 1
                        event_label_dist[ev.get("label", "unknown")] += 1
                except Exception:
                    pass

        report = {
            "total_videos": len(index),
            "total_duration_sec": round(total_duration, 2),
            "total_duration_hours": round(total_duration / 3600, 4),
            "average_clip_duration_sec": round(avg_duration, 2),
            "category_distribution": dict(category_dist),
            "class_imbalance": imbalance,
            "annotation_coverage": {
                "annotated": annotated_count,
                "unannotated": len(index) - annotated_count,
                "coverage_pct": round(
                    100 * annotated_count / len(index), 1
                ) if index else 0,
                "unannotated_files": unannotated[:20],
            },
            "event_statistics": {
                "total_events": total_events,
                "label_distribution": dict(event_label_dist),
            },
        }

        return report

    def print_report(self):
        """Print a human-readable audit report."""
        report = self.run_audit()

        print("\n" + "=" * 60)
        print("  PROJECT 007 — DATASET AUDIT REPORT")
        print("=" * 60)
        print(f"  Total videos       : {report['total_videos']}")
        print(f"  Total duration     : {report['total_duration_hours']:.2f} hours")
        print(f"  Avg clip duration  : {report['average_clip_duration_sec']:.1f}s")
        print(f"  Annotation coverage: {report['annotation_coverage']['coverage_pct']:.1f}%")
        print(f"  Total events       : {report['event_statistics']['total_events']}")
        print()
        print("  Category Distribution:")
        for cat, info in report["class_imbalance"].items():
            bar = "█" * int(info["ratio_to_max"] * 20)
            print(f"    {cat:<20} {info['count']:>3}  {bar}")
        print("=" * 60)

        return report

    def save_report(self, output_path: str = "evaluation/reports/dataset_audit.json"):
        """Save audit report to JSON."""
        report = self.run_audit()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=4)
        logger.info(f"Audit report saved to {out}")
        return report


if __name__ == "__main__":
    audit = DatasetAudit()
    audit.print_report()
    audit.save_report()
