"""
PROJECT 007 — P6.0-A Regression Test
Quick headless replay of fight_test.mp4 to confirm egomotion integration
does not regress detection quality on static camera footage.
"""
import json
import sys
from pathlib import Path
from evaluation.replay_engine import ReplayEngine

engine = ReplayEngine(sync_mode=True)
result = engine.replay_video(
    "dataset/interaction/fight_test.mp4",
    category="interaction",
    show_ui=False,
)

if not result:
    print("ERROR: Replay returned empty result")
    sys.exit(1)

metrics = result.get("metrics", {})
precision = metrics.get("precision", 0)
recall = metrics.get("recall", 0)
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("\n" + "=" * 60)
print("  P6.0-A REGRESSION TEST — fight_test.mp4")
print("=" * 60)
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  TP frames : {metrics.get('true_positives_frames', 0)}")
print(f"  FP frames : {metrics.get('false_positives_frames', 0)}")
print(f"  TN frames : {metrics.get('true_negatives_frames', 0)}")
print(f"  FN frames : {metrics.get('false_negatives_frames', 0)}")
print("=" * 60)

# Baseline from P5.1: F1 = 0.9532, Precision = 0.9600, Recall = 0.9465
if f1 >= 0.90:
    print("  RESULT: PASS ✓ (F1 >= 0.90 threshold)")
else:
    print("  RESULT: FAIL ✗ (F1 < 0.90 threshold — regression detected!)")
    sys.exit(1)

# Save regression results
out_path = Path("evaluation/reports")
out_path.mkdir(parents=True, exist_ok=True)
with open(out_path / "p6_regression_test.json", "w") as f:
    json.dump({
        "test": "P6.0-A Egomotion Regression",
        "video": "fight_test.mp4",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "metrics": metrics,
    }, f, indent=4)
print(f"  Results saved to evaluation/reports/p6_regression_test.json")
