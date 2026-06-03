import json
import csv
from pathlib import Path
from config import (
    ARM_SWING_THRESHOLD,
    APPROACH_VELOCITY_THRESHOLD,
    FALL_SCORE_THRESHOLD,
    INTRUSION_MAX_OCCUPANCY
)

def analyze():
    extremes_path = Path("evaluation/reports/feature_extremes.json")
    if not extremes_path.exists():
        print(f"File not found: {extremes_path}")
        return

    with open(extremes_path, "r") as f:
        extremes = json.load(f)

    print("="*50)
    print("FEATURE EXTREMES")
    print("="*50)
    print(json.dumps(extremes, indent=2))
    print("\n")

    print("="*50)
    print("THRESHOLD COMPARISON REPORT")
    print("="*50)

    comparisons = [
        ("ARM_SWING", ARM_SWING_THRESHOLD, extremes.get("max_arm_velocity", 0)),
        ("RAPID_APPROACH", APPROACH_VELOCITY_THRESHOLD, extremes.get("max_approach_velocity", 0)),
        ("FALL_EVENT", FALL_SCORE_THRESHOLD, extremes.get("max_fall_score", 0)),
        ("PROXIMITY_INTRUSION", INTRUSION_MAX_OCCUPANCY, extremes.get("max_proximity_ratio", 0)),
    ]

    for rule, thresh, obs in comparisons:
        pct = (obs / thresh * 100) if thresh > 0 else 0
        print(f"{rule}")
        print(f"threshold = {thresh}")
        print(f"max_observed = {obs:.4f}")
        print(f"threshold_reached = {pct:.1f}%")
        print()

    # Load CSV
    csv_path = Path("evaluation/reports/detection_audit.csv")
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    frames = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(row)

    if not frames:
        print("No frames in CSV.")
        return

    def print_top_20(metric_key, title):
        print("="*50)
        print(f"TOP 20 FRAMES BY: {title}")
        print("="*50)
        sorted_frames = sorted(frames, key=lambda x: float(x.get(metric_key, 0)), reverse=True)
        print(f"{'Rank':<5} | {'Frame ID':<10} | {'Time (s)':<10} | {'Value':<10}")
        print("-" * 50)
        for i, row in enumerate(sorted_frames[:20], 1):
            frame_id = int(row['frame_id'])
            time_s = frame_id / 30.0  # Assumes 30 FPS for timestamp approximation
            val = float(row[metric_key])
            print(f"{i:<5} | {frame_id:<10} | {time_s:<10.2f} | {val:<10.4f}")
        print("\n")

    print_top_20("arm_velocity", "ARM VELOCITY")
    print_top_20("approach_velocity", "APPROACH VELOCITY")
    print_top_20("fall_score", "FALL SCORE")

if __name__ == "__main__":
    analyze()
