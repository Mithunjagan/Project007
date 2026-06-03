"""
PROJECT 007 — P3.0 Inter-Annotator Agreement
Computes agreement metrics between multiple annotation files per video.

Usage:
    python -m evaluation.inter_annotator <video_id> <anno_file_1> <anno_file_2>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

from utils.logger import get_logger

logger = get_logger(__name__)


def _frame_labels(annotation: dict, total_frames: int) -> list:
    """
    Convert event-based annotation to per-frame label list.
    Unlabeled frames get 'normal'.
    """
    labels = ["normal"] * total_frames
    for ev in annotation.get("events", []):
        sf = ev.get("start_frame", 0)
        ef = ev.get("end_frame", 0)
        label = ev.get("label", "normal")
        for i in range(sf, min(ef + 1, total_frames)):
            labels[i] = label
    return labels


def compute_agreement(anno1: dict, anno2: dict, total_frames: int = 0) -> dict:
    """
    Compute agreement metrics between two annotations.

    Returns dict with:
      - agreement_pct
      - cohens_kappa
      - disagreements: list of {frame, label1, label2}
    """
    if total_frames == 0:
        total_frames = max(
            anno1.get("total_frames", 0),
            anno2.get("total_frames", 0),
            1,
        )

    labels1 = _frame_labels(anno1, total_frames)
    labels2 = _frame_labels(anno2, total_frames)

    # Raw agreement
    agree = sum(1 for a, b in zip(labels1, labels2) if a == b)
    agreement_pct = round(100 * agree / total_frames, 2)

    # Collect all labels
    all_labels = sorted(set(labels1 + labels2))
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    n_labels = len(all_labels)

    # Confusion matrix for kappa
    matrix = [[0] * n_labels for _ in range(n_labels)]
    for a, b in zip(labels1, labels2):
        matrix[label_to_idx[a]][label_to_idx[b]] += 1

    # Cohen's Kappa
    p_observed = agree / total_frames
    p_expected = 0.0
    for i in range(n_labels):
        row_sum = sum(matrix[i])
        col_sum = sum(matrix[j][i] for j in range(n_labels))
        p_expected += (row_sum / total_frames) * (col_sum / total_frames)

    if p_expected >= 1.0:
        kappa = 1.0
    else:
        kappa = round((p_observed - p_expected) / (1 - p_expected), 4)

    # Disagreement report (sample up to 100)
    disagreements = []
    for i, (a, b) in enumerate(zip(labels1, labels2)):
        if a != b and len(disagreements) < 100:
            disagreements.append({
                "frame": i,
                "annotator_1": a,
                "annotator_2": b,
            })

    return {
        "total_frames": total_frames,
        "agreement_pct": agreement_pct,
        "cohens_kappa": kappa,
        "disagreement_count": total_frames - agree,
        "disagreements": disagreements,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inter-Annotator Agreement")
    parser.add_argument("video_id", help="Video ID")
    parser.add_argument("anno1", help="Path to annotation file 1")
    parser.add_argument("anno2", help="Path to annotation file 2")
    parser.add_argument("--total-frames", type=int, default=0, help="Override total frame count")
    args = parser.parse_args()

    with open(args.anno1, "r") as f:
        a1 = json.load(f)
    with open(args.anno2, "r") as f:
        a2 = json.load(f)

    result = compute_agreement(a1, a2, args.total_frames)
    print(json.dumps(result, indent=4))
