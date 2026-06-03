"""
PROJECT 007 — P5.0 Violence Classifier Training
Trains the LSTM-based ViolenceNet on annotated surveillance videos.

Usage:
    python -m training.train_violence_classifier [--dataset-dir dataset] [--epochs 50] [--lr 0.001]

Flow:
    1. Scan dataset for annotated videos (VideoIndexer)
    2. Encode frames with MobileNetV3-Small → 576-dim embeddings
    3. Run pipeline in sync_mode to extract per-frame motion features
    4. Build sliding windows of DL_TEMPORAL_WINDOW frames
    5. Map annotation labels to 4 violence classes
    6. Train ViolenceNet (LSTM + classifier head) with class-weighted CE loss
"""

import argparse
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    DL_TEMPORAL_WINDOW,
    DL_ENCODER_DEVICE,
    DL_CLASSIFIER_PATH,
    DL_ENCODER_MODEL,
    EVENT_LABELS,
    DL_ENCODE_EVERY_N_FRAMES,
)
from dataset_tools.annotation_template import load_annotation
from dataset_tools.video_indexer import VideoIndexer
from utils.logger import get_logger
from pipeline.core import PipelineRunner
from models.violence_classifier import ViolenceNet

logger = get_logger(__name__)

# ═══════════════════════════════════════════════
# Label mapping: annotation event → violence class
# ═══════════════════════════════════════════════
LABEL_MAP = {
    "normal": 0,                       # NORMAL
    "camera_rush": 1,                  # SUSPICIOUS
    "proximity_intrusion": 1,          # SUSPICIOUS
    "camera_shake": 2,                 # HIGH_RISK
    "lens_occlusion": 2,               # HIGH_RISK
    "high_energy_interaction": 3,      # CRITICAL
}

CLASS_NAMES = {
    0: "NORMAL",
    1: "SUSPICIOUS",
    2: "HIGH_RISK",
    3: "CRITICAL",
}

NUM_CLASSES = 4

_MOTION_KEYS = [
    "arm_velocity",
    "body_displacement",
    "fall_score",
    "approach_velocity",
]
_SCENE_KEYS = [
    "occupancy_ratio",
    "scene_stability",
]


# ═══════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════
def _frame_label(frame_idx: int, events: list, fps: float) -> int:
    """Return the violence class for a frame given annotation events."""
    for ev in events:
        sf = ev.get("start_frame", 0)
        ef = ev.get("end_frame", 0)
        if sf <= frame_idx <= ef:
            return LABEL_MAP.get(ev.get("label", "normal"), 0)
    return 0  # default: normal


def _build_windows(embeddings: list, labels: list, window: int):
    """
    Build sliding windows from per-frame embeddings and labels.

    Each window of *window* frames maps to the label of the centre frame.
    Returns (X, y) as numpy arrays.
    """
    if len(embeddings) < window:
        return np.empty((0,)), np.empty((0,), dtype=np.int64)

    X_windows = []
    y_labels = []
    half = window // 2

    for i in range(half, len(embeddings) - half):
        seq = embeddings[i - half: i - half + window]
        X_windows.append(np.stack(seq, axis=0))     # (T, D)
        y_labels.append(labels[i])                   # centre label

    return np.array(X_windows), np.array(y_labels, dtype=np.int64)


# ═══════════════════════════════════════════════
# Video processing
# ═══════════════════════════════════════════════
def process_video(
    video_path: str,
    annotation: dict,
    window: int = DL_TEMPORAL_WINDOW,
):
    """
    Encode all frames of a video and build training windows.

    Returns (X, y) numpy arrays or (None, None) on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    events = annotation.get("events", [])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    runner = PipelineRunner(
        width, height,
        sync_mode=True,
        enable_recording=False,
        log_telemetry=False,
    )

    logger.info(f"  Encoding {total} frames (fps={fps:.1f}) via PipelineRunner …")

    features = []
    labels = []
    frame_count = 0
    now_mono = time.perf_counter()
    now_wall = time.time()
    time_delta = 1.0 / fps

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        now_mono += time_delta
        now_wall += time_delta

        runner.step(frame, frame_count, now_mono, now_wall)

        # Get features and label if this frame was encoded
        if frame_count % DL_ENCODE_EVERY_N_FRAMES == 0:
            if len(runner._frame_embeddings) > 0:
                emb = runner._frame_embeddings[-1]
                motion = runner._frame_motion_ctx[-1]
                scene = runner._frame_scene_ctx[-1]

                motion_vec = np.array(
                    [float(motion.get(k, 0.0)) for k in _MOTION_KEYS],
                    dtype=np.float32,
                )
                scene_vec = np.array(
                    [float(scene.get(k, 0.0)) for k in _SCENE_KEYS],
                    dtype=np.float32,
                )
                feat = np.concatenate([emb, motion_vec, scene_vec])
                features.append(feat)

                labels.append(_frame_label(frame_count, events, fps))

    runner.cleanup()
    cap.release()
    logger.info(f"  Encoded {len(features)} frames → building windows")

    X, y = _build_windows(features, labels, window)
    return X, y


# ═══════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════
def train(
    dataset_dir: str = "dataset",
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 32,
    val_split: float = 0.2,
):
    """Main training entry point."""
    logger.info("=" * 60)
    logger.info("P5.0 Violence Classifier — Training")
    logger.info("=" * 60)

    # ── 1. Discover annotated videos ──
    indexer = VideoIndexer(dataset_dir)
    all_videos = indexer.index_all()
    annotated = [v for v in all_videos if v["has_annotation"]]

    if not annotated:
        logger.warning(
            "No annotated videos found in '%s'. "
            "Please annotate at least one video before training.",
            dataset_dir,
        )
        return

    logger.info(f"Found {len(annotated)} annotated video(s)")

    # ── 2. Encode all videos ──
    device = torch.device(DL_ENCODER_DEVICE if torch.cuda.is_available() else "cpu")

    all_X = []
    all_y = []

    for i, info in enumerate(annotated):
        vid_path = info["path"]
        vid_id = Path(vid_path).stem
        logger.info(f"[{i + 1}/{len(annotated)}] {vid_id}")

        anno = load_annotation(vid_id, str(Path(dataset_dir) / "annotations"))
        if anno is None:
            logger.warning(f"  Annotation file missing for {vid_id}, skipping")
            continue

        X, y = process_video(vid_path, anno, DL_TEMPORAL_WINDOW)
        if X is None or len(X) == 0:
            logger.warning(f"  No valid windows from {vid_id}")
            continue

        all_X.append(X)
        all_y.append(y)
        logger.info(f"  {len(X)} windows extracted")

    if not all_X:
        logger.warning("No training windows could be generated. Exiting.")
        return

    X_all = np.concatenate(all_X, axis=0)   # (N, T, D)
    y_all = np.concatenate(all_y, axis=0)   # (N,)

    logger.info(f"Total windows: {len(X_all)}")
    class_counts = Counter(y_all.tolist())
    for cls_id in sorted(class_counts):
        logger.info(f"  Class {cls_id} ({CLASS_NAMES[cls_id]}): {class_counts[cls_id]}")

    # ── 3. Train/val split (80/20 by frames) ──
    n = len(X_all)
    indices = np.random.permutation(n)
    split = int(n * (1.0 - val_split))
    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train = torch.tensor(X_all[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y_all[train_idx], dtype=torch.long)
    X_val = torch.tensor(X_all[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y_all[val_idx], dtype=torch.long)

    logger.info(f"Train: {len(X_train)}  |  Val: {len(X_val)}")

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # ── 4. Class weights (inverse frequency) ──
    total_samples = sum(class_counts.values())
    weights = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for cls_id in range(NUM_CLASSES):
        count = class_counts.get(cls_id, 0)
        if count > 0:
            weights[cls_id] = total_samples / (NUM_CLASSES * count)
        else:
            weights[cls_id] = 1.0
    weights = weights.to(device)
    logger.info(f"Class weights: {weights.tolist()}")

    # ── 5. Model, loss, optimizer ──
    input_dim = X_all.shape[2]  # 576
    model = ViolenceNet(input_dim=input_dim).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info(f"ViolenceNet: input_dim={input_dim}, device={device}")
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {total_params:,}")

    # ── 6. Training ──
    save_dir = Path(DL_CLASSIFIER_PATH).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(DL_CLASSIFIER_PATH)

    best_val_acc = 0.0
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        # — Train —
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # — Validate —
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)

        val_loss = val_loss / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        logger.info(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "num_classes": NUM_CLASSES,
                "epoch": epoch,
                "val_acc": val_acc,
                "class_names": CLASS_NAMES,
                "label_map": LABEL_MAP,
            }, save_path)
            logger.info(f"  ✓ Best model saved → {save_path} (val_acc={val_acc:.4f})")

    elapsed = time.time() - train_start
    logger.info("=" * 60)
    logger.info(f"Training complete in {elapsed:.1f}s")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Model saved to: {save_path}")
    logger.info("=" * 60)


# ═══════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="P5.0 — Train LSTM Violence Classifier"
    )
    parser.add_argument(
        "--dataset-dir", default="dataset",
        help="Path to the dataset directory (default: dataset)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate for Adam optimizer (default: 0.001)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Mini-batch size (default: 32)",
    )
    args = parser.parse_args()

    train(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )
