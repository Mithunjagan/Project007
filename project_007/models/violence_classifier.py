"""
PROJECT 007 — Deep-Learning Violence Classifier (P5.0)
LSTM-based temporal classifier that consumes frame embeddings,
motion features, and scene features to predict threat state.

Classes:
    NORMAL=0, SUSPICIOUS=1, HIGH_RISK=2, CRITICAL=3
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Config imports (graceful fallback) ──────────────────────────
try:
    from config import DL_CLASSIFIER_PATH, DL_TEMPORAL_WINDOW
except ImportError:
    DL_CLASSIFIER_PATH = "models/saved/violence_classifier.pt"
    DL_TEMPORAL_WINDOW = 16
    logger.warning(
        "DL_CLASSIFIER_PATH / DL_TEMPORAL_WINDOW not found in config — "
        "using defaults"
    )

# ── Constants ───────────────────────────────────────────────────
STATE_MAP = {0: "NORMAL", 1: "SUSPICIOUS", 2: "HIGH_RISK", 3: "CRITICAL"}

INPUT_DIM = 582          # 576 frame embedding + 4 motion + 2 scene
HIDDEN_SIZE = 128
NUM_LAYERS = 1
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


# ════════════════════════════════════════════════════════════════
# ViolenceNet — PyTorch LSTM model
# ════════════════════════════════════════════════════════════════
if HAS_TORCH:

    class ViolenceNet(nn.Module):
        """LSTM-based temporal violence classifier.

        Input shape  : (batch, seq_len, 582)
        Output shape : (batch, 4)  — raw logits
        """

        def __init__(
            self,
            input_dim: int = INPUT_DIM,
            hidden_size: int = HIDDEN_SIZE,
            num_layers: int = NUM_LAYERS,
            num_classes: int = NUM_CLASSES,
        ):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: (batch, seq_len, input_dim)

            Returns:
                logits: (batch, num_classes)
            """
            # lstm_out: (batch, seq_len, hidden_size)
            lstm_out, _ = self.lstm(x)
            # Take the last time-step's hidden state
            last_hidden = lstm_out[:, -1, :]
            logits = self.fc(last_hidden)
            return logits


# ════════════════════════════════════════════════════════════════
# ViolenceClassifier — inference wrapper
# ════════════════════════════════════════════════════════════════
class ViolenceClassifier:
    """High-level inference wrapper around ViolenceNet.

    Handles model loading, feature concatenation, and softmax
    conversion so callers only deal with plain Python objects.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model: object = None  # ViolenceNet when loaded
        self._device: str = "cpu"

        if not HAS_TORCH:
            logger.warning("PyTorch not installed — violence classifier disabled")
            return

        model_path = model_path or DL_CLASSIFIER_PATH
        self._model_path = Path(model_path)

        if self._model_path.exists():
            self._load(self._model_path)
        else:
            logger.info(
                "No saved violence model found at %s — classifier inactive",
                self._model_path,
            )

    # ── public API ──────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if a trained model is loaded and ready."""
        return self.model is not None

    def predict(
        self,
        frame_embeddings: list[np.ndarray],
        motion_features: list[dict],
        scene_features: list[dict],
    ) -> dict:
        """Run temporal inference over a window of frames.

        Args:
            frame_embeddings: list of 576-dim arrays (one per frame).
            motion_features:  list of dicts with keys
                              arm_velocity, body_displacement,
                              fall_score, approach_velocity.
            scene_features:   list of dicts with keys
                              occupancy_ratio, scene_stability.

        Returns:
            {
                "state":       str,          # e.g. "NORMAL"
                "confidence":  float,        # softmax probability
                "class_probs": list[float],  # length-4 softmax
            }
        """
        if not self.is_available():
            return {
                "state": "NORMAL",
                "confidence": 0.0,
                "class_probs": [1.0, 0.0, 0.0, 0.0],
            }

        seq_len = len(frame_embeddings)
        combined = np.zeros((seq_len, INPUT_DIM), dtype=np.float32)

        for t in range(seq_len):
            # 576-dim frame embedding
            emb = np.asarray(frame_embeddings[t], dtype=np.float32).ravel()
            if emb.shape[0] != 576:
                logger.warning(
                    "Frame embedding dim=%d at t=%d, expected 576 — zero-padding",
                    emb.shape[0], t,
                )
                padded = np.zeros(576, dtype=np.float32)
                padded[: min(emb.shape[0], 576)] = emb[: 576]
                emb = padded

            # 4 motion features
            mf = motion_features[t] if t < len(motion_features) else {}
            motion_vec = np.array(
                [float(mf.get(k, 0.0)) for k in _MOTION_KEYS],
                dtype=np.float32,
            )

            # 2 scene features
            sf = scene_features[t] if t < len(scene_features) else {}
            scene_vec = np.array(
                [float(sf.get(k, 0.0)) for k in _SCENE_KEYS],
                dtype=np.float32,
            )

            combined[t] = np.concatenate([emb, motion_vec, scene_vec])

        # (1, seq_len, 582)
        tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)
        tensor = tensor.to(self._device)

        with torch.no_grad():
            logits = self.model(tensor)                    # (1, 4)
            probs = torch.softmax(logits, dim=-1)[0]       # (4,)

        probs_list = probs.cpu().tolist()
        pred_idx = int(torch.argmax(probs).item())

        return {
            "state": STATE_MAP[pred_idx],
            "confidence": float(probs_list[pred_idx]),
            "class_probs": [round(p, 6) for p in probs_list],
        }

    # ── internal ────────────────────────────────────────────────

    def _load(self, path: Path) -> None:
        """Load a saved ViolenceNet checkpoint."""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            checkpoint = torch.load(path, map_location=device, weights_only=False)

            model = ViolenceNet()

            # Support both raw state-dict and wrapped checkpoint
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            model.to(device)
            model.eval()

            self.model = model
            self._device = device
            logger.info(
                "Violence classifier loaded from %s (device=%s)", path, device
            )
        except Exception as exc:
            logger.error("Failed to load violence model from %s: %s", path, exc)
            self.model = None
