"""
PROJECT 007 — P4.0 Hybrid ML Fusion Engine
Combines deterministic rule-engine scores with ML probability predictions.

Safety Constraints:
- ML can only increase or decrease confidence.
- ML CANNOT directly trigger CRITICAL state.
- Only the deterministic HysteresisStateMachine may escalate to CRITICAL.
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from training.feature_schema import ALL_FEATURES, CLASS_LABELS
from utils.logger import get_logger

logger = get_logger(__name__)

# Default fusion weights
DEFAULT_DETERMINISTIC_WEIGHT = 0.70
DEFAULT_ML_WEIGHT = 0.30


class MLFusionEngine:
    """
    Hybrid fusion: blends deterministic evidence scores with ML class probabilities.

    final_score = w_det * deterministic_score + w_ml * ml_probability

    Safety rules:
    1. ML can adjust confidence up or down.
    2. ML CANNOT directly trigger CRITICAL.
    3. If deterministic state is NORMAL or SUSPICIOUS, ML cannot push to CRITICAL.
    4. ML influence is capped: maximum adjustment is +/-0.3 from deterministic score.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        deterministic_weight: float = DEFAULT_DETERMINISTIC_WEIGHT,
        ml_weight: float = DEFAULT_ML_WEIGHT,
        ml_cap: float = 0.3,
    ):
        self.w_det = deterministic_weight
        self.w_ml = ml_weight
        self.ml_cap = ml_cap
        self.model = None
        self.feature_names = list(ALL_FEATURES)

        if model_path and Path(model_path).exists():
            self._load_model(model_path)

    def _load_model(self, path: str):
        """Load a trained ML model (sklearn or xgboost pickle)."""
        try:
            with open(path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"MLFusion: loaded model from {path}")
        except Exception as e:
            logger.error(f"MLFusion: failed to load model from {path}: {e}")
            self.model = None

    def is_active(self) -> bool:
        """Returns True if an ML model is loaded and ready."""
        return self.model is not None

    def fuse(
        self,
        features: dict,
        deterministic_score: float,
        deterministic_state: str,
    ) -> dict:
        """
        Compute hybrid fused score.

        Parameters
        ----------
        features : dict
            Feature vector from FeatureExtractor.extract().
        deterministic_score : float
            Evidence score from the deterministic FusionEngine.
        deterministic_state : str
            Current state from HysteresisStateMachine (NORMAL, SUSPICIOUS, HIGH_RISK, CRITICAL).

        Returns
        -------
        dict : {
            "fused_score": float,
            "deterministic_score": float,
            "ml_score": float,
            "ml_prediction": str,
            "ml_confidence": float,
            "ml_probabilities": dict,
            "top_features": list,
            "ml_active": bool,
            "safety_clamped": bool,
        }
        """
        result = {
            "deterministic_score": deterministic_score,
            "ml_active": False,
            "ml_score": 0.0,
            "ml_prediction": "normal",
            "ml_confidence": 0.0,
            "ml_probabilities": {},
            "top_features": [],
            "fused_score": deterministic_score,
            "safety_clamped": False,
        }

        if self.model is None:
            return result

        # Build feature vector
        x = np.array([[features.get(f, 0.0) for f in self.feature_names]])

        try:
            proba = self.model.predict_proba(x)[0]
            pred_idx = int(np.argmax(proba))

            # Map model classes to our class labels
            if hasattr(self.model, "classes_"):
                pred_class_id = int(self.model.classes_[pred_idx])
            else:
                pred_class_id = pred_idx

            pred_label = CLASS_LABELS.get(pred_class_id, f"class_{pred_class_id}")

            # ML "threat score" = 1 - P(normal)
            # Find the normal class index
            normal_prob = 0.0
            if hasattr(self.model, "classes_"):
                for i, cls_id in enumerate(self.model.classes_):
                    if int(cls_id) == 0:  # normal
                        normal_prob = float(proba[i])
                        break
            else:
                normal_prob = float(proba[0]) if len(proba) > 0 else 1.0

            ml_threat_score = 1.0 - normal_prob

            # Feature importance
            importances = self.model.feature_importances_ if hasattr(self.model, "feature_importances_") else []
            if len(importances) > 0:
                feat_contrib = sorted(
                    zip(self.feature_names, importances.tolist()),
                    key=lambda x: -x[1]
                )[:5]
            else:
                feat_contrib = []

            # Build probabilities dict
            probabilities = {}
            if hasattr(self.model, "classes_"):
                for i, p in enumerate(proba):
                    cls_name = CLASS_LABELS.get(int(self.model.classes_[i]), f"class_{self.model.classes_[i]}")
                    probabilities[cls_name] = round(float(p), 4)

            # ── Hybrid Fusion ──
            raw_fused = (self.w_det * deterministic_score) + (self.w_ml * ml_threat_score)

            # ── Safety Constraint: Cap ML influence ──
            ml_adjustment = raw_fused - deterministic_score
            if abs(ml_adjustment) > self.ml_cap:
                clamped_adjustment = self.ml_cap if ml_adjustment > 0 else -self.ml_cap
                raw_fused = deterministic_score + clamped_adjustment
                safety_clamped = True
            else:
                safety_clamped = False

            # ── Safety Constraint: ML cannot push to CRITICAL ──
            # Only deterministic state machine can reach CRITICAL.
            # If the deterministic state is below CRITICAL, cap the fused score below CRITICAL threshold.
            if deterministic_state in ["NORMAL", "SUSPICIOUS"]:
                raw_fused = min(raw_fused, 0.84)  # Below CRITICAL threshold (0.85)
                if raw_fused > deterministic_score + self.ml_cap:
                    safety_clamped = True

            raw_fused = max(0.0, min(1.0, raw_fused))

            result.update({
                "ml_active": True,
                "ml_score": round(ml_threat_score, 4),
                "ml_prediction": pred_label,
                "ml_confidence": round(float(proba[pred_idx]), 4),
                "ml_probabilities": probabilities,
                "top_features": [[f, round(v, 4)] for f, v in feat_contrib],
                "fused_score": round(raw_fused, 4),
                "safety_clamped": safety_clamped,
            })

        except Exception as e:
            logger.warning(f"MLFusion prediction failed: {e}")
            # Fallback to deterministic only
            result["fused_score"] = deterministic_score

        return result

    def update_weights(self, deterministic_weight: float, ml_weight: float):
        """Update fusion weights at runtime."""
        self.w_det = deterministic_weight
        self.w_ml = ml_weight
        logger.info(f"MLFusion weights updated: det={self.w_det}, ml={self.w_ml}")

    def get_config(self) -> dict:
        """Return current configuration."""
        return {
            "deterministic_weight": self.w_det,
            "ml_weight": self.w_ml,
            "ml_cap": self.ml_cap,
            "model_loaded": self.model is not None,
        }
