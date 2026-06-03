"""
PROJECT 007 — P4.0 Random Forest Classifier
Trains and evaluates a Random Forest on the extracted feature dataset.

Usage:
    python -m models.random_forest [--dataset dataset/training_dataset.parquet]
"""

import argparse
import json
import time
import pickle
from pathlib import Path

import numpy as np

try:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        accuracy_score,
        precision_recall_fscore_support,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from training.feature_schema import ALL_FEATURES, CLASS_LABELS
from utils.logger import get_logger

logger = get_logger(__name__)


class RandomForestModel:
    """Wrapper around sklearn RandomForest for PROJECT 007."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 12, random_state: int = 42):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required: pip install scikit-learn")

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.feature_names = list(ALL_FEATURES)
        self.is_trained = False

    def train(self, dataset_path: str, test_size: float = 0.2) -> dict:
        """
        Train the model on the dataset.

        Returns
        -------
        dict : Training results including metrics.
        """
        df = pd.read_parquet(dataset_path)

        # Filter to known features
        available_features = [f for f in self.feature_names if f in df.columns]
        if not available_features:
            raise ValueError("No matching features found in dataset")

        X = df[available_features].fillna(0).values
        y = df["label"].values.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        logger.info(f"Training Random Forest: {X_train.shape[0]} train, {X_test.shape[0]} test")

        start = time.time()
        self.model.fit(X_train, y_train)
        train_time = time.time() - start
        self.is_trained = True

        # Evaluate
        start = time.time()
        y_pred = self.model.predict(X_test)
        inference_time = (time.time() - start) / max(1, len(X_test))

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None, zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred)

        # Feature importance
        importances = self.model.feature_importances_
        feat_imp = sorted(
            zip(available_features, importances.tolist()),
            key=lambda x: -x[1]
        )

        # Per-class metrics
        per_class = {}
        unique_classes = sorted(set(y_test) | set(y_pred))
        for i, cls_id in enumerate(unique_classes):
            if i < len(precision):
                cls_name = CLASS_LABELS.get(cls_id, f"class_{cls_id}")
                per_class[cls_name] = {
                    "precision": round(float(precision[i]), 4),
                    "recall": round(float(recall[i]), 4),
                    "f1": round(float(f1[i]), 4),
                    "support": int(support[i]),
                }

        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        results = {
            "model": "RandomForest",
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features_used": len(available_features),
            "train_time_sec": round(train_time, 3),
            "inference_latency_ms": round(inference_time * 1000, 4),
            "accuracy": round(float(accuracy), 4),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4),
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "feature_importance": feat_imp[:20],
        }

        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Macro F1: {macro_f1:.4f}")
        logger.info(f"  Train time: {train_time:.2f}s")
        logger.info(f"  Inference: {inference_time * 1000:.4f}ms/sample")

        return results

    def predict(self, features: dict) -> dict:
        """
        Predict class and return explainable output.

        Returns
        -------
        dict : {"prediction": str, "confidence": float, "probabilities": dict, "top_features": list}
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        x = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        proba = self.model.predict_proba(x)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = CLASS_LABELS.get(pred_idx, f"class_{pred_idx}")

        # Feature importance for this prediction
        importances = self.model.feature_importances_
        feat_contrib = sorted(
            zip(self.feature_names, importances.tolist()),
            key=lambda x: -x[1]
        )[:5]

        probabilities = {}
        for i, p in enumerate(proba):
            cls_name = CLASS_LABELS.get(self.model.classes_[i], f"class_{self.model.classes_[i]}")
            probabilities[cls_name] = round(float(p), 4)

        return {
            "prediction": pred_label,
            "confidence": round(float(proba[pred_idx]), 4),
            "probabilities": probabilities,
            "top_features": [[f, round(v, 4)] for f, v in feat_contrib],
        }

    def save(self, path: str = "models/saved/random_forest.pkl"):
        """Save trained model to disk."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {out}")

    def load(self, path: str = "models/saved/random_forest.pkl"):
        """Load a trained model."""
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True
        logger.info(f"Model loaded from {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P4.0 Random Forest")
    parser.add_argument("--dataset", default="dataset/training_dataset.parquet")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    model = RandomForestModel()
    results = model.train(args.dataset)

    Path("evaluation/reports").mkdir(parents=True, exist_ok=True)
    with open("evaluation/reports/rf_results.json", "w") as f:
        json.dump(results, f, indent=4)

    if args.save:
        model.save()

    print(json.dumps(results, indent=4))
