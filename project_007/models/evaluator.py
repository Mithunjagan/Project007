"""
PROJECT 007 — P4.0 Model Evaluator
Compares Random Forest vs XGBoost baseline models.

Usage:
    python -m models.evaluator [--dataset dataset/training_dataset.parquet]
"""

import argparse
import json
from pathlib import Path

from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
from training.feature_schema import CLASS_LABELS
from utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_models(dataset_path: str = "dataset/training_dataset.parquet") -> dict:
    """
    Train and compare both baseline models.

    Returns
    -------
    dict : Comparison results with metrics for each model.
    """
    results = {}

    # 1. Random Forest
    logger.info("=" * 50)
    logger.info("Training Random Forest...")
    logger.info("=" * 50)
    try:
        rf = RandomForestModel()
        rf_results = rf.train(dataset_path)
        results["random_forest"] = rf_results
        rf.save()
    except Exception as e:
        logger.error(f"Random Forest failed: {e}")
        results["random_forest"] = {"error": str(e)}

    # 2. XGBoost
    logger.info("=" * 50)
    logger.info("Training XGBoost...")
    logger.info("=" * 50)
    try:
        xgb_model = XGBoostModel()
        xgb_results = xgb_model.train(dataset_path)
        results["xgboost"] = xgb_results
        xgb_model.save()
    except Exception as e:
        logger.error(f"XGBoost failed: {e}")
        results["xgboost"] = {"error": str(e)}

    # 3. Comparison summary
    comparison = {}
    for model_name, r in results.items():
        if "error" not in r:
            comparison[model_name] = {
                "accuracy": r.get("accuracy", 0),
                "macro_f1": r.get("macro_f1", 0),
                "macro_precision": r.get("macro_precision", 0),
                "macro_recall": r.get("macro_recall", 0),
                "inference_latency_ms": r.get("inference_latency_ms", 0),
                "train_time_sec": r.get("train_time_sec", 0),
            }

    results["comparison"] = comparison

    # Determine best model
    best = max(comparison.items(), key=lambda x: x[1]["macro_f1"]) if comparison else None
    if best:
        results["recommended_model"] = best[0]
        logger.info(f"Recommended model: {best[0]} (F1={best[1]['macro_f1']:.4f})")

    # Save
    out = Path("evaluation/reports/ml_baseline.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"ML baseline report saved to {out}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P4.0 Model Evaluator")
    parser.add_argument("--dataset", default="dataset/training_dataset.parquet")
    args = parser.parse_args()

    results = evaluate_models(args.dataset)
    print(json.dumps(results.get("comparison", {}), indent=4))
