"""
PROJECT 007 — P4.5 Feature Importance Analysis
Computes global feature importance from RF, XGBoost, and optionally SHAP.

Usage:
    python -m evaluation.feature_analysis [--dataset dataset/training_dataset.parquet]
"""

import argparse
import json
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from training.feature_schema import ALL_FEATURES, CLASS_LABELS
from utils.logger import get_logger

logger = get_logger(__name__)


def analyze_features(
    dataset_path: str = "dataset/training_dataset.parquet",
    output_path: str = "evaluation/reports/feature_analysis.json",
) -> dict:
    """
    Compute feature importance from multiple methods.
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn required")

    df = pd.read_parquet(dataset_path)
    available_features = [f for f in ALL_FEATURES if f in df.columns]

    if not available_features:
        return {"error": "no_features"}

    X = df[available_features].fillna(0).values
    y = df["label"].values.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    report = {"features": available_features}

    # 1. Random Forest importance
    logger.info("Computing Random Forest feature importance...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_imp = sorted(
        zip(available_features, rf.feature_importances_.tolist()),
        key=lambda x: -x[1]
    )
    report["random_forest"] = {
        "importance": [[f, round(v, 6)] for f, v in rf_imp],
        "top_5": [f for f, _ in rf_imp[:5]],
    }
    logger.info(f"  Top 5 RF: {[f for f, _ in rf_imp[:5]]}")

    # 2. XGBoost importance
    if HAS_XGB:
        logger.info("Computing XGBoost feature importance...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, use_label_encoder=False,
            eval_metric="mlogloss", n_jobs=-1, verbosity=0,
        )
        xgb_model.fit(X_train, y_train)
        xgb_imp = sorted(
            zip(available_features, xgb_model.feature_importances_.tolist()),
            key=lambda x: -x[1]
        )
        report["xgboost"] = {
            "importance": [[f, round(v, 6)] for f, v in xgb_imp],
            "top_5": [f for f, _ in xgb_imp[:5]],
        }
        logger.info(f"  Top 5 XGB: {[f for f, _ in xgb_imp[:5]]}")
    else:
        report["xgboost"] = {"error": "xgboost not installed"}

    # 3. SHAP (if available)
    if HAS_SHAP:
        logger.info("Computing SHAP values (this may take a while)...")
        try:
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_test[:200])  # Limit for speed

            # Mean absolute SHAP per feature
            if isinstance(shap_values, list):
                # Multi-class: average across classes
                mean_shap = np.mean(
                    [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
                )
            else:
                mean_shap = np.abs(shap_values).mean(axis=0)

            shap_imp = sorted(
                zip(available_features, mean_shap.tolist()),
                key=lambda x: -x[1]
            )
            report["shap"] = {
                "importance": [[f, round(v, 6)] for f, v in shap_imp],
                "top_5": [f for f, _ in shap_imp[:5]],
            }
            logger.info(f"  Top 5 SHAP: {[f for f, _ in shap_imp[:5]]}")
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            report["shap"] = {"error": str(e)}
    else:
        report["shap"] = {"available": False, "note": "Install shap: pip install shap"}

    # 4. Feature correlation (top correlated pairs)
    logger.info("Computing feature correlations...")
    corr_matrix = pd.DataFrame(X, columns=available_features).corr()
    high_corr = []
    for i in range(len(available_features)):
        for j in range(i + 1, len(available_features)):
            c = abs(corr_matrix.iloc[i, j])
            if c > 0.8:
                high_corr.append({
                    "feature_1": available_features[i],
                    "feature_2": available_features[j],
                    "correlation": round(float(c), 4),
                })
    high_corr.sort(key=lambda x: -x["correlation"])
    report["high_correlations"] = high_corr[:20]

    # 5. Consensus ranking
    ranks = {}
    for method_key in ["random_forest", "xgboost"]:
        method = report.get(method_key, {})
        if "importance" in method:
            for rank, (feat, _) in enumerate(method["importance"]):
                ranks.setdefault(feat, []).append(rank)
    if "shap" in report and "importance" in report["shap"]:
        for rank, (feat, _) in enumerate(report["shap"]["importance"]):
            ranks.setdefault(feat, []).append(rank)

    avg_ranks = [(f, round(sum(r) / len(r), 2)) for f, r in ranks.items()]
    avg_ranks.sort(key=lambda x: x[1])
    report["consensus_ranking"] = [[f, r] for f, r in avg_ranks]

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=4)
    logger.info(f"Feature analysis saved to {out}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P4.5 Feature Analysis")
    parser.add_argument("--dataset", default="dataset/training_dataset.parquet")
    args = parser.parse_args()

    analyze_features(args.dataset)
