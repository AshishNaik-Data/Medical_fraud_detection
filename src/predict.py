"""
Inference pipeline: load trained models and generate predictions for the test set.

Usage:
    python predict.py                   # Use ensemble model
    python predict.py --baseline        # Use baseline XGBoost only
    python predict.py --neo4j           # Use Neo4j for graph features
"""
import json
import logging
import sys

import numpy as np
import pandas as pd

from config import MODEL_DIR, OUTPUT_DIR, RANDOM_SEED
from feature_engineering import build_scaled_features
from utils import load_artifact, save_artifact, set_global_seed, setup_logging

logger = setup_logging()


def predict_baseline(X_test: np.ndarray, feature_names: np.ndarray) -> np.ndarray:
    """Generate predictions using the baseline XGBoost model."""
    artifact = load_artifact(MODEL_DIR / "baseline_xgboost.pkl")
    model = artifact["model"]
    threshold = artifact["threshold"]
    model_feats = artifact["feature_names"]

    # Align features (use same pruned columns as training)
    feat_idx = [np.where(feature_names == f)[0][0] for f in model_feats
                if f in feature_names]
    X_aligned = X_test[:, feat_idx]

    probs = model.predict_proba(X_aligned)[:, 1]
    logger.info(f"Baseline predictions: mean_prob={probs.mean():.4f}, "
                f"threshold={threshold:.4f}, "
                f"predicted_fraud={int((probs >= threshold).sum())}")
    return probs, threshold


def predict_ensemble(X_test: np.ndarray, feature_names: np.ndarray) -> np.ndarray:
    """Generate predictions using the stacking ensemble."""
    artifact = load_artifact(MODEL_DIR / "ensemble.pkl")
    meta = artifact["meta_learner"]
    base_models = artifact["base_models"]
    threshold = artifact["threshold"]
    model_feats = artifact["feature_names"]

    # Align features
    feat_idx = [np.where(feature_names == f)[0][0] for f in model_feats
                if f in feature_names]
    X_aligned = X_test[:, feat_idx]

    # Get base model predictions
    xgb_preds = base_models["xgboost_final"].predict_proba(X_aligned)[:, 1]

    if artifact.get("lgb_available") and "lightgbm_final" in base_models:
        lgb_preds = base_models["lightgbm_final"].predict_proba(X_aligned)[:, 1]
    else:
        lgb_preds = xgb_preds

    if artifact.get("cb_available") and "catboost_final" in base_models:
        cb_preds = base_models["catboost_final"].predict_proba(X_aligned)[:, 1]
    else:
        cb_preds = xgb_preds

    stack = np.column_stack([xgb_preds, lgb_preds, cb_preds])
    probs = meta.predict_proba(stack)[:, 1]

    logger.info(f"Ensemble predictions: mean_prob={probs.mean():.4f}, "
                f"threshold={threshold:.4f}, "
                f"predicted_fraud={int((probs >= threshold).sum())}")
    return probs, threshold


def main():
    set_global_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    use_baseline = "--baseline" in sys.argv
    use_neo4j = "--neo4j" in sys.argv

    logger.info("Building test features...")
    data = build_scaled_features(use_neo4j=use_neo4j)

    X_test = data["X_test"]
    test_providers = data["test_providers"]
    feature_names = data["feature_names"]

    if use_baseline:
        probs, threshold = predict_baseline(X_test, feature_names)
        model_name = "baseline"
    else:
        try:
            probs, threshold = predict_ensemble(X_test, feature_names)
            model_name = "ensemble"
        except FileNotFoundError:
            logger.warning("Ensemble not found, falling back to baseline")
            probs, threshold = predict_baseline(X_test, feature_names)
            model_name = "baseline"

    # Build output DataFrame
    results = pd.DataFrame({
        "Provider": test_providers,
        "FraudProbability": probs.round(6),
        "PredictedFraud": (probs >= threshold).astype(int),
        "PredictedLabel": np.where(probs >= threshold, "Yes", "No"),
    })

    # Sort by fraud probability descending (most suspicious first)
    results = results.sort_values("FraudProbability", ascending=False).reset_index(drop=True)

    # Save predictions
    out_path = OUTPUT_DIR / f"test_predictions_{model_name}.csv"
    results.to_csv(out_path, index=False)
    logger.info(f"Saved {len(results)} predictions to {out_path}")

    # Summary stats
    n_fraud = results["PredictedFraud"].sum()
    logger.info(f"Predicted fraud providers: {n_fraud}/{len(results)} "
                f"({n_fraud/len(results):.1%})")

    # Save prediction summary
    summary = {
        "model": model_name,
        "threshold": float(threshold),
        "n_test_providers": len(results),
        "n_predicted_fraud": int(n_fraud),
        "predicted_fraud_rate": round(n_fraud / len(results), 4),
        "mean_probability": round(float(probs.mean()), 4),
        "median_probability": round(float(np.median(probs)), 4),
    }
    with open(OUTPUT_DIR / f"prediction_summary_{model_name}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPredictions saved to: {out_path}")
    print(f"Predicted fraud: {n_fraud}/{len(results)} ({n_fraud/len(results):.1%})")

    return results


if __name__ == "__main__":
    main()
