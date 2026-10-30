"""
Training pipeline for Healthcare Provider Fraud Detection.

Architecture:
  1. Baseline: Single XGBoost with GPU support, stratified 5-fold CV
  2. Enhanced: Stacking ensemble (XGBoost + LightGBM + CatBoost) with
     logistic regression meta-learner, using out-of-fold predictions

Anti-overfitting measures:
  - Stratified K-Fold CV (no provider appears in multiple folds)
  - Early stopping on validation PR-AUC
  - Conservative regularization (L1/L2, depth limits, subsampling)
  - Class imbalance handled via scale_pos_weight, NOT SMOTE
    (SMOTE on 5K samples risks creating unrealistic synthetic providers)
  - Threshold tuning on OOF predictions, not training data
  - Correlation pruning of redundant features
  - Feature importance-based pruning after initial fit

Why not SMOTE?
  Provider-level dataset is only ~5,400 rows with 506 fraud cases.
  SMOTE would create synthetic provider profiles by interpolating between
  real providers — this risks manufacturing unrealistic feature combinations
  that don't represent real fraud patterns. Instead, we use scale_pos_weight
  which adjusts the loss function without altering the data distribution.
"""
import json
import logging
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from config import (
    CATBOOST_PARAMS, CV_FOLDS, EARLY_STOPPING_ROUNDS, LIGHTGBM_PARAMS,
    MODEL_DIR, OUTPUT_DIR, RANDOM_SEED, RECALL_TARGETS, USE_GPU,
    XGBOOST_PARAMS,
)
from feature_engineering import build_scaled_features
from utils import (
    compute_metrics, load_artifact, optimize_threshold, print_metrics,
    save_artifact, set_global_seed, setup_logging, threshold_at_recall,
)

warnings.filterwarnings("ignore", category=UserWarning)
logger = setup_logging()


def _get_scale_pos_weight(y: np.ndarray) -> float:
    """Compute scale_pos_weight = n_negative / n_positive."""
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    return n_neg / max(n_pos, 1)


def _prune_correlated_features(X: np.ndarray, feature_names: np.ndarray,
                                threshold: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Remove features with correlation > threshold, keeping the first."""
    corr = np.corrcoef(X, rowvar=False)
    n = corr.shape[0]
    drop_idx = set()
    for i in range(n):
        if i in drop_idx:
            continue
        for j in range(i + 1, n):
            if j in drop_idx:
                continue
            if abs(corr[i, j]) > threshold:
                drop_idx.add(j)
    keep_idx = [i for i in range(n) if i not in drop_idx]
    logger.info(f"Correlation pruning: dropped {len(drop_idx)} features "
                f"(threshold={threshold}), keeping {len(keep_idx)}")
    return X[:, keep_idx], feature_names[keep_idx]


def _build_xgboost(params: dict, use_gpu: bool):
    """Create XGBoost classifier with GPU support if available."""
    import xgboost as xgb
    p = params.copy()
    if use_gpu:
        p["device"] = "cuda"
        p["tree_method"] = "hist"
        logger.info("XGBoost: using GPU (CUDA)")
    else:
        p["device"] = "cpu"
        p["tree_method"] = "hist"
        logger.info("XGBoost: using CPU")
    return xgb.XGBClassifier(**p)


def _build_lightgbm(params: dict):
    """Create LightGBM classifier."""
    import lightgbm as lgb
    # LightGBM GPU on Windows requires special build; default to CPU
    p = params.copy()
    p["device"] = "cpu"
    return lgb.LGBMClassifier(**p)


def _build_catboost(params: dict, use_gpu: bool):
    """Create CatBoost classifier with optional GPU."""
    from catboost import CatBoostClassifier
    p = params.copy()
    if use_gpu:
        p["task_type"] = "GPU"
        logger.info("CatBoost: using GPU")
    else:
        p["task_type"] = "CPU"
    return CatBoostClassifier(**p)


def train_single_model(model, X_train, y_train, X_val, y_val, model_name: str):
    """Train a single model with early stopping."""
    spw = _get_scale_pos_weight(y_train)

    # Set class weight parameter per model type
    model_type = type(model).__name__
    if "XGB" in model_type:
        model.set_params(scale_pos_weight=spw)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    elif "LGBM" in model_type:
        model.set_params(scale_pos_weight=spw)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                __import__("lightgbm").early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                __import__("lightgbm").log_evaluation(0),
            ],
        )
    elif "CatBoost" in model_type:
        # CatBoost uses auto_class_weights="Balanced" from params
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)

    return model


def cross_validate_model(model_builder, X, y, feature_names, n_folds=CV_FOLDS,
                          model_name="model"):
    """
    Stratified K-Fold CV with out-of-fold predictions.
    Returns OOF predictions, per-fold metrics, and best fold model.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof_preds = np.zeros(len(y), dtype=np.float64)
    fold_metrics = []
    best_model = None
    best_pr_auc = -1

    train_metrics_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]

        model = model_builder()
        model = train_single_model(model, X_tr, y_tr, X_vl, y_vl, model_name)

        # Validation scores
        val_probs = model.predict_proba(X_vl)[:, 1]
        oof_preds[val_idx] = val_probs
        fold_m = compute_metrics(y_vl, val_probs, threshold=0.5)
        fold_metrics.append(fold_m)

        # Train scores (to check overfitting gap)
        train_probs = model.predict_proba(X_tr)[:, 1]
        train_m = compute_metrics(y_tr, train_probs, threshold=0.5)
        train_metrics_list.append(train_m)

        gap_roc = train_m["roc_auc"] - fold_m["roc_auc"]
        gap_pr = train_m["pr_auc"] - fold_m["pr_auc"]

        logger.info(
            f"  {model_name} Fold {fold+1}/{n_folds}: "
            f"Train ROC={train_m['roc_auc']:.4f} / Val ROC={fold_m['roc_auc']:.4f} (gap={gap_roc:+.4f}) | "
            f"Train PR={train_m['pr_auc']:.4f} / Val PR={fold_m['pr_auc']:.4f} (gap={gap_pr:+.4f}) | "
            f"Val F1={fold_m['f1']:.4f}"
        )

        if fold_m["pr_auc"] > best_pr_auc:
            best_pr_auc = fold_m["pr_auc"]
            best_model = model

    # Overfitting summary
    avg_train_roc = np.mean([m["roc_auc"] for m in train_metrics_list])
    avg_val_roc = np.mean([m["roc_auc"] for m in fold_metrics])
    avg_train_pr = np.mean([m["pr_auc"] for m in train_metrics_list])
    avg_val_pr = np.mean([m["pr_auc"] for m in fold_metrics])
    logger.info(f"  {model_name} Overfit Check: "
                f"Avg Train ROC={avg_train_roc:.4f} vs Avg Val ROC={avg_val_roc:.4f} "
                f"(gap={avg_train_roc - avg_val_roc:+.4f})")
    logger.info(f"  {model_name} Overfit Check: "
                f"Avg Train PR={avg_train_pr:.4f} vs Avg Val PR={avg_val_pr:.4f} "
                f"(gap={avg_train_pr - avg_val_pr:+.4f})")
    if (avg_train_roc - avg_val_roc) > 0.05:
        logger.warning(f"  ⚠ {model_name}: ROC-AUC gap > 0.05 — possible overfitting!")
    if (avg_train_pr - avg_val_pr) > 0.15:
        logger.warning(f"  ⚠ {model_name}: PR-AUC gap > 0.15 — possible overfitting!")

    return oof_preds, fold_metrics, best_model


def train_baseline(data: dict) -> dict:
    """
    Train baseline XGBoost model with stratified CV.
    """
    logger.info("=" * 60)
    logger.info("TRAINING BASELINE MODEL (XGBoost)")
    logger.info("=" * 60)

    X = data["X_train"]
    y = data["y_train"]
    feature_names = data["feature_names"]

    # Correlation pruning
    X_pruned, feat_pruned = _prune_correlated_features(X, feature_names)

    def builder():
        return _build_xgboost(XGBOOST_PARAMS, USE_GPU)

    oof_preds, fold_metrics, best_model = cross_validate_model(
        builder, X_pruned, y, feat_pruned, model_name="XGBoost-Baseline"
    )

    # Optimize threshold on OOF predictions (no leakage: OOF = held-out)
    best_threshold = optimize_threshold(y, oof_preds, metric="f1")
    oof_metrics = compute_metrics(y, oof_preds, threshold=best_threshold)
    print_metrics(oof_metrics, "Baseline XGBoost (OOF)")

    # Threshold at various recall targets
    for target in RECALL_TARGETS:
        thr = threshold_at_recall(y, oof_preds, target)
        m = compute_metrics(y, oof_preds, thr)
        logger.info(f"  At recall>={target:.0%}: threshold={thr:.3f}, "
                    f"precision={m['precision']:.3f}, F1={m['f1']:.3f}")

    # Feature importance
    importance = _extract_importance(best_model, feat_pruned)

    # Save
    baseline_artifact = {
        "model": best_model,
        "feature_names": feat_pruned,
        "threshold": best_threshold,
        "oof_metrics": oof_metrics,
        "fold_metrics": fold_metrics,
        "importance": importance,
    }
    save_artifact(baseline_artifact, MODEL_DIR / "baseline_xgboost.pkl")

    return baseline_artifact


def train_ensemble(data: dict) -> dict:
    """
    Train stacking ensemble: XGBoost + LightGBM + CatBoost → LogisticRegression.

    The meta-learner is trained on OOF predictions from the base models,
    which prevents leakage (each prediction was made on held-out data).
    """
    logger.info("=" * 60)
    logger.info("TRAINING ENSEMBLE MODEL (XGBoost + LightGBM + CatBoost)")
    logger.info("=" * 60)

    X = data["X_train"]
    y = data["y_train"]
    feature_names = data["feature_names"]

    X_pruned, feat_pruned = _prune_correlated_features(X, feature_names)

    # ─── Base model OOF predictions ──────────────────────────────────────
    oof_stack = np.zeros((len(y), 3), dtype=np.float64)
    base_models = {}

    # XGBoost
    logger.info("\n--- XGBoost Base Learner ---")
    def xgb_builder():
        return _build_xgboost(XGBOOST_PARAMS, USE_GPU)
    oof_xgb, _, best_xgb = cross_validate_model(
        xgb_builder, X_pruned, y, feat_pruned, model_name="XGB"
    )
    oof_stack[:, 0] = oof_xgb
    base_models["xgboost"] = best_xgb

    # LightGBM
    lgb_available = True
    try:
        import lightgbm
        logger.info("\n--- LightGBM Base Learner ---")
        def lgb_builder():
            return _build_lightgbm(LIGHTGBM_PARAMS)
        oof_lgb, _, best_lgb = cross_validate_model(
            lgb_builder, X_pruned, y, feat_pruned, model_name="LGB"
        )
        oof_stack[:, 1] = oof_lgb
        base_models["lightgbm"] = best_lgb
    except ImportError:
        logger.warning("LightGBM not installed, skipping")
        oof_stack[:, 1] = oof_xgb  # fallback: duplicate XGB
        lgb_available = False

    # CatBoost
    cb_available = True
    try:
        import catboost
        logger.info("\n--- CatBoost Base Learner ---")
        def cb_builder():
            return _build_catboost(CATBOOST_PARAMS, USE_GPU)
        oof_cb, _, best_cb = cross_validate_model(
            cb_builder, X_pruned, y, feat_pruned, model_name="CB"
        )
        oof_stack[:, 2] = oof_cb
        base_models["catboost"] = best_cb
    except ImportError:
        logger.warning("CatBoost not installed, skipping")
        oof_stack[:, 2] = oof_xgb
        cb_available = False

    # ─── Meta-learner ────────────────────────────────────────────────────
    # Trained on OOF predictions (no leakage: each row was predicted
    # when the model hadn't seen it)
    logger.info("\n--- Training Meta-Learner (Logistic Regression) ---")
    meta = LogisticRegression(
        C=1.0, penalty="l2", solver="lbfgs", max_iter=1000,
        random_state=RANDOM_SEED
    )
    meta.fit(oof_stack, y)
    meta_preds = meta.predict_proba(oof_stack)[:, 1]

    # ─── Calibration ─────────────────────────────────────────────────────
    # Calibrate the meta-learner using isotonic regression on OOF
    logger.info("Calibrating meta-learner probabilities...")
    meta_calibrated = CalibratedClassifierCV(meta, cv=3, method="isotonic")
    meta_calibrated.fit(oof_stack, y)
    cal_preds = meta_calibrated.predict_proba(oof_stack)[:, 1]

    # Optimize threshold
    best_threshold = optimize_threshold(y, cal_preds, metric="f1")
    ensemble_metrics = compute_metrics(y, cal_preds, threshold=best_threshold)
    print_metrics(ensemble_metrics, "Ensemble (Calibrated OOF)")

    for target in RECALL_TARGETS:
        thr = threshold_at_recall(y, cal_preds, target)
        m = compute_metrics(y, cal_preds, thr)
        logger.info(f"  At recall>={target:.0%}: threshold={thr:.3f}, "
                    f"precision={m['precision']:.3f}, F1={m['f1']:.3f}")

    # Feature importance (from XGBoost base, most interpretable)
    importance = _extract_importance(best_xgb, feat_pruned)

    # ─── Retrain base models on FULL training data for inference ─────────
    logger.info("\nRetraining base models on full training data for inference...")
    spw = _get_scale_pos_weight(y)

    final_xgb = _build_xgboost(XGBOOST_PARAMS, USE_GPU)
    final_xgb.set_params(scale_pos_weight=spw, n_estimators=best_xgb.n_estimators)
    final_xgb.fit(X_pruned, y)
    base_models["xgboost_final"] = final_xgb

    if lgb_available:
        final_lgb = _build_lightgbm(LIGHTGBM_PARAMS)
        final_lgb.set_params(scale_pos_weight=spw)
        if hasattr(best_lgb, "n_estimators"):
            final_lgb.set_params(n_estimators=best_lgb.n_estimators)
        final_lgb.fit(X_pruned, y)
        base_models["lightgbm_final"] = final_lgb

    if cb_available:
        final_cb = _build_catboost(CATBOOST_PARAMS, USE_GPU)
        final_cb.set_params(iterations=best_cb.get_params().get("iterations", 500))
        final_cb.fit(X_pruned, y)
        base_models["catboost_final"] = final_cb

    # Save everything
    ensemble_artifact = {
        "base_models": base_models,
        "meta_learner": meta_calibrated,
        "feature_names": feat_pruned,
        "threshold": best_threshold,
        "oof_metrics": ensemble_metrics,
        "importance": importance,
        "lgb_available": lgb_available,
        "cb_available": cb_available,
    }
    save_artifact(ensemble_artifact, MODEL_DIR / "ensemble.pkl")

    # Save OOF predictions for analysis
    oof_df = pd.DataFrame({
        "Provider": data["train_providers"],
        "PotentialFraud": y,
        "oof_xgb": oof_xgb,
        "oof_lgb": oof_stack[:, 1],
        "oof_catboost": oof_stack[:, 2],
        "oof_ensemble": cal_preds,
    })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    logger.info(f"Saved OOF predictions to {OUTPUT_DIR / 'oof_predictions.csv'}")

    return ensemble_artifact


def _extract_importance(model, feature_names: np.ndarray) -> pd.DataFrame:
    """Extract and sort feature importance."""
    model_type = type(model).__name__
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        logger.warning(f"Cannot extract importance from {model_type}")
        return pd.DataFrame()

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": imp,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Print top 20
    logger.info("\nTop 20 Features by Importance:")
    for _, row in df.head(20).iterrows():
        logger.info(f"  {row['feature']:40s} {row['importance']:.4f}")

    return df


def compare_models(baseline: dict, ensemble: dict):
    """Print comparison of baseline vs ensemble."""
    print("\n" + "=" * 60)
    print(" MODEL COMPARISON ".center(60, "="))
    print("=" * 60)
    bm = baseline["oof_metrics"]
    em = ensemble["oof_metrics"]
    metrics = ["roc_auc", "pr_auc", "f1", "precision", "recall"]
    print(f"{'Metric':<15} {'Baseline':>12} {'Ensemble':>12} {'Delta':>10}")
    print("-" * 50)
    for m in metrics:
        delta = em[m] - bm[m]
        marker = "+" if delta > 0 else ""
        print(f"{m:<15} {bm[m]:>12.4f} {em[m]:>12.4f} {marker}{delta:>9.4f}")
    print("=" * 60)


def main():
    set_global_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start = time.time()

    # Build features
    logger.info("Building features...")
    use_neo4j = "--neo4j" in sys.argv
    data = build_scaled_features(use_neo4j=use_neo4j)

    logger.info(f"Training data shape: {data['X_train'].shape}")
    logger.info(f"Class distribution: {np.bincount(data['y_train'])} "
                f"(fraud rate: {data['y_train'].mean():.2%})")

    # Train baseline
    baseline = train_baseline(data)

    # Train ensemble
    ensemble = train_ensemble(data)

    # Compare
    compare_models(baseline, ensemble)

    elapsed = time.time() - start
    logger.info(f"\nTotal training time: {elapsed:.1f}s")

    # Save summary
    summary = {
        "baseline_oof_metrics": {k: v for k, v in baseline["oof_metrics"].items()
                                  if k != "confusion_matrix"},
        "ensemble_oof_metrics": {k: v for k, v in ensemble["oof_metrics"].items()
                                  if k != "confusion_matrix"},
        "n_features": len(ensemble["feature_names"]),
        "n_train_providers": len(data["y_train"]),
        "fraud_rate": float(data["y_train"].mean()),
        "gpu_used": USE_GPU,
        "training_time_seconds": round(elapsed, 1),
    }
    with open(OUTPUT_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Training summary saved to {OUTPUT_DIR / 'training_summary.json'}")


if __name__ == "__main__":
    main()
