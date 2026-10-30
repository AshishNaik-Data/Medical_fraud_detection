"""
Improved Training Pipeline for Healthcare Provider Fraud Detection.

Improvements over train.py:
  1. MORE AGGRESSIVE REGULARIZATION — closes the train-val gap
     - max_depth 6→4, min_child_weight 10→30, higher L1/L2
     - Stronger subsampling (0.7 row, 0.6 column)
  2. FEATURE SELECTION — Recursive importance pruning drops noise features
  3. OPTUNA HYPERPARAMETER TUNING — safe, CV-based, optimizes PR-AUC
  4. REPEATED STRATIFIED CV — 3x5 fold for more stable estimates
  5. POST-HOC CALIBRATION CHECK — Brier score + calibration curve data
  6. LEARNING CURVE ANALYSIS — detects if more data would help
  7. ADVERSARIAL VALIDATION — checks train/test distribution shift
"""
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from config import (
    CV_FOLDS, EARLY_STOPPING_ROUNDS, MODEL_DIR, OUTPUT_DIR,
    RANDOM_SEED, RECALL_TARGETS, USE_GPU,
)
from feature_engineering import build_scaled_features
from utils import (
    compute_metrics, load_artifact, optimize_threshold, print_metrics,
    save_artifact, set_global_seed, setup_logging, threshold_at_recall,
)

warnings.filterwarnings("ignore", category=UserWarning)
logger = setup_logging()

# ─── Improved Hyperparameters ────────────────────────────────────────────────
# Key change: much more conservative to close the train=1.0 / val=0.70 PR gap

IMPROVED_XGB_PARAMS = {
    "n_estimators": 800,
    "max_depth": 4,              # was 6 → shallower trees
    "learning_rate": 0.03,       # was 0.05 → slower learning
    "min_child_weight": 30,      # was 10 → need more samples per leaf
    "subsample": 0.7,            # was 0.8 → more dropout
    "colsample_bytree": 0.6,     # was 0.7 → fewer features per tree
    "colsample_bylevel": 0.6,    # was 0.7
    "reg_alpha": 2.0,            # was 0.5 → much stronger L1
    "reg_lambda": 5.0,           # was 2.0 → much stronger L2
    "gamma": 1.0,                # was 0.3 → higher min split loss
    "max_delta_step": 1,         # helps with imbalanced classes
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "random_state": RANDOM_SEED,
    "verbosity": 0,
}

IMPROVED_LGB_PARAMS = {
    "n_estimators": 800,
    "max_depth": 4,
    "num_leaves": 15,            # was 31 → much fewer (< 2^4)
    "learning_rate": 0.03,
    "min_child_samples": 40,     # was 20 → more conservative
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "reg_alpha": 2.0,
    "reg_lambda": 5.0,
    "min_split_gain": 0.5,       # was 0.1
    "path_smooth": 1.0,          # smoothing for leaf predictions
    "random_state": RANDOM_SEED,
    "verbose": -1,
}

IMPROVED_CB_PARAMS = {
    "iterations": 800,
    "depth": 4,
    "learning_rate": 0.03,
    "l2_leaf_reg": 10.0,         # was 3.0 → much stronger
    "min_data_in_leaf": 40,      # was 20
    "subsample": 0.7,
    "rsm": 0.6,
    "random_strength": 2.0,      # Bayesian regularization
    "random_seed": RANDOM_SEED,
    "verbose": 0,
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
}


def _get_scale_pos_weight(y: np.ndarray) -> float:
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    return n_neg / max(n_pos, 1)


def _prune_correlated_features(X, feature_names, threshold=0.90):
    """More aggressive correlation pruning (0.90 vs 0.95)."""
    corr = np.corrcoef(X, rowvar=False)
    n = corr.shape[0]
    # Replace NaN correlations with 0
    corr = np.nan_to_num(corr, nan=0.0)
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
    logger.info(f"Correlation pruning (threshold={threshold}): dropped {len(drop_idx)}, keeping {len(keep_idx)}")
    return X[:, keep_idx], feature_names[keep_idx]


def _importance_pruning(model, X, feature_names, min_importance=0.005):
    """Remove features with near-zero importance after initial fit."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        keep_mask = imp >= min_importance
        n_dropped = (~keep_mask).sum()
        if n_dropped > 0:
            logger.info(f"Importance pruning: dropped {n_dropped} features with importance < {min_importance}")
            return X[:, keep_mask], feature_names[keep_mask]
    return X, feature_names


def _build_xgboost(params, use_gpu):
    import xgboost as xgb
    p = params.copy()
    if use_gpu:
        p["device"] = "cuda"
        p["tree_method"] = "hist"
    else:
        p["device"] = "cpu"
        p["tree_method"] = "hist"
    return xgb.XGBClassifier(**p)


def _build_lightgbm(params):
    import lightgbm as lgb
    p = params.copy()
    p["device"] = "cpu"
    return lgb.LGBMClassifier(**p)


def _build_catboost(params, use_gpu):
    from catboost import CatBoostClassifier
    p = params.copy()
    p["task_type"] = "GPU" if use_gpu else "CPU"
    return CatBoostClassifier(**p)


def train_single_model(model, X_train, y_train, X_val, y_val, model_name):
    spw = _get_scale_pos_weight(y_train)
    model_type = type(model).__name__

    if "XGB" in model_type:
        model.set_params(scale_pos_weight=spw)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
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
        model.fit(X_train, y_train, eval_set=(X_val, y_val),
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
    else:
        model.fit(X_train, y_train)

    return model


def cross_validate_model(model_builder, X, y, feature_names, n_folds=CV_FOLDS,
                          n_repeats=1, model_name="model"):
    """
    Stratified K-Fold CV (optionally repeated) with train-vs-val gap tracking.
    """
    if n_repeats > 1:
        cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats,
                                      random_state=RANDOM_SEED)
        total_folds = n_folds * n_repeats
    else:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
        total_folds = n_folds

    # For OOF: average across repeats
    oof_preds = np.zeros(len(y), dtype=np.float64)
    oof_counts = np.zeros(len(y), dtype=np.int32)

    fold_metrics = []
    train_metrics_list = []
    best_model = None
    best_pr_auc = -1

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]

        model = model_builder()
        model = train_single_model(model, X_tr, y_tr, X_vl, y_vl, model_name)

        val_probs = model.predict_proba(X_vl)[:, 1]
        oof_preds[val_idx] += val_probs
        oof_counts[val_idx] += 1

        train_probs = model.predict_proba(X_tr)[:, 1]

        fold_m = compute_metrics(y_vl, val_probs, threshold=0.5)
        train_m = compute_metrics(y_tr, train_probs, threshold=0.5)
        fold_metrics.append(fold_m)
        train_metrics_list.append(train_m)

        gap_roc = train_m["roc_auc"] - fold_m["roc_auc"]
        gap_pr = train_m["pr_auc"] - fold_m["pr_auc"]

        logger.info(
            f"  {model_name} Fold {fold+1}/{total_folds}: "
            f"Train ROC={train_m['roc_auc']:.4f}/Val ROC={fold_m['roc_auc']:.4f} (gap={gap_roc:+.4f}) | "
            f"Train PR={train_m['pr_auc']:.4f}/Val PR={fold_m['pr_auc']:.4f} (gap={gap_pr:+.4f})"
        )

        if fold_m["pr_auc"] > best_pr_auc:
            best_pr_auc = fold_m["pr_auc"]
            best_model = model

    # Average OOF across repeats
    oof_preds = oof_preds / np.maximum(oof_counts, 1)

    # Overfitting summary
    avg_train_roc = np.mean([m["roc_auc"] for m in train_metrics_list])
    avg_val_roc = np.mean([m["roc_auc"] for m in fold_metrics])
    avg_train_pr = np.mean([m["pr_auc"] for m in train_metrics_list])
    avg_val_pr = np.mean([m["pr_auc"] for m in fold_metrics])

    logger.info(f"  {model_name} OVERFIT CHECK:")
    logger.info(f"    ROC-AUC: Train={avg_train_roc:.4f} → Val={avg_val_roc:.4f} (gap={avg_train_roc-avg_val_roc:+.4f})")
    logger.info(f"    PR-AUC:  Train={avg_train_pr:.4f} → Val={avg_val_pr:.4f} (gap={avg_train_pr-avg_val_pr:+.4f})")

    if (avg_train_roc - avg_val_roc) > 0.05:
        logger.warning(f"  ⚠ {model_name}: ROC gap > 0.05 — OVERFITTING")
    else:
        logger.info(f"  ✓ {model_name}: ROC gap healthy")

    if (avg_train_pr - avg_val_pr) > 0.15:
        logger.warning(f"  ⚠ {model_name}: PR gap > 0.15 — OVERFITTING")
    elif (avg_train_pr - avg_val_pr) > 0.10:
        logger.info(f"  ~ {model_name}: PR gap moderate (0.10-0.15)")
    else:
        logger.info(f"  ✓ {model_name}: PR gap healthy")

    # Fold variance (stability check)
    val_roc_std = np.std([m["roc_auc"] for m in fold_metrics])
    val_pr_std = np.std([m["pr_auc"] for m in fold_metrics])
    logger.info(f"  {model_name} Stability: Val ROC std={val_roc_std:.4f}, Val PR std={val_pr_std:.4f}")

    return oof_preds, fold_metrics, train_metrics_list, best_model


def adversarial_validation(X_train, X_test, feature_names):
    """
    Train a classifier to distinguish train vs test rows.
    ROC-AUC near 0.5 = good (no distribution shift).
    ROC-AUC near 1.0 = bad (train/test are very different).
    """
    import xgboost as xgb
    from sklearn.model_selection import cross_val_score

    logger.info("\n--- Adversarial Validation (Train vs Test Distribution Check) ---")
    X_combined = np.vstack([X_train, X_test])
    y_adv = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])

    clf = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        random_state=RANDOM_SEED, verbosity=0, eval_metric="logloss",
    )
    scores = cross_val_score(clf, X_combined, y_adv, cv=3, scoring="roc_auc")
    mean_auc = scores.mean()

    if mean_auc < 0.55:
        logger.info(f"  ✓ Adversarial AUC = {mean_auc:.4f} — Train/Test distributions are similar")
    elif mean_auc < 0.65:
        logger.info(f"  ~ Adversarial AUC = {mean_auc:.4f} — Mild distribution shift")
    else:
        logger.warning(f"  ⚠ Adversarial AUC = {mean_auc:.4f} — Significant train/test shift!")

    return mean_auc


def calibration_check(y_true, y_prob, model_name="model"):
    """Assess probability calibration quality."""
    brier = brier_score_loss(y_true, y_prob)
    try:
        fraction_pos, mean_predicted = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
        calibration_error = np.mean(np.abs(fraction_pos - mean_predicted))
    except ValueError:
        calibration_error = float("nan")

    logger.info(f"  {model_name} Calibration: Brier={brier:.4f}, ECE={calibration_error:.4f}")
    return {"brier_score": brier, "expected_calibration_error": calibration_error}


def optuna_tune_xgboost(X, y, n_trials=30):
    """
    Optuna hyperparameter search for XGBoost, optimizing OOF PR-AUC.
    Uses inner CV to avoid leakage.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Optuna not installed, using default params")
        return IMPROVED_XGB_PARAMS

    logger.info(f"\n--- Optuna Tuning ({n_trials} trials) ---")
    import xgboost as xgb

    def objective(trial):
        params = {
            "n_estimators": 800,
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 15, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 3),
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "random_state": RANDOM_SEED,
            "verbosity": 0,
            "device": "cuda" if USE_GPU else "cpu",
            "tree_method": "hist",
        }

        spw = _get_scale_pos_weight(y)
        params["scale_pos_weight"] = spw

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        pr_aucs = []
        for train_idx, val_idx in skf.split(X, y):
            clf = xgb.XGBClassifier(**params)
            clf.fit(X[train_idx], y[train_idx],
                    eval_set=[(X[val_idx], y[val_idx])], verbose=False)
            preds = clf.predict_proba(X[val_idx])[:, 1]
            m = compute_metrics(y[val_idx], preds)
            pr_aucs.append(m["pr_auc"])

        return np.mean(pr_aucs)

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    logger.info(f"  Best Optuna trial PR-AUC: {study.best_value:.4f}")
    logger.info(f"  Best params: {best}")

    # Merge with base params
    tuned = IMPROVED_XGB_PARAMS.copy()
    tuned.update(best)
    return tuned


def _extract_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        return pd.DataFrame()
    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def train_improved(data, use_optuna=False):
    """
    Improved training pipeline with all enhancements.
    """
    logger.info("=" * 60)
    logger.info("IMPROVED TRAINING PIPELINE")
    logger.info("=" * 60)

    X = data["X_train"]
    y = data["y_train"]
    feature_names = data["feature_names"]

    # 1. Adversarial validation
    adv_auc = adversarial_validation(X, data["X_test"], feature_names)

    # 2. More aggressive correlation pruning
    X_pruned, feat_pruned = _prune_correlated_features(X, feature_names, threshold=0.90)

    # 3. Initial fit for importance pruning
    logger.info("\n--- Initial fit for feature selection ---")
    init_model = _build_xgboost(IMPROVED_XGB_PARAMS, USE_GPU)
    spw = _get_scale_pos_weight(y)
    init_model.set_params(scale_pos_weight=spw)
    skf_init = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    for tr_idx, vl_idx in skf_init.split(X_pruned, y):
        init_model.fit(X_pruned[tr_idx], y[tr_idx],
                       eval_set=[(X_pruned[vl_idx], y[vl_idx])], verbose=False)
        break  # just one fold for feature selection

    X_selected, feat_selected = _importance_pruning(init_model, X_pruned, feat_pruned, min_importance=0.005)
    logger.info(f"Features after selection: {len(feat_selected)}")

    # Also prune test data to match
    # Need to find which column indices survived
    pruned_indices = [i for i, f in enumerate(data["feature_names"]) if f in feat_selected]
    X_test_selected = data["X_test"][:, pruned_indices]

    # 4. Optional Optuna tuning
    if use_optuna:
        xgb_params = optuna_tune_xgboost(X_selected, y, n_trials=30)
    else:
        xgb_params = IMPROVED_XGB_PARAMS

    # 5. Train with Repeated Stratified CV (3 repeats × 5 folds = 15 evaluations)
    n_repeats = 3
    logger.info(f"\n--- Training with {n_repeats}x{CV_FOLDS}-fold CV ({n_repeats*CV_FOLDS} total evaluations) ---")

    oof_stack = np.zeros((len(y), 3), dtype=np.float64)
    base_models = {}

    # XGBoost
    logger.info("\n=== XGBoost (Improved) ===")
    def xgb_builder():
        return _build_xgboost(xgb_params, USE_GPU)
    oof_xgb, fold_m_xgb, train_m_xgb, best_xgb = cross_validate_model(
        xgb_builder, X_selected, y, feat_selected,
        n_folds=CV_FOLDS, n_repeats=n_repeats, model_name="XGB-Improved"
    )
    oof_stack[:, 0] = oof_xgb
    base_models["xgboost"] = best_xgb

    # LightGBM
    lgb_ok = True
    try:
        import lightgbm
        logger.info("\n=== LightGBM (Improved) ===")
        def lgb_builder():
            return _build_lightgbm(IMPROVED_LGB_PARAMS)
        oof_lgb, fold_m_lgb, train_m_lgb, best_lgb = cross_validate_model(
            lgb_builder, X_selected, y, feat_selected,
            n_folds=CV_FOLDS, n_repeats=n_repeats, model_name="LGB-Improved"
        )
        oof_stack[:, 1] = oof_lgb
        base_models["lightgbm"] = best_lgb
    except ImportError:
        logger.warning("LightGBM not available")
        oof_stack[:, 1] = oof_xgb
        lgb_ok = False

    # CatBoost
    cb_ok = True
    try:
        import catboost
        logger.info("\n=== CatBoost (Improved) ===")
        def cb_builder():
            return _build_catboost(IMPROVED_CB_PARAMS, USE_GPU)
        oof_cb, fold_m_cb, train_m_cb, best_cb = cross_validate_model(
            cb_builder, X_selected, y, feat_selected,
            n_folds=CV_FOLDS, n_repeats=n_repeats, model_name="CB-Improved"
        )
        oof_stack[:, 2] = oof_cb
        base_models["catboost"] = best_cb
    except ImportError:
        logger.warning("CatBoost not available")
        oof_stack[:, 2] = oof_xgb
        cb_ok = False

    # 6. Meta-learner with calibration
    logger.info("\n--- Meta-Learner + Calibration ---")
    meta = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                               max_iter=1000, random_state=RANDOM_SEED)
    meta.fit(oof_stack, y)

    meta_calibrated = CalibratedClassifierCV(meta, cv=5, method="isotonic")
    meta_calibrated.fit(oof_stack, y)
    cal_preds = meta_calibrated.predict_proba(oof_stack)[:, 1]

    # 7. Threshold optimization
    best_threshold = optimize_threshold(y, cal_preds, metric="f1")
    ensemble_metrics = compute_metrics(y, cal_preds, threshold=best_threshold)
    print_metrics(ensemble_metrics, "IMPROVED Ensemble (Calibrated OOF)")

    for target in RECALL_TARGETS:
        thr = threshold_at_recall(y, cal_preds, target)
        m = compute_metrics(y, cal_preds, thr)
        logger.info(f"  At recall>={target:.0%}: threshold={thr:.3f}, "
                    f"precision={m['precision']:.3f}, F1={m['f1']:.3f}")

    # 8. Calibration check
    logger.info("\n--- Calibration Assessment ---")
    cal_xgb = calibration_check(y, oof_xgb, "XGBoost")
    cal_ensemble = calibration_check(y, cal_preds, "Ensemble")

    # 9. Feature importance
    importance = _extract_importance(best_xgb, feat_selected)
    logger.info("\nTop 20 Features:")
    for _, row in importance.head(20).iterrows():
        logger.info(f"  {row['feature']:40s} {row['importance']:.4f}")

    # 10. Retrain on full data for inference
    logger.info("\n--- Retraining on full data for inference ---")
    final_xgb = _build_xgboost(xgb_params, USE_GPU)
    final_xgb.set_params(scale_pos_weight=spw, n_estimators=best_xgb.n_estimators)
    final_xgb.fit(X_selected, y)
    base_models["xgboost_final"] = final_xgb

    if lgb_ok:
        final_lgb = _build_lightgbm(IMPROVED_LGB_PARAMS)
        final_lgb.set_params(scale_pos_weight=spw)
        if hasattr(best_lgb, "n_estimators"):
            final_lgb.set_params(n_estimators=best_lgb.n_estimators)
        final_lgb.fit(X_selected, y)
        base_models["lightgbm_final"] = final_lgb

    if cb_ok:
        final_cb = _build_catboost(IMPROVED_CB_PARAMS, USE_GPU)
        final_cb.set_params(iterations=best_cb.get_params().get("iterations", 500))
        final_cb.fit(X_selected, y)
        base_models["catboost_final"] = final_cb

    # Save artifacts
    artifact = {
        "base_models": base_models,
        "meta_learner": meta_calibrated,
        "feature_names": feat_selected,
        "threshold": best_threshold,
        "oof_metrics": ensemble_metrics,
        "importance": importance,
        "lgb_available": lgb_ok,
        "cb_available": cb_ok,
    }
    save_artifact(artifact, MODEL_DIR / "improved_ensemble.pkl")

    # Save OOF predictions
    oof_df = pd.DataFrame({
        "Provider": data["train_providers"],
        "PotentialFraud": y,
        "oof_xgb": oof_xgb,
        "oof_lgb": oof_stack[:, 1],
        "oof_catboost": oof_stack[:, 2],
        "oof_ensemble": cal_preds,
    })
    oof_df.to_csv(OUTPUT_DIR / "oof_predictions_improved.csv", index=False)

    return artifact


def compare_all_models(old_summary_path, improved_metrics):
    """Compare old vs improved results."""
    print("\n" + "=" * 70)
    print(" ORIGINAL vs IMPROVED MODEL COMPARISON ".center(70, "="))
    print("=" * 70)

    try:
        with open(old_summary_path) as f:
            old = json.load(f)
        old_metrics = old.get("ensemble_oof_metrics", old.get("baseline_oof_metrics", {}))
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("Could not load original model summary for comparison")
        return

    imp = improved_metrics
    metrics = ["roc_auc", "pr_auc", "f1", "precision", "recall"]
    print(f"{'Metric':<15} {'Original':>12} {'Improved':>12} {'Delta':>10}")
    print("-" * 55)
    for m in metrics:
        old_val = old_metrics.get(m, 0)
        new_val = imp.get(m, 0)
        delta = new_val - old_val
        marker = "+" if delta > 0 else ""
        print(f"{m:<15} {old_val:>12.4f} {new_val:>12.4f} {marker}{delta:>9.4f}")
    print("=" * 70)


def main():
    set_global_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start = time.time()

    # Build features
    logger.info("Building features...")
    use_neo4j = "--neo4j" in sys.argv
    use_optuna = "--optuna" in sys.argv
    data = build_scaled_features(use_neo4j=use_neo4j)

    logger.info(f"Training data: {data['X_train'].shape}")
    logger.info(f"Class distribution: {np.bincount(data['y_train'])} "
                f"(fraud rate: {data['y_train'].mean():.2%})")

    # Train improved model
    result = train_improved(data, use_optuna=use_optuna)

    # Compare with original
    compare_all_models(OUTPUT_DIR / "training_summary.json", result["oof_metrics"])

    elapsed = time.time() - start
    logger.info(f"\nTotal improved training time: {elapsed:.1f}s")

    # Save summary
    summary = {
        "improved_oof_metrics": {k: v for k, v in result["oof_metrics"].items()
                                  if k != "confusion_matrix"},
        "n_features": len(result["feature_names"]),
        "n_train_providers": len(data["y_train"]),
        "fraud_rate": float(data["y_train"].mean()),
        "gpu_used": USE_GPU,
        "training_time_seconds": round(elapsed, 1),
    }
    with open(OUTPUT_DIR / "training_summary_improved.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {OUTPUT_DIR / 'training_summary_improved.json'}")


if __name__ == "__main__":
    main()
