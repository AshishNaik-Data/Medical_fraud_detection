"""
Shared utilities: logging, seed control, artifact persistence, metrics.
"""
import logging
import random
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score,
)

from config import LOG_DIR, RANDOM_SEED


def setup_logging(name: str = "fraud_detection", level=logging.INFO) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def set_global_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def save_artifact(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: Path):
    return joblib.load(path)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute comprehensive classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "threshold": threshold,
    }
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1") -> float:
    """Find the threshold that maximizes F1 or balances precision/recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # F1 from precision-recall curve
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    if best_idx < len(thresholds):
        return float(thresholds[best_idx])
    return 0.5


def threshold_at_recall(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float) -> float:
    """Find highest threshold that achieves at least target_recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # recall is decreasing; find where it first drops below target
    valid = recall >= target_recall
    if valid.any():
        # Pick the threshold with highest precision among those meeting recall target
        idx = np.where(valid)[0]
        # Among valid indices, pick the one with best F1
        f1s = 2 * (precision[idx] * recall[idx]) / (precision[idx] + recall[idx] + 1e-8)
        best = idx[np.argmax(f1s)]
        if best < len(thresholds):
            return float(thresholds[best])
    return 0.5


def print_metrics(metrics: dict, prefix: str = ""):
    header = f" {prefix} " if prefix else ""
    print(f"\n{'='*60}")
    print(f"{header:=^60}")
    print(f"{'='*60}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC    : {metrics['pr_auc']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Threshold : {metrics['threshold']:.4f}")
    cm = np.array(metrics["confusion_matrix"])
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"    FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
    print(f"{'='*60}\n")
