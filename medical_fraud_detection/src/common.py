from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
import json
import os

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


@dataclass
class ModelArtifact:
    model_type: str
    backend: str
    model: Any
    preprocessor: Any
    feature_names: List[str]
    raw_feature_names: List[str]
    threshold: float
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    secondary_model: Any = None


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config.setdefault("project", {})
    config.setdefault("paths", {})
    config.setdefault("data_sources", {})
    config.setdefault("neo4j", {})

    config_dir = config_path.parent
    root_value = os.getenv("MEDICAL_FRAUD_ROOT_DIR") or config["project"].get("root_dir") or ".."
    root_dir = _resolve_path(root_value, config_dir)
    config["project"]["root_dir"] = str(root_dir)

    for key, value in config["paths"].items():
        config["paths"][key] = str(_resolve_path(value, root_dir))
    for key, value in config["data_sources"].items():
        config["data_sources"][key] = str(_resolve_path(value, root_dir))

    env_overrides = {
        "uri": os.getenv("NEO4J_URI"),
        "user": os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME"),
        "password": os.getenv("NEO4J_PASSWORD"),
        "database": os.getenv("NEO4J_DATABASE"),
    }
    for key, value in env_overrides.items():
        if value:
            config["neo4j"][key] = value
    return config


def _resolve_path(value: str | Path, base_dir: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path.resolve()


def ensure_project_dirs(config: Dict[str, Any]) -> None:
    for path_value in config.get("paths", {}).values():
        Path(path_value).mkdir(parents=True, exist_ok=True)


def project_root(config: Dict[str, Any]) -> Path:
    return Path(config["project"]["root_dir"]).resolve()


def set_matplotlib_cache(config: Dict[str, Any]) -> None:
    mpl_dir = project_root(config) / ".mplconfig"
    cache_dir = project_root(config) / ".cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


def save_artifact(artifact: ModelArtifact, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, target)


def load_artifact(path: str | Path) -> ModelArtifact:
    return joblib.load(Path(path))


def dump_json(payload: Dict[str, Any], path: str | Path) -> None:
    serializable = json.loads(json.dumps(payload, default=_json_default))
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)


def _json_default(value: Any) -> Any:
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def select_threshold_for_recall(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    target_recall: float,
) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return 0.5

    thresholds = np.unique(np.round(y_prob, 6))
    thresholds = np.concatenate(([0.0], thresholds, [1.0]))
    candidates: List[tuple[float, float, float]] = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        candidates.append((threshold, recall, f1 if recall >= target_recall else -1.0))

    meeting_target = [item for item in candidates if item[1] >= target_recall]
    if meeting_target:
        return max(meeting_target, key=lambda item: (item[2], -item[0]))[0]
    return min(candidates, key=lambda item: (-item[1], -item[2], item[0]))[0]


def compute_classification_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }

    if len(np.unique(y_true)) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["log_loss"] = float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6)))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["log_loss"] = float("nan")
    return metrics


def get_feature_names(preprocessor: Any) -> List[str]:
    if hasattr(preprocessor, "get_feature_names_out"):
        return [str(name) for name in preprocessor.get_feature_names_out()]
    return []


def predict_with_artifact(
    artifact: ModelArtifact,
    frame: pd.DataFrame,
    batch_size: int = 64,
) -> np.ndarray:
    missing = sorted(set(artifact.raw_feature_names) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    transformed = artifact.preprocessor.transform(frame[artifact.raw_feature_names])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    primary = _predict_model_proba(
        model=artifact.model,
        backend=artifact.backend,
        features=transformed,
        batch_size=batch_size,
        best_iteration=artifact.metadata.get("best_iteration"),
    )
    if artifact.secondary_model is None or not artifact.metadata.get("use_soft_voting", False):
        return primary

    secondary = _predict_sklearn_proba(artifact.secondary_model, transformed)
    weight = float(artifact.metadata.get("secondary_model_weight", 0.5))
    return ((1.0 - weight) * primary) + (weight * secondary)


def _predict_model_proba(
    model: Any,
    backend: str,
    features: np.ndarray,
    batch_size: int,
    best_iteration: int | None = None,
) -> np.ndarray:
    if backend == "lightgbm":
        outputs = []
        for start in range(0, len(features), batch_size):
            batch = features[start : start + batch_size]
            outputs.append(model.predict(batch, num_iteration=best_iteration))
        return np.concatenate(outputs)
    return _predict_sklearn_proba(model, features)


def _predict_sklearn_proba(model: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        return 1.0 / (1.0 + np.exp(-scores))
    raise AttributeError("Model does not expose probability predictions.")


def latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    checkpoint_dir = Path(checkpoint_dir)
    candidates = sorted(checkpoint_dir.glob("fraud_model_iter_*.pkl"))
    return candidates[-1] if candidates else None


def append_training_log(log_row: Dict[str, Any], log_path: str | Path) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([log_row])
    if log_path.exists():
        row_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        row_df.to_csv(log_path, index=False)


def minmax_scale(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return array
    if np.isclose(array.max(), array.min()):
        return np.zeros_like(array)
    return (array - array.min()) / (array.max() - array.min())
