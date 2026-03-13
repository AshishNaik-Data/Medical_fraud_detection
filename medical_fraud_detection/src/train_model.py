from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.common import (
    DEFAULT_CONFIG_PATH,
    ModelArtifact,
    append_training_log,
    compute_classification_metrics,
    dump_json,
    ensure_project_dirs,
    get_feature_names,
    latest_checkpoint,
    load_artifact,
    load_config,
    predict_with_artifact,
    save_artifact,
    select_threshold_for_recall,
)

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional dependency
    lgb = None


EXCLUDED_COLUMNS = {
    "claim_event_id",
    "claim_group_id",
    "patient_id",
    "doctor_id",
    "claim_time",
    "claim_date",
    "primary_diagnosis_description",
    "icd9_description",
    "icd10_description",
    "diagnosis_signature",
    "fraud_reason_summary",
    "triggered_rules",
    "target_label",
    "fraud_probability",
    "fraud_prediction",
    "data_split",
}


class FraudModelTrainer:
    """Train a checkpointed fraud model with a LightGBM-first strategy."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = Path(config_path)
        self.config = load_config(config_path)
        ensure_project_dirs(self.config)
        self.paths = {key: Path(value) for key, value in self.config["paths"].items()}
        self.model_config = self.config["model"]
        self.log_path = self.paths["logs_dir"] / "training_log.csv"

    def load_dataset(self) -> Tuple[pd.DataFrame, str]:
        dataset = pd.read_csv(self.paths["processed_dir"] / "modeling_dataset.csv", parse_dates=["claim_time"])
        target_column = self.model_config["target_column"]
        weak_label_column = self.model_config["weak_label_column"]

        if target_column in dataset.columns and dataset[target_column].notna().sum() > 0:
            target = target_column
        elif self.model_config["allow_weak_labels"]:
            target = weak_label_column
        else:
            raise ValueError(
                f"Target column '{target_column}' is missing and weak-label fallback is disabled."
            )

        dataset[target] = dataset[target].fillna(0).astype(int)
        return dataset, target

    def _dataset_signature(self, dataset: pd.DataFrame, target: str) -> Dict[str, object]:
        return {
            "row_count": int(len(dataset)),
            "positive_count": int(dataset[target].sum()),
            "target_column": target,
        }

    def _build_split(self, dataset: pd.DataFrame, target: str) -> pd.DataFrame:
        split_path = self.paths["processed_dir"] / "train_valid_split.csv"
        if split_path.exists():
            manifest = pd.read_csv(split_path)
            if set(manifest["claim_event_id"]) == set(dataset["claim_event_id"]):
                return dataset.merge(manifest, on="claim_event_id", how="left")

        stratify_labels = dataset[target] if dataset[target].nunique() > 1 else None
        train_ids, valid_ids = train_test_split(
            dataset["claim_event_id"],
            test_size=self.model_config["test_size"],
            stratify=stratify_labels,
            random_state=self.model_config["random_state"],
        )
        manifest = pd.DataFrame({"claim_event_id": dataset["claim_event_id"]})
        manifest["data_split"] = np.where(manifest["claim_event_id"].isin(valid_ids), "valid", "train")
        manifest.to_csv(split_path, index=False)
        return dataset.merge(manifest, on="claim_event_id", how="left")

    def _infer_feature_columns(self, dataset: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
        candidates = []
        for column in dataset.columns:
            if column in EXCLUDED_COLUMNS or column == target:
                continue
            if column.startswith("provisional_") or column.startswith("base_rule_") or column.startswith("final_rule_"):
                continue
            candidates.append(column)

        categorical = []
        numeric = []
        blocked_text = {"icd9_code", "icd10_code"}
        for column in candidates:
            if column in blocked_text:
                continue
            if pd.api.types.is_object_dtype(dataset[column]) or pd.api.types.is_string_dtype(dataset[column]):
                categorical.append(column)
            elif pd.api.types.is_datetime64_any_dtype(dataset[column]):
                continue
            else:
                numeric.append(column)
        return numeric, categorical

    def _build_preprocessor(self, dataset: pd.DataFrame, target: str) -> Tuple[ColumnTransformer, List[str]]:
        numeric_features, categorical_features = self._infer_feature_columns(dataset, target)
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                        ]
                    ),
                    numeric_features,
                ),
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "onehot",
                                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ),
                        ]
                    ),
                    categorical_features,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        raw_feature_names = numeric_features + categorical_features
        return preprocessor, raw_feature_names

    def _prepare_matrices(
        self,
        dataset: pd.DataFrame,
        target: str,
        resume_artifact: ModelArtifact | None = None,
    ) -> Dict[str, np.ndarray | pd.DataFrame | List[str] | ColumnTransformer]:
        split_dataset = self._build_split(dataset, target)
        train_df = split_dataset.loc[split_dataset["data_split"] == "train"].copy()
        valid_df = split_dataset.loc[split_dataset["data_split"] == "valid"].copy()

        if resume_artifact is not None:
            preprocessor = resume_artifact.preprocessor
            raw_feature_names = resume_artifact.raw_feature_names
        else:
            preprocessor, raw_feature_names = self._build_preprocessor(train_df, target)
            preprocessor.fit(train_df[raw_feature_names])

        x_train = preprocessor.transform(train_df[raw_feature_names])
        x_valid = preprocessor.transform(valid_df[raw_feature_names])
        if hasattr(x_train, "toarray"):
            x_train = x_train.toarray()
            x_valid = x_valid.toarray()

        feature_names = get_feature_names(preprocessor)
        if not feature_names:
            feature_names = [f"feature_{index}" for index in range(x_train.shape[1])]

        return {
            "split_dataset": split_dataset,
            "train_df": train_df,
            "valid_df": valid_df,
            "x_train": x_train,
            "x_valid": x_valid,
            "y_train": train_df[target].to_numpy(),
            "y_valid": valid_df[target].to_numpy(),
            "preprocessor": preprocessor,
            "raw_feature_names": raw_feature_names,
            "feature_names": feature_names,
        }

    def _resolve_backend(self) -> str:
        if self.model_config["primary_model"] == "lightgbm" and lgb is not None:
            return "lightgbm"
        if self.model_config.get("allow_fallback_model", True):
            return self.model_config["fallback_model"]
        raise RuntimeError("LightGBM is not installed and fallback training is disabled.")

    def _build_checkpoint_artifact(
        self,
        model,
        backend: str,
        preprocessor: ColumnTransformer,
        raw_feature_names: List[str],
        feature_names: List[str],
        threshold: float,
        metrics: Dict[str, float],
        metadata: Dict[str, object],
        secondary_model=None,
    ) -> ModelArtifact:
        return ModelArtifact(
            model_type="fraud_detection",
            backend=backend,
            model=model,
            preprocessor=preprocessor,
            feature_names=feature_names,
            raw_feature_names=raw_feature_names,
            threshold=threshold,
            metrics=metrics,
            metadata=metadata,
            secondary_model=secondary_model,
        )

    def _train_lightgbm(
        self,
        matrices: Dict[str, object],
        resume_artifact: ModelArtifact | None,
        dataset_signature: Dict[str, object],
    ) -> ModelArtifact:
        x_train = matrices["x_train"]
        x_valid = matrices["x_valid"]
        y_train = matrices["y_train"]
        y_valid = matrices["y_valid"]
        preprocessor = matrices["preprocessor"]
        raw_feature_names = matrices["raw_feature_names"]
        feature_names = matrices["feature_names"]

        train_dataset = lgb.Dataset(x_train, label=y_train, feature_name=feature_names, free_raw_data=False)
        valid_dataset = lgb.Dataset(x_valid, label=y_valid, feature_name=feature_names, free_raw_data=False)
        pos_count = max(int(y_train.sum()), 1)
        neg_count = max(int(len(y_train) - y_train.sum()), 1)
        params = {
            "objective": "binary",
            "metric": self.model_config["eval_metric"],
            "learning_rate": self.model_config["learning_rate"],
            "max_depth": self.model_config["max_depth"],
            "num_leaves": min(63, (2 ** self.model_config["max_depth"]) - 1),
            "verbosity": -1,
            "device_type": "gpu" if self.model_config.get("enable_gpu") else self.model_config["device_type"],
            "scale_pos_weight": neg_count / pos_count,
            "seed": self.model_config["random_state"],
        }

        current_model = resume_artifact.model if resume_artifact and resume_artifact.backend == "lightgbm" else None
        current_iteration = int(resume_artifact.metadata.get("trained_iterations", 0)) if resume_artifact else 0
        best_auc = float(resume_artifact.metadata.get("best_auc", float("-inf"))) if resume_artifact else float("-inf")
        best_artifact = resume_artifact if resume_artifact else None
        no_improve = int(resume_artifact.metadata.get("no_improve_checkpoints", 0)) if resume_artifact else 0
        checkpoint_interval = int(self.model_config["checkpoint_interval"])
        max_rounds = int(self.model_config["num_estimators"])
        patience_checkpoints = max(1, int(self.model_config["early_stopping_rounds"]) // checkpoint_interval)

        for trained_rounds in range(current_iteration + checkpoint_interval, max_rounds + 1, checkpoint_interval):
            current_model = lgb.train(
                params=params,
                train_set=train_dataset,
                num_boost_round=checkpoint_interval,
                valid_sets=[valid_dataset],
                init_model=current_model,
                keep_training_booster=True,
            )

            y_prob = current_model.predict(x_valid)
            threshold = select_threshold_for_recall(
                y_valid,
                y_prob,
                self.model_config["threshold_recall_target"],
            )
            metrics = compute_classification_metrics(y_valid, y_prob, threshold)
            metadata = {
                "trained_iterations": trained_rounds,
                "best_iteration": trained_rounds,
                "best_auc": max(best_auc, metrics["roc_auc"]),
                "no_improve_checkpoints": no_improve,
                "feature_means": np.mean(x_train, axis=0).tolist(),
                "validation_rows": len(y_valid),
                "training_rows": len(y_train),
                "target_column": self.model_config["target_column"],
                "resume_enabled": True,
                "dataset_signature": dataset_signature,
            }
            artifact = self._build_checkpoint_artifact(
                model=current_model,
                backend="lightgbm",
                preprocessor=preprocessor,
                raw_feature_names=raw_feature_names,
                feature_names=feature_names,
                threshold=threshold,
                metrics=metrics,
                metadata=metadata,
            )

            checkpoint_path = self.paths["checkpoints_dir"] / f"fraud_model_iter_{trained_rounds}.pkl"
            save_artifact(artifact, checkpoint_path)

            is_best = metrics["roc_auc"] > best_auc
            append_training_log(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "backend": "lightgbm",
                    "iteration": trained_rounds,
                    **metrics,
                    "is_best": int(is_best),
                },
                self.log_path,
            )
            if is_best:
                best_auc = metrics["roc_auc"]
                no_improve = 0
                metadata["best_auc"] = best_auc
                metadata["no_improve_checkpoints"] = no_improve
                artifact.metadata = metadata
                best_artifact = artifact
                save_artifact(artifact, self.paths["models_dir"] / "fraud_model_best.pkl")
            else:
                no_improve += 1

            if no_improve >= patience_checkpoints:
                break

        if best_artifact is None:
            raise RuntimeError("LightGBM training did not produce a valid checkpoint.")
        return best_artifact

    def _train_gradient_boosting(
        self,
        matrices: Dict[str, object],
        resume_artifact: ModelArtifact | None,
        dataset_signature: Dict[str, object],
    ) -> ModelArtifact:
        x_train = matrices["x_train"]
        x_valid = matrices["x_valid"]
        y_train = matrices["y_train"]
        y_valid = matrices["y_valid"]
        preprocessor = matrices["preprocessor"]
        raw_feature_names = matrices["raw_feature_names"]
        feature_names = matrices["feature_names"]
        checkpoint_interval = int(self.model_config["checkpoint_interval"])
        max_rounds = int(self.model_config["num_estimators"])
        patience_checkpoints = max(1, int(self.model_config["early_stopping_rounds"]) // checkpoint_interval)
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

        if resume_artifact and resume_artifact.backend == "gradient_boosting":
            model = resume_artifact.model
            current_iteration = int(resume_artifact.metadata.get("trained_iterations", 0))
            best_auc = float(resume_artifact.metadata.get("best_auc", float("-inf")))
            best_artifact = resume_artifact
            no_improve = int(resume_artifact.metadata.get("no_improve_checkpoints", 0))
        else:
            model = GradientBoostingClassifier(
                learning_rate=self.model_config["learning_rate"],
                n_estimators=checkpoint_interval,
                max_depth=self.model_config["max_depth"],
                random_state=self.model_config["random_state"],
                warm_start=True,
            )
            current_iteration = 0
            best_auc = float("-inf")
            best_artifact = None
            no_improve = 0

        for trained_rounds in range(current_iteration + checkpoint_interval, max_rounds + 1, checkpoint_interval):
            model.set_params(n_estimators=trained_rounds, warm_start=True)
            model.fit(x_train, y_train, sample_weight=sample_weight)
            y_prob = model.predict_proba(x_valid)[:, 1]
            threshold = select_threshold_for_recall(
                y_valid,
                y_prob,
                self.model_config["threshold_recall_target"],
            )
            metrics = compute_classification_metrics(y_valid, y_prob, threshold)
            metadata = {
                "trained_iterations": trained_rounds,
                "best_iteration": trained_rounds,
                "best_auc": max(best_auc, metrics["roc_auc"]),
                "no_improve_checkpoints": no_improve,
                "feature_means": np.mean(x_train, axis=0).tolist(),
                "validation_rows": len(y_valid),
                "training_rows": len(y_train),
                "target_column": self.model_config["target_column"],
                "resume_enabled": True,
                "dataset_signature": dataset_signature,
            }
            artifact = self._build_checkpoint_artifact(
                model=model,
                backend="gradient_boosting",
                preprocessor=preprocessor,
                raw_feature_names=raw_feature_names,
                feature_names=feature_names,
                threshold=threshold,
                metrics=metrics,
                metadata=metadata,
            )
            checkpoint_path = self.paths["checkpoints_dir"] / f"fraud_model_iter_{trained_rounds}.pkl"
            save_artifact(artifact, checkpoint_path)

            is_best = metrics["roc_auc"] > best_auc
            append_training_log(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "backend": "gradient_boosting",
                    "iteration": trained_rounds,
                    **metrics,
                    "is_best": int(is_best),
                },
                self.log_path,
            )
            if is_best:
                best_auc = metrics["roc_auc"]
                no_improve = 0
                metadata["best_auc"] = best_auc
                metadata["no_improve_checkpoints"] = no_improve
                artifact.metadata = metadata
                best_artifact = artifact
                save_artifact(artifact, self.paths["models_dir"] / "fraud_model_best.pkl")
            else:
                no_improve += 1

            if no_improve >= patience_checkpoints:
                break

        if best_artifact is None:
            raise RuntimeError("Gradient boosting fallback did not produce a valid checkpoint.")
        return best_artifact

    def _train_random_forest(
        self,
        matrices: Dict[str, object],
        raw_feature_names: List[str],
        feature_names: List[str],
        preprocessor: ColumnTransformer,
    ) -> Tuple[RandomForestClassifier, Dict[str, float]]:
        x_train = matrices["x_train"]
        x_valid = matrices["x_valid"]
        y_train = matrices["y_train"]
        y_valid = matrices["y_valid"]

        model = RandomForestClassifier(
            n_estimators=self.model_config["rf_n_estimators"],
            max_depth=self.model_config["rf_max_depth"],
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=self.model_config["random_state"],
        )
        model.fit(x_train, y_train)
        y_prob = model.predict_proba(x_valid)[:, 1]
        threshold = select_threshold_for_recall(
            y_valid,
            y_prob,
            self.model_config["threshold_recall_target"],
        )
        metrics = compute_classification_metrics(y_valid, y_prob, threshold)
        artifact = self._build_checkpoint_artifact(
            model=model,
            backend="random_forest",
            preprocessor=preprocessor,
            raw_feature_names=raw_feature_names,
            feature_names=feature_names,
            threshold=threshold,
            metrics=metrics,
            metadata={"feature_means": np.mean(x_train, axis=0).tolist()},
        )
        save_artifact(artifact, self.paths["models_dir"] / "fraud_model_random_forest.pkl")
        return model, metrics

    def train(self) -> Dict[str, object]:
        dataset, target = self.load_dataset()
        dataset_signature = self._dataset_signature(dataset, target)
        resume_path = latest_checkpoint(self.paths["checkpoints_dir"])
        resume_artifact = load_artifact(resume_path) if resume_path else None
        if resume_artifact is not None:
            if resume_artifact.metadata.get("dataset_signature") != dataset_signature:
                resume_artifact = None
        matrices = self._prepare_matrices(dataset, target, resume_artifact)
        backend = self._resolve_backend()

        if backend == "lightgbm":
            best_artifact = self._train_lightgbm(matrices, resume_artifact, dataset_signature)
        elif backend == "gradient_boosting":
            best_artifact = self._train_gradient_boosting(matrices, resume_artifact, dataset_signature)
        else:
            raise ValueError(f"Unsupported training backend: {backend}")

        random_forest_model, rf_metrics = self._train_random_forest(
            matrices=matrices,
            raw_feature_names=matrices["raw_feature_names"],
            feature_names=matrices["feature_names"],
            preprocessor=matrices["preprocessor"],
        )

        x_valid = matrices["x_valid"]
        y_valid = matrices["y_valid"]
        primary_valid_prob = predict_with_artifact(
            best_artifact,
            matrices["valid_df"],
            batch_size=self.model_config["batch_size"],
        )
        if self.model_config.get("use_ensemble", True):
            primary_metrics = dict(best_artifact.metrics)
            ensemble_prob = 0.5 * primary_valid_prob + 0.5 * random_forest_model.predict_proba(x_valid)[:, 1]
            ensemble_threshold = select_threshold_for_recall(
                y_valid,
                ensemble_prob,
                self.model_config["threshold_recall_target"],
            )
            ensemble_metrics = compute_classification_metrics(y_valid, ensemble_prob, ensemble_threshold)
            best_artifact.secondary_model = random_forest_model
            best_artifact.threshold = ensemble_threshold
            best_artifact.metrics = ensemble_metrics
            best_artifact.metadata["use_soft_voting"] = True
            best_artifact.metadata["secondary_model_weight"] = 0.5
            best_artifact.metadata["random_forest_metrics"] = rf_metrics
            best_artifact.metadata["primary_validation_metrics"] = primary_metrics
        else:
            ensemble_metrics = best_artifact.metrics

        save_artifact(best_artifact, self.paths["models_dir"] / "fraud_model_best.pkl")
        save_artifact(best_artifact, self.paths["models_dir"] / "fraud_model.pkl")

        predictions = dataset.copy()
        predictions = predictions.merge(
            matrices["split_dataset"][["claim_event_id", "data_split"]],
            on="claim_event_id",
            how="left",
            suffixes=("", "_from_split"),
        )
        predictions["fraud_probability"] = predict_with_artifact(
            best_artifact,
            predictions,
            batch_size=self.model_config["batch_size"],
        )
        predictions["fraud_prediction"] = (
            predictions["fraud_probability"] >= best_artifact.threshold
        ).astype(int)
        predictions.to_csv(self.paths["processed_dir"] / "predictions.csv", index=False)

        summary = {
            "backend": backend,
            "target": target,
            "best_metrics": best_artifact.metrics,
            "random_forest_metrics": rf_metrics,
            "validation_rows": len(y_valid),
        }
        dump_json(summary, self.paths["logs_dir"] / "training_summary.json")
        return {
            "backend": backend,
            "target": target,
            "artifact": best_artifact,
            "ensemble_metrics": ensemble_metrics,
            "rf_metrics": rf_metrics,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fraud detection model with checkpointing.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = FraudModelTrainer(args.config)
    results = trainer.train()
    print(
        f"Training complete using {results['backend']} backend. "
        f"Validation ROC AUC={results['ensemble_metrics']['roc_auc']:.4f}, "
        f"Recall={results['ensemble_metrics']['recall']:.4f}"
    )


if __name__ == "__main__":
    main()
