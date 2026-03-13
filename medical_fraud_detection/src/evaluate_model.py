from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.common import (
    DEFAULT_CONFIG_PATH,
    compute_classification_metrics,
    dump_json,
    load_artifact,
    load_config,
    predict_with_artifact,
)


def evaluate(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    config = load_config(config_path)
    processed_dir = Path(config["paths"]["processed_dir"])
    models_dir = Path(config["paths"]["models_dir"])
    dataset = pd.read_csv(processed_dir / "modeling_dataset.csv", parse_dates=["claim_time"])
    split = pd.read_csv(processed_dir / "train_valid_split.csv")
    artifact = load_artifact(models_dir / "fraud_model_best.pkl")

    target = config["model"]["target_column"]
    if target not in dataset.columns or dataset[target].notna().sum() == 0:
        target = config["model"]["weak_label_column"]
    dataset[target] = dataset[target].fillna(0).astype(int)

    evaluation_frame = dataset.merge(split, on="claim_event_id", how="left")
    valid_frame = evaluation_frame.loc[evaluation_frame["data_split"] == "valid"].copy()
    probabilities = predict_with_artifact(
        artifact,
        valid_frame,
        batch_size=config["model"]["batch_size"],
    )
    metrics = compute_classification_metrics(valid_frame[target], probabilities, artifact.threshold)
    metrics["target_column"] = target
    metrics["validation_rows"] = int(len(valid_frame))

    dump_json(metrics, Path(config["paths"]["logs_dir"]) / "evaluation_metrics.json")
    predictions = valid_frame[
        [
            "claim_event_id",
            "claim_group_id",
            "patient_id",
            "doctor_id",
            target,
            "triggered_rules",
            "fraud_reason_summary",
        ]
    ].copy()
    predictions["fraud_probability"] = probabilities
    predictions["fraud_prediction"] = (predictions["fraud_probability"] >= artifact.threshold).astype(int)
    predictions.to_csv(processed_dir / "evaluation_predictions.csv", index=False)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the fraud detection model.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(args.config)
    print(
        "Evaluation complete. "
        f"ROC AUC={metrics['roc_auc']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}"
    )


if __name__ == "__main__":
    main()
