from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.common import (
    DEFAULT_CONFIG_PATH,
    ensure_project_dirs,
    load_artifact,
    load_config,
    predict_with_artifact,
    set_matplotlib_cache,
)

try:
    import shap
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    shap = None


class FraudExplainer:
    """Generate global and local fraud explanations."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        self.config = load_config(config_path)
        ensure_project_dirs(self.config)
        set_matplotlib_cache(self.config)
        self.paths = {key: Path(value) for key, value in self.config["paths"].items()}
        self.artifact = load_artifact(self.paths["models_dir"] / "fraud_model_best.pkl")

    def _load_predictions(self) -> pd.DataFrame:
        return pd.read_csv(self.paths["processed_dir"] / "predictions.csv", parse_dates=["claim_time"])

    def _transform(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = self.artifact.preprocessor.transform(frame[self.artifact.raw_feature_names])
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return transformed

    def _global_importance(self, sample: pd.DataFrame) -> pd.DataFrame:
        transformed = self._transform(sample)
        if self.artifact.backend == "lightgbm":
            importance = self.artifact.model.feature_importance(importance_type="gain")
        elif hasattr(self.artifact.model, "feature_importances_"):
            importance = self.artifact.model.feature_importances_
        elif shap is not None:
            explainer = shap.Explainer(self.artifact.model, transformed)
            shap_values = explainer(transformed)
            importance = np.abs(shap_values.values).mean(axis=0)
        else:
            feature_means = np.asarray(self.artifact.metadata.get("feature_means", np.zeros(transformed.shape[1])))
            importance = np.abs(transformed - feature_means).mean(axis=0)

        report = pd.DataFrame(
            {
                "feature_name": self.artifact.feature_names,
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)
        return report

    def _local_contributions(self, frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        transformed = self._transform(frame)
        if self.artifact.backend == "lightgbm":
            contributions = self.artifact.model.predict(
                transformed,
                pred_contrib=True,
                num_iteration=self.artifact.metadata.get("best_iteration"),
            )
            bias = contributions[:, -1]
            return contributions[:, :-1], bias

        if shap is not None:
            explainer = shap.Explainer(self.artifact.model, transformed)
            shap_values = explainer(transformed)
            return shap_values.values, np.asarray(shap_values.base_values)

        feature_means = np.asarray(self.artifact.metadata.get("feature_means", np.zeros(transformed.shape[1])))
        if hasattr(self.artifact.model, "feature_importances_"):
            importances = np.asarray(self.artifact.model.feature_importances_)
        else:
            importances = np.ones(transformed.shape[1], dtype=float)
        contributions = (transformed - feature_means) * importances
        return contributions, np.zeros(len(frame))

    def explain(self, claim_event_id: str | None = None) -> dict:
        predictions = self._load_predictions().sort_values("fraud_probability", ascending=False)
        global_report = self._global_importance(predictions.head(200))
        global_report.to_csv(self.paths["processed_dir"] / "global_feature_importance.csv", index=False)

        if claim_event_id is None:
            claim_event_id = predictions.iloc[0]["claim_event_id"]
        claim_frame = predictions.loc[predictions["claim_event_id"] == claim_event_id].head(1).copy()
        if claim_frame.empty:
            raise ValueError(f"Claim event '{claim_event_id}' was not found in predictions.csv")

        contributions, bias = self._local_contributions(claim_frame)
        local_report = pd.DataFrame(
            {
                "feature_name": self.artifact.feature_names,
                "contribution": contributions[0],
            }
        )
        local_report["absolute_contribution"] = local_report["contribution"].abs()
        local_report = local_report.sort_values("absolute_contribution", ascending=False).head(10)

        risk_connections = []
        row = claim_frame.iloc[0]
        if row.get("doctor_network_risk_score", 0.0) >= 0.5:
            risk_connections.append("doctor belongs to a high-risk provider community")
        if row.get("patient_provider_overlap", 0) > 0:
            risk_connections.append(
                f"patient shares care pathways with {int(row['patient_provider_overlap'])} other provider(s)"
            )
        if row.get("shared_patient_ratio", 0.0) > 0.4:
            risk_connections.append("doctor shares an unusually high portion of patients with peers")

        explanation = {
            "claim_event_id": claim_event_id,
            "fraud_probability": float(row["fraud_probability"]),
            "baseline_score": float(bias[0]) if np.ndim(bias) else float(bias),
            "top_features": local_report.to_dict(orient="records"),
            "top_global_features": global_report.head(15).to_dict(orient="records"),
            "doctor_network_risk_score": float(row.get("doctor_network_risk_score", 0.0)),
            "suspicious_graph_connections": risk_connections,
            "narrative": row.get("fraud_reason_summary", ""),
        }

        pd.DataFrame(local_report).to_csv(self.paths["processed_dir"] / "claim_local_explanation.csv", index=False)
        pd.DataFrame([explanation]).to_json(
            self.paths["processed_dir"] / "claim_explanation_summary.json",
            orient="records",
            indent=2,
        )
        return explanation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fraud explainability reports.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML.")
    parser.add_argument("--claim-id", default=None, help="Specific claim_event_id to explain.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    explainer = FraudExplainer(args.config)
    explanation = explainer.explain(args.claim_id)
    print(
        f"Explained claim {explanation['claim_event_id']} with probability "
        f"{explanation['fraud_probability']:.4f}"
    )


if __name__ == "__main__":
    main()
