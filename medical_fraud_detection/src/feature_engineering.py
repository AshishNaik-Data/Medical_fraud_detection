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

from src.common import DEFAULT_CONFIG_PATH, ensure_project_dirs, load_config
from src.data_processing import MedicalFraudDataProcessor
from src.fraud_rules import FraudRuleEngine
from src.graph_features import FraudGraphFeatureBuilder


def rolling_window_count(group: pd.DataFrame, days: int) -> pd.Series:
    timestamps = (group["claim_time"].astype("int64").to_numpy()) // 86_400_000_000_000
    output = np.zeros(len(group), dtype=int)
    left = 0
    for right, value in enumerate(timestamps):
        while value - timestamps[left] > days:
            left += 1
        output[right] = right - left + 1
    return pd.Series(output, index=group.index)


class FeatureEngineeringPipeline:
    """Create the hybrid tabular + graph fraud feature matrix."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = Path(config_path)
        self.config = load_config(config_path)
        ensure_project_dirs(self.config)
        self.processed_dir = Path(self.config["paths"]["processed_dir"])
        self.feature_config = self.config["features"]

    def load_or_build_claims(self) -> pd.DataFrame:
        claims_path = self.processed_dir / "claims_enriched.csv"
        if claims_path.exists():
            return pd.read_csv(claims_path, parse_dates=["claim_time"])
        processor = MedicalFraudDataProcessor(self.config_path)
        return processor.run()

    def add_claim_temporal_features(self, claims: pd.DataFrame) -> pd.DataFrame:
        claims = claims.sort_values(["patient_id", "claim_time", "claim_event_id"], kind="stable").copy()
        claims["claim_date"] = pd.to_datetime(claims["claim_date"], errors="coerce")
        window_days = int(self.feature_config["claim_frequency_window_days"])
        repeated_window = int(self.feature_config["repeated_claim_window_days"])

        claims["claim_frequency"] = (
            claims.groupby("patient_id", group_keys=False)[["claim_time"]]
            .apply(rolling_window_count, days=window_days)
            .reset_index(level=0, drop=True)
        )
        claims["claim_time_gap"] = (
            claims.groupby("patient_id")["claim_time"].diff().dt.days.fillna(window_days + 1).clip(lower=0)
        )

        claims = claims.sort_values(
            ["doctor_id", "patient_id", "claim_time", "claim_event_id"], kind="stable"
        ).reset_index(drop=True)
        claims["doctor_patient_claim_gap_days"] = (
            claims.groupby(["doctor_id", "patient_id"])["claim_time"].diff().dt.days.fillna(window_days + 1)
        ).clip(lower=0)

        claims["same_day_patient_claims"] = claims.groupby(["patient_id", "claim_date"])["claim_event_id"].transform("size")
        claims["same_day_doctor_claims"] = claims.groupby(["doctor_id", "claim_date"])["claim_event_id"].transform("size")
        claims["same_day_doctor_patient_claims"] = claims.groupby(
            ["doctor_id", "patient_id", "claim_date"]
        )["claim_event_id"].transform("size")

        claims = claims.sort_values(
            ["patient_id", "doctor_id", "diagnosis_signature", "claim_time", "claim_event_id"], kind="stable"
        ).reset_index(drop=True)
        duplicate_gap = claims.groupby(["patient_id", "doctor_id", "diagnosis_signature"])["claim_time"].diff().dt.days
        claims["repeated_claim_flag"] = duplicate_gap.fillna(repeated_window + 1).between(0, repeated_window).astype(int)
        claims["duplicate_claim_flag"] = (
            (claims["repeated_claim_flag"] == 1) | (claims["same_day_doctor_patient_claims"] > 1)
        ).astype(int)

        claims = claims.sort_values("claim_event_id").reset_index(drop=True)
        return claims

    def add_provider_features(self, claims: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        claims["claims_per_doctor"] = claims.groupby("doctor_id")["claim_event_id"].transform("size")
        claims["patients_per_doctor"] = claims.groupby("doctor_id")["patient_id"].transform("nunique")
        claims["code_group_frequency"] = claims["code_group_title"].map(claims["code_group_title"].value_counts(normalize=True))
        claims["code_group_rarity"] = 1.0 - claims["code_group_frequency"].fillna(0.0)

        doctor_code_support = (
            claims.groupby(["doctor_id", "code_group_title"])["claim_event_id"].size()
            / claims.groupby("doctor_id")["claim_event_id"].size()
        ).rename("doctor_code_group_rate")
        specialty_code_support = (
            claims.groupby(["doctor_specialty", "code_group_title"])["claim_event_id"].size()
            / claims.groupby("doctor_specialty")["claim_event_id"].size()
        ).rename("specialty_code_group_rate")

        claims = claims.merge(
            doctor_code_support.reset_index(),
            on=["doctor_id", "code_group_title"],
            how="left",
        )
        claims = claims.merge(
            specialty_code_support.reset_index(),
            on=["doctor_specialty", "code_group_title"],
            how="left",
        )
        claims["doctor_code_group_rate"] = claims["doctor_code_group_rate"].fillna(0.0)
        claims["specialty_code_group_rate"] = claims["specialty_code_group_rate"].fillna(0.0)
        claims["doctor_specialty_consistency"] = (
            0.6 * claims["doctor_code_group_rate"] + 0.4 * claims["specialty_code_group_rate"]
        )
        claims["specialty_mismatch"] = (
            claims["doctor_specialty_consistency"] < self.feature_config["specialty_support_threshold"]
        ).astype(int)

        base_amount = float(self.feature_config["amount_proxy_base"])
        claims["claim_amount"] = (
            base_amount
            * (
                1.0
                + (1.75 * claims["code_group_rarity"])
                + (0.20 * claims["claim_group_line_count"].clip(upper=8))
                + (0.12 * claims["claim_group_unique_code_groups"].clip(upper=4))
                + (0.10 * claims["duplicate_claim_flag"])
            )
        ).round(2)

        claims["multi_location_claims"] = 0
        claims["location_data_available"] = 0

        provider_snapshot = (
            claims.groupby("doctor_id")
            .agg(
                doctor_specialty=("doctor_specialty", "first"),
                claims_per_doctor=("claim_event_id", "size"),
                patients_per_doctor=("patient_id", "nunique"),
                specialty_mismatch_rate=("specialty_mismatch", "mean"),
            )
            .reset_index()
        )
        return claims, provider_snapshot

    def run(self) -> pd.DataFrame:
        claims = self.load_or_build_claims()
        claims["claim_time"] = pd.to_datetime(claims["claim_time"], errors="coerce")
        claims = self.add_claim_temporal_features(claims)
        claims, provider_snapshot = self.add_provider_features(claims)

        graph_builder = FraudGraphFeatureBuilder(self.config_path)
        graph_assets = graph_builder.run()
        claims = claims.merge(graph_assets["claim_graph_features"], on=["claim_event_id", "doctor_id", "patient_id"], how="left")
        claims.to_csv(self.processed_dir / "claims_with_graph_features.csv", index=False)

        rule_engine = FraudRuleEngine(self.config_path)
        claims = rule_engine.apply(claims)

        if self.config["model"]["target_column"] in claims.columns:
            claims["target_label"] = claims[self.config["model"]["target_column"]]
        else:
            claims["target_label"] = claims[self.config["model"]["weak_label_column"]]

        ordered_columns = [
            "claim_event_id",
            "claim_group_id",
            "claim_time",
            "claim_date",
            "patient_id",
            "doctor_id",
            "doctor_specialty",
            "doctor_subspecialty",
            "code_group_title",
            "primary_diagnosis_description",
        ]
        remaining_columns = [column for column in claims.columns if column not in ordered_columns]
        claims = claims[ordered_columns + remaining_columns]

        claims.to_csv(self.processed_dir / "modeling_dataset.csv", index=False)
        provider_snapshot.to_csv(self.processed_dir / "provider_feature_snapshot.csv", index=False)
        return claims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full feature engineering pipeline.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = FeatureEngineeringPipeline(args.config)
    dataset = pipeline.run()
    positive_rate = dataset["weak_fraud_label"].mean()
    print(
        f"Saved modeling dataset with shape {dataset.shape}. Weak fraud rate: {positive_rate:.3f}"
    )


if __name__ == "__main__":
    main()
