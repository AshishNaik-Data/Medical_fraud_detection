from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.common import DEFAULT_CONFIG_PATH, ensure_project_dirs, load_config


RULE_CYPHER_QUERIES = {
    "too_many_claims_per_doctor": """
MATCH (d:Doctor)-[:SUBMITTED]->(c:Claim)
WITH d, count(c) AS claim_count
WHERE claim_count > $min_claims
RETURN d.doctor_id AS doctor_id, claim_count
ORDER BY claim_count DESC
""".strip(),
    "doctor_treating_outside_specialty": """
MATCH (d:Doctor)-[:SUBMITTED]->(c:Claim)
WHERE c.specialty_mismatch = true
RETURN d.doctor_id AS doctor_id, c.claim_event_id AS claim_event_id, c.code_group_title AS code_group
""".strip(),
    "multiple_claims_same_patient_same_day": """
MATCH (p:Patient)-[:HAS_CLAIM]->(c:Claim)
WITH p, date(c.claim_time) AS claim_day, count(c) AS claim_count
WHERE claim_count > 1
RETURN p.patient_id AS patient_id, claim_day, claim_count
""".strip(),
    "doctor_in_many_fraud_communities": """
MATCH (d:Doctor)
WHERE d.community_risk_score >= $community_risk_cutoff
RETURN d.doctor_id AS doctor_id, d.community_id AS community_id, d.community_risk_score
ORDER BY d.community_risk_score DESC
""".strip(),
}


class FraudRuleEngine:
    """Create explainable weak fraud labels from provider, claim, and graph anomalies."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        self.config = load_config(config_path)
        ensure_project_dirs(self.config)
        self.processed_dir = Path(self.config["paths"]["processed_dir"])
        self.feature_config = self.config["features"]

    def load_feature_frame(self) -> pd.DataFrame:
        return pd.read_csv(self.processed_dir / "claims_with_graph_features.csv", parse_dates=["claim_time"])

    def _build_provider_anomaly_summary(self, claims: pd.DataFrame) -> pd.DataFrame:
        provider_summary = (
            claims.groupby("doctor_id")
            .agg(
                claims_per_doctor=("claim_event_id", "size"),
                patients_per_doctor=("patient_id", "nunique"),
                duplicate_claim_rate=("duplicate_claim_flag", "mean"),
                specialty_mismatch_rate=("specialty_mismatch", "mean"),
                shared_patient_ratio=("shared_patient_ratio", "mean"),
                doctor_centrality=("doctor_centrality", "mean"),
                provider_network_density=("provider_network_density", "mean"),
                max_same_day_claims=("same_day_doctor_claims", "max"),
            )
            .reset_index()
        )

        anomaly_inputs = [
            "claims_per_doctor",
            "patients_per_doctor",
            "duplicate_claim_rate",
            "specialty_mismatch_rate",
            "shared_patient_ratio",
            "doctor_centrality",
            "provider_network_density",
            "max_same_day_claims",
        ]
        z_frame = provider_summary[anomaly_inputs].copy()
        for column in anomaly_inputs:
            std = z_frame[column].std(ddof=0)
            if np.isclose(std, 0.0):
                z_frame[column] = 0.0
            else:
                z_frame[column] = (z_frame[column] - z_frame[column].mean()) / std

        provider_summary["provider_anomaly_score"] = z_frame.clip(lower=0.0).mean(axis=1)
        provider_summary["abnormal_provider_behavior"] = (
            provider_summary["provider_anomaly_score"] >= self.feature_config["provider_anomaly_z_threshold"]
        ).astype(int)
        return provider_summary

    def _compile_rule_list(self, row: pd.Series) -> str:
        active_rules: List[str] = []
        if row["too_many_claims_per_doctor"]:
            active_rules.append("excessive_claim_volume")
        if row["specialty_mismatch"]:
            active_rules.append("specialty_mismatch")
        if row["multiple_claims_same_patient_same_day"]:
            active_rules.append("same_patient_same_day")
        if row["repeated_claim_flag"]:
            active_rules.append("repeated_claim")
        if row["unrealistic_treatment_schedule"]:
            active_rules.append("unrealistic_schedule")
        if row["abnormal_provider_behavior"]:
            active_rules.append("abnormal_provider_behavior")
        if row["doctor_in_risky_community"]:
            active_rules.append("risky_provider_community")
        return "; ".join(active_rules) if active_rules else "no_rules_triggered"

    def _compile_reason(self, row: pd.Series) -> str:
        reasons = []
        if row["too_many_claims_per_doctor"]:
            reasons.append("doctor has unusually high claim volume")
        if row["specialty_mismatch"]:
            reasons.append("the diagnosis pattern is unusual for the doctor's specialty")
        if row["repeated_claim_flag"]:
            reasons.append("similar claims were repeated within a short window")
        if row["unrealistic_treatment_schedule"]:
            reasons.append("the treatment cadence looks clinically unrealistic")
        if row["doctor_in_risky_community"]:
            reasons.append("the provider sits inside a high-risk referral community")
        if not reasons:
            return "Claim did not trigger a high-risk fraud rule."
        return "Claim flagged because " + " and ".join(reasons[:3]) + "."

    def apply(self, claims: pd.DataFrame | None = None) -> pd.DataFrame:
        claims = claims.copy() if claims is not None else self.load_feature_frame()
        provider_summary = self._build_provider_anomaly_summary(claims)
        claims = claims.merge(
            provider_summary[["doctor_id", "provider_anomaly_score", "abnormal_provider_behavior"]],
            on="doctor_id",
            how="left",
        )

        high_claim_cutoff = claims["claims_per_doctor"].quantile(self.feature_config["high_claim_quantile"])
        daily_cutoff = int(np.ceil(claims["same_day_doctor_claims"].quantile(self.feature_config["doctor_daily_claim_quantile"])))
        daily_cutoff = max(daily_cutoff, 2)
        claims["too_many_claims_per_doctor"] = (
            (claims["claims_per_doctor"] >= high_claim_cutoff) | (claims["same_day_doctor_claims"] >= daily_cutoff)
        ).astype(int)
        claims["multiple_claims_same_patient_same_day"] = (claims["same_day_patient_claims"] > 1).astype(int)
        claims["unrealistic_treatment_schedule"] = (
            (claims["doctor_patient_claim_gap_days"] <= 1) | (claims["same_day_doctor_patient_claims"] > 1)
        ).astype(int)

        rule_weights: Dict[str, float] = {
            "too_many_claims_per_doctor": 1.3,
            "specialty_mismatch": 1.0,
            "multiple_claims_same_patient_same_day": 1.0,
            "repeated_claim_flag": 1.25,
            "unrealistic_treatment_schedule": 1.0,
            "abnormal_provider_behavior": 1.25,
        }
        claims["base_rule_score"] = sum(
            claims[rule_name].astype(float) * weight for rule_name, weight in rule_weights.items()
        )
        claims["provisional_weak_label"] = (
            (claims["base_rule_score"] >= self.feature_config["weak_label_min_score"])
            | ((claims["repeated_claim_flag"] == 1) & (claims["specialty_mismatch"] == 1))
        ).astype(int)

        community_risk = (
            claims.groupby("louvain_community")
            .agg(
                community_provisional_rate=("provisional_weak_label", "mean"),
                community_claim_count=("claim_event_id", "size"),
                community_average_centrality=("doctor_centrality", "mean"),
            )
            .reset_index()
        )
        community_risk["community_risk_score"] = (
            (0.6 * community_risk["community_provisional_rate"])
            + (0.2 * (community_risk["community_claim_count"] / community_risk["community_claim_count"].max()))
            + (0.2 * community_risk["community_average_centrality"])
        )
        risk_cutoff = community_risk["community_risk_score"].quantile(self.feature_config["community_risk_quantile"])
        claims = claims.drop(columns=["community_risk_score"], errors="ignore").merge(
            community_risk[["louvain_community", "community_risk_score"]],
            on="louvain_community",
            how="left",
        )
        claims["doctor_in_risky_community"] = (
            claims["community_risk_score"].fillna(0.0) >= risk_cutoff
        ).astype(int)
        claims["doctor_network_risk_score"] = (
            0.4 * claims["provider_anomaly_score"].fillna(0.0)
            + 0.3 * claims["shared_patient_ratio"].fillna(0.0)
            + 0.3 * claims["community_risk_score"].fillna(0.0)
        )

        claims["final_rule_score"] = claims["base_rule_score"] + claims["doctor_in_risky_community"]
        claims["weak_fraud_label"] = (
            (claims["final_rule_score"] >= self.feature_config["weak_label_min_score"])
            | (
                (claims["too_many_claims_per_doctor"] == 1)
                & (claims["abnormal_provider_behavior"] == 1)
            )
            | (
                (claims["repeated_claim_flag"] == 1)
                & (claims["unrealistic_treatment_schedule"] == 1)
            )
        ).astype(int)

        claims["triggered_rules"] = claims.apply(self._compile_rule_list, axis=1)
        claims["fraud_reason_summary"] = claims.apply(self._compile_reason, axis=1)

        provider_summary = (
            claims.groupby("doctor_id")
            .agg(
                doctor_specialty=("doctor_specialty", "first"),
                claim_count=("claim_event_id", "size"),
                suspicious_claims=("weak_fraud_label", "sum"),
                average_rule_score=("final_rule_score", "mean"),
                average_network_risk=("doctor_network_risk_score", "mean"),
            )
            .reset_index()
            .sort_values(["suspicious_claims", "average_rule_score"], ascending=False)
        )
        provider_summary.to_csv(self.processed_dir / "provider_summary.csv", index=False)
        claims.to_csv(self.processed_dir / "claims_with_rules.csv", index=False)
        return claims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply rule-based fraud detectors.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = FraudRuleEngine(args.config)
    results = engine.apply()
    positives = int(results["weak_fraud_label"].sum())
    print(f"Saved claims_with_rules.csv with {positives} weak fraud labels out of {len(results)} claims")


if __name__ == "__main__":
    main()
