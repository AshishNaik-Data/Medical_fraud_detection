from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.common import DEFAULT_CONFIG_PATH, ensure_project_dirs, load_config


RAW_FILENAMES = {
    "claim": "claim.csv",
    "claim_patient": "claim_associated_with_patient.csv",
    "claim_prescriber": "claim_submitted_by_prescriber.csv",
    "specialty_subspecialty": "specialties_subspecialties.csv",
    "prescriber_subspecialty": "subspecialties_prescriber.csv",
}


class MedicalFraudDataProcessor:
    """Materialize and normalize the raw claim data into an event-level dataset."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        self.config = load_config(config_path)
        ensure_project_dirs(self.config)
        self.raw_dir = Path(self.config["paths"]["raw_dir"])
        self.processed_dir = Path(self.config["paths"]["processed_dir"])

    def materialize_raw_data(self) -> Dict[str, Path]:
        copied_paths: Dict[str, Path] = {}
        for key, source_path in self.config["data_sources"].items():
            destination = self.raw_dir / RAW_FILENAMES[key]
            source = Path(source_path).expanduser()
            if source.exists():
                if source.resolve() != destination.resolve():
                    shutil.copy2(source, destination)
                copied_paths[key] = destination
                continue
            if destination.exists():
                copied_paths[key] = destination
                continue
            raise FileNotFoundError(
                f"Expected raw dataset at {source} or {destination}. "
                "Update config/data_sources or place the CSV files in data/raw/."
            )
            copied_paths[key] = destination
        return copied_paths

    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        paths = self.materialize_raw_data()
        return {name: pd.read_csv(path) for name, path in paths.items()}

    def run(self) -> pd.DataFrame:
        frames = self.load_raw_data()

        claim_lines = frames["claim"].rename(
            columns={
                "id": "claim_group_id",
                "time": "claim_time",
                "Code Group Title": "code_group_title",
                "ICD -9": "icd9_code",
                "ICD-10": "icd10_code",
                "ICD-9 Description": "icd9_description",
                "ICD-10 Description": "icd10_description",
            }
        )
        claim_lines["claim_group_id"] = claim_lines["claim_group_id"].astype(str)
        claim_lines["claim_time"] = pd.to_datetime(claim_lines["claim_time"], errors="coerce")
        claim_lines = claim_lines.sort_values(["claim_group_id", "claim_time"], kind="stable").reset_index(drop=True)
        claim_lines["claim_sequence_in_group"] = claim_lines.groupby("claim_group_id").cumcount() + 1
        claim_lines["claim_event_id"] = (
            claim_lines["claim_group_id"] + "_evt_" + claim_lines["claim_sequence_in_group"].astype(str)
        )

        claim_lines["code_group_title"] = claim_lines["code_group_title"].fillna("Unknown")
        claim_lines["icd9_code"] = claim_lines["icd9_code"].fillna("UNKNOWN")
        claim_lines["icd10_code"] = claim_lines["icd10_code"].fillna("UNKNOWN")
        claim_lines["icd9_description"] = claim_lines["icd9_description"].fillna("")
        claim_lines["icd10_description"] = claim_lines["icd10_description"].fillna("")
        claim_lines["primary_diagnosis_description"] = (
            claim_lines["icd10_description"]
            .where(claim_lines["icd10_description"].str.len() > 0, claim_lines["icd9_description"])
            .replace("", "Unknown diagnosis")
        )

        group_features = (
            claim_lines.groupby("claim_group_id")
            .agg(
                claim_group_line_count=("claim_event_id", "size"),
                claim_group_unique_code_groups=("code_group_title", "nunique"),
                claim_group_unique_icd10=("icd10_code", "nunique"),
                claim_group_first_time=("claim_time", "min"),
                claim_group_last_time=("claim_time", "max"),
            )
            .reset_index()
        )
        group_features["claim_group_span_days"] = (
            group_features["claim_group_last_time"] - group_features["claim_group_first_time"]
        ).dt.days.fillna(0)
        claim_lines = claim_lines.merge(group_features, on="claim_group_id", how="left")

        claim_patient = frames["claim_patient"].rename(
            columns={
                "Claim": "claim_group_id",
                "Patient": "patient_id",
            }
        )
        claim_prescriber = frames["claim_prescriber"].rename(
            columns={
                "Claim": "claim_group_id",
                "Prescriber": "doctor_id",
            }
        )
        specialty_map = frames["specialty_subspecialty"].rename(
            columns={
                "Specialty": "doctor_specialty",
                "SubSpecialty": "doctor_subspecialty",
            }
        )
        prescriber_map = frames["prescriber_subspecialty"].rename(
            columns={
                "Prescriber": "doctor_id",
                "SubSpecialty": "doctor_subspecialty",
            }
        )

        doctor_dimension = prescriber_map.merge(specialty_map, on="doctor_subspecialty", how="left")
        doctor_dimension["doctor_specialty"] = doctor_dimension["doctor_specialty"].fillna("Unknown")

        enriched = (
            claim_lines.merge(claim_patient, on="claim_group_id", how="left")
            .merge(claim_prescriber, on="claim_group_id", how="left")
            .merge(doctor_dimension, on="doctor_id", how="left")
        )
        enriched["patient_id"] = enriched["patient_id"].fillna("unknown_patient")
        enriched["doctor_id"] = enriched["doctor_id"].fillna("unknown_doctor")
        enriched["doctor_subspecialty"] = enriched["doctor_subspecialty"].fillna("Unknown")
        enriched["doctor_specialty"] = enriched["doctor_specialty"].fillna("Unknown")
        enriched["claim_date"] = enriched["claim_time"].dt.strftime("%Y-%m-%d")
        enriched["claim_year"] = enriched["claim_time"].dt.year.fillna(0).astype(int)
        enriched["claim_month"] = enriched["claim_time"].dt.month.fillna(0).astype(int)
        enriched["claim_dayofweek"] = enriched["claim_time"].dt.dayofweek.fillna(0).astype(int)
        enriched["claim_weekday_name"] = enriched["claim_time"].dt.day_name().fillna("Unknown")
        enriched["diagnosis_signature"] = (
            enriched["code_group_title"].astype(str)
            + "|"
            + enriched["icd10_code"].astype(str)
            + "|"
            + enriched["icd9_code"].astype(str)
        )

        doctor_reference = (
            enriched[
                [
                    "doctor_id",
                    "doctor_specialty",
                    "doctor_subspecialty",
                ]
            ]
            .drop_duplicates()
            .sort_values("doctor_id")
        )
        patient_reference = enriched[["patient_id"]].drop_duplicates().sort_values("patient_id")
        specialty_reference = (
            enriched[["doctor_specialty", "doctor_subspecialty"]]
            .drop_duplicates()
            .sort_values(["doctor_specialty", "doctor_subspecialty"])
        )

        enriched.to_csv(self.processed_dir / "claims_enriched.csv", index=False)
        doctor_reference.to_csv(self.processed_dir / "doctor_reference.csv", index=False)
        patient_reference.to_csv(self.processed_dir / "patient_reference.csv", index=False)
        specialty_reference.to_csv(self.processed_dir / "specialty_reference.csv", index=False)
        return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize raw medical insurance fraud datasets.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = MedicalFraudDataProcessor(args.config)
    dataset = processor.run()
    print(f"Saved normalized claims dataset with shape {dataset.shape} to {processor.processed_dir}")


if __name__ == "__main__":
    main()
