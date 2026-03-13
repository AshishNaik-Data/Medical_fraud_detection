from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.common import DEFAULT_CONFIG_PATH, ensure_project_dirs, load_config

try:
    from neo4j import GraphDatabase
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    GraphDatabase = None

try:
    from py2neo import Graph as Py2NeoGraph
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Py2NeoGraph = None


SCHEMA_QUERIES = [
    "CREATE CONSTRAINT patient_id IF NOT EXISTS FOR (p:Patient) REQUIRE p.patient_id IS UNIQUE",
    "CREATE CONSTRAINT doctor_id IF NOT EXISTS FOR (d:Doctor) REQUIRE d.doctor_id IS UNIQUE",
    "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.claim_event_id IS UNIQUE",
    "CREATE CONSTRAINT specialty_name IF NOT EXISTS FOR (s:Specialty) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT subspecialty_name IF NOT EXISTS FOR (ss:Subspecialty) REQUIRE ss.name IS UNIQUE",
]


class Neo4jFraudGraphLoader:
    """Export processed graph tables and optionally load them into Neo4j."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        self.config = load_config(config_path)
        ensure_project_dirs(self.config)
        self.processed_dir = Path(self.config["paths"]["processed_dir"])
        self.neo4j_config = self.config["neo4j"]
        self.import_dir = self.processed_dir / "neo4j_import"
        self.import_dir.mkdir(parents=True, exist_ok=True)

    def load_processed_data(self) -> Dict[str, pd.DataFrame]:
        return {
            "modeling_dataset": pd.read_csv(self.processed_dir / "modeling_dataset.csv", parse_dates=["claim_time"]),
            "predictions": pd.read_csv(self.processed_dir / "predictions.csv", parse_dates=["claim_time"]),
            "graph_edges": pd.read_csv(self.processed_dir / "graph_edges.csv"),
        }

    def build_node_and_relationship_tables(self) -> Dict[str, pd.DataFrame]:
        assets = self.load_processed_data()
        dataset = assets["predictions"]

        patient_nodes = dataset[["patient_id"]].drop_duplicates().rename(columns={"patient_id": "id"})
        doctor_nodes = dataset[["doctor_id", "doctor_specialty", "doctor_subspecialty"]].drop_duplicates().rename(
            columns={"doctor_id": "id"}
        )
        claim_nodes = dataset[
            [
                "claim_event_id",
                "claim_group_id",
                "claim_time",
                "code_group_title",
                "fraud_probability",
                "fraud_prediction",
                "weak_fraud_label",
                "specialty_mismatch",
                "community_risk_score",
                "fraud_reason_summary",
            ]
        ].drop_duplicates().rename(columns={"claim_event_id": "id"})
        specialty_nodes = dataset[["doctor_specialty"]].drop_duplicates().rename(columns={"doctor_specialty": "id"})
        subspecialty_nodes = dataset[["doctor_subspecialty"]].drop_duplicates().rename(
            columns={"doctor_subspecialty": "id"}
        )

        relationships = {
            "patient_has_claim": dataset[["patient_id", "claim_event_id"]].drop_duplicates(),
            "doctor_submitted_claim": dataset[["doctor_id", "claim_event_id"]].drop_duplicates(),
            "doctor_specialized_in": dataset[["doctor_id", "doctor_specialty"]].drop_duplicates(),
            "specialty_has_subspecialty": dataset[["doctor_specialty", "doctor_subspecialty"]].drop_duplicates(),
            "doctor_treated_patient": dataset[["doctor_id", "patient_id"]].drop_duplicates(),
        }

        tables = {
            "patient_nodes": patient_nodes,
            "doctor_nodes": doctor_nodes,
            "claim_nodes": claim_nodes,
            "specialty_nodes": specialty_nodes,
            "subspecialty_nodes": subspecialty_nodes,
            **relationships,
        }
        for name, frame in tables.items():
            frame.to_csv(self.import_dir / f"{name}.csv", index=False)
        return tables

    def _connect(self):
        if GraphDatabase is not None:
            driver = GraphDatabase.driver(
                self.neo4j_config["uri"],
                auth=(self.neo4j_config["user"], self.neo4j_config["password"]),
            )
            return ("neo4j", driver)
        if Py2NeoGraph is not None:
            graph = Py2NeoGraph(
                self.neo4j_config["uri"],
                auth=(self.neo4j_config["user"], self.neo4j_config["password"]),
                name=self.neo4j_config["database"],
            )
            return ("py2neo", graph)
        raise RuntimeError("Neither neo4j nor py2neo is installed. Install one of them to load the graph.")

    def load_to_neo4j(self, tables: Dict[str, pd.DataFrame] | None = None) -> None:
        tables = tables or self.build_node_and_relationship_tables()
        backend, connection = self._connect()

        if backend == "neo4j":
            with connection.session(database=self.neo4j_config["database"]) as session:
                for query in SCHEMA_QUERIES:
                    session.run(query)

                self._load_nodes_with_driver(session, "Patient", tables["patient_nodes"], "patient_id")
                self._load_nodes_with_driver(session, "Doctor", tables["doctor_nodes"], "doctor_id")
                self._load_nodes_with_driver(session, "Claim", tables["claim_nodes"], "claim_event_id")
                self._load_nodes_with_driver(session, "Specialty", tables["specialty_nodes"], "name")
                self._load_nodes_with_driver(session, "Subspecialty", tables["subspecialty_nodes"], "name")

                relationship_specs = [
                    ("Patient", "patient_id", "HAS_CLAIM", "Claim", "claim_event_id", tables["patient_has_claim"]),
                    ("Doctor", "doctor_id", "SUBMITTED", "Claim", "claim_event_id", tables["doctor_submitted_claim"]),
                    ("Doctor", "doctor_id", "SPECIALIZED_IN", "Specialty", "name", tables["doctor_specialized_in"]),
                    (
                        "Specialty",
                        "name",
                        "HAS_SUBSPECIALTY",
                        "Subspecialty",
                        "name",
                        tables["specialty_has_subspecialty"],
                    ),
                    ("Doctor", "doctor_id", "TREATED", "Patient", "patient_id", tables["doctor_treated_patient"]),
                ]
                for start_label, start_key, rel_type, end_label, end_key, frame in relationship_specs:
                    self._load_relationships_with_driver(
                        session,
                        start_label,
                        start_key,
                        rel_type,
                        end_label,
                        end_key,
                        frame,
                    )
            connection.close()
        else:
            graph = connection
            for query in SCHEMA_QUERIES:
                graph.run(query)
            for _, frame in tables.items():
                if frame.empty:
                    continue
            raise NotImplementedError("py2neo loading is intentionally left as a fallback placeholder.")

    def _load_nodes_with_driver(self, session, label: str, frame: pd.DataFrame, key_name: str) -> None:
        rows = frame.fillna("").to_dict(orient="records")
        query = f"""
        UNWIND $rows AS row
        MERGE (n:{label} {{{key_name}: row.id}})
        SET n += row
        """
        session.run(query, rows=rows)

    def _load_relationships_with_driver(
        self,
        session,
        start_label: str,
        start_key: str,
        rel_type: str,
        end_label: str,
        end_key: str,
        frame: pd.DataFrame,
    ) -> None:
        if frame.empty:
            return
        rows = frame.fillna("").to_dict(orient="records")
        start_column, end_column = frame.columns.tolist()
        query = f"""
        UNWIND $rows AS row
        MATCH (a:{start_label} {{{start_key}: row.{start_column}}})
        MATCH (b:{end_label} {{{end_key}: row.{end_column}}})
        MERGE (a)-[r:{rel_type}]->(b)
        """
        session.run(query, rows=rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export or load Neo4j graph assets.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML.")
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load the generated tables into Neo4j after exporting them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = Neo4jFraudGraphLoader(args.config)
    tables = loader.build_node_and_relationship_tables()
    if args.load:
        loader.load_to_neo4j(tables)
        print("Exported Neo4j import tables and attempted database load.")
    else:
        print("Exported Neo4j import tables to data/processed/neo4j_import/")


if __name__ == "__main__":
    main()
