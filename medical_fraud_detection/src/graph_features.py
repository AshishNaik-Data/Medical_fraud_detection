from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import networkx as nx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.common import DEFAULT_CONFIG_PATH, ensure_project_dirs, load_config, minmax_scale


class FraudGraphFeatureBuilder:
    """Build provider-network graph features for claims and doctors."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        self.config = load_config(config_path)
        ensure_project_dirs(self.config)
        self.processed_dir = Path(self.config["paths"]["processed_dir"])

    def load_claims(self) -> pd.DataFrame:
        claims = pd.read_csv(self.processed_dir / "claims_enriched.csv", parse_dates=["claim_time"])
        return claims

    def build_graph_tables(self, claims: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        patients = claims[["patient_id"]].drop_duplicates().assign(
            node_id=lambda frame: "patient::" + frame["patient_id"],
            label=lambda frame: frame["patient_id"],
            node_type="Patient",
        )
        doctors = claims[["doctor_id", "doctor_specialty", "doctor_subspecialty"]].drop_duplicates().assign(
            node_id=lambda frame: "doctor::" + frame["doctor_id"],
            label=lambda frame: frame["doctor_id"],
            node_type="Doctor",
        )
        claim_nodes = claims[
            [
                "claim_event_id",
                "claim_group_id",
                "claim_time",
                "code_group_title",
                "primary_diagnosis_description",
            ]
        ].drop_duplicates().assign(
            node_id=lambda frame: "claim::" + frame["claim_event_id"],
            label=lambda frame: frame["claim_event_id"],
            node_type="Claim",
        )
        specialties = claims[["doctor_specialty"]].drop_duplicates().rename(columns={"doctor_specialty": "label"})
        specialties = specialties.assign(
            node_id=lambda frame: "specialty::" + frame["label"],
            node_type="Specialty",
        )
        subspecialties = claims[["doctor_subspecialty"]].drop_duplicates().rename(
            columns={"doctor_subspecialty": "label"}
        )
        subspecialties = subspecialties.assign(
            node_id=lambda frame: "subspecialty::" + frame["label"],
            node_type="Subspecialty",
        )

        nodes = pd.concat(
            [
                patients[["node_id", "label", "node_type"]],
                doctors[["node_id", "label", "node_type", "doctor_specialty", "doctor_subspecialty"]],
                claim_nodes[
                    [
                        "node_id",
                        "label",
                        "node_type",
                        "claim_group_id",
                        "claim_time",
                        "code_group_title",
                        "primary_diagnosis_description",
                    ]
                ],
                specialties[["node_id", "label", "node_type"]],
                subspecialties[["node_id", "label", "node_type"]],
            ],
            ignore_index=True,
            sort=False,
        )

        edges = pd.concat(
            [
                claims[["patient_id", "claim_event_id"]]
                .drop_duplicates()
                .assign(
                    source=lambda frame: "patient::" + frame["patient_id"],
                    target=lambda frame: "claim::" + frame["claim_event_id"],
                    relationship="HAS_CLAIM",
                    weight=1,
                )[["source", "target", "relationship", "weight"]],
                claims[["doctor_id", "claim_event_id"]]
                .drop_duplicates()
                .assign(
                    source=lambda frame: "doctor::" + frame["doctor_id"],
                    target=lambda frame: "claim::" + frame["claim_event_id"],
                    relationship="SUBMITTED",
                    weight=1,
                )[["source", "target", "relationship", "weight"]],
                claims[["doctor_id", "doctor_specialty"]]
                .drop_duplicates()
                .assign(
                    source=lambda frame: "doctor::" + frame["doctor_id"],
                    target=lambda frame: "specialty::" + frame["doctor_specialty"],
                    relationship="SPECIALIZED_IN",
                    weight=1,
                )[["source", "target", "relationship", "weight"]],
                claims[["doctor_specialty", "doctor_subspecialty"]]
                .drop_duplicates()
                .assign(
                    source=lambda frame: "specialty::" + frame["doctor_specialty"],
                    target=lambda frame: "subspecialty::" + frame["doctor_subspecialty"],
                    relationship="HAS_SUBSPECIALTY",
                    weight=1,
                )[["source", "target", "relationship", "weight"]],
                claims[["doctor_id", "patient_id"]]
                .drop_duplicates()
                .assign(
                    source=lambda frame: "doctor::" + frame["doctor_id"],
                    target=lambda frame: "patient::" + frame["patient_id"],
                    relationship="TREATED",
                    weight=1,
                )[["source", "target", "relationship", "weight"]],
            ],
            ignore_index=True,
        )
        return nodes, edges

    def build_doctor_projection(self, claims: pd.DataFrame) -> nx.Graph:
        graph = nx.Graph()
        for doctor_id in sorted(claims["doctor_id"].unique()):
            graph.add_node(doctor_id)

        shared_weights: Dict[Tuple[str, str], int] = defaultdict(int)
        patient_doctors = claims.groupby("patient_id")["doctor_id"].agg(lambda values: sorted(set(values)))
        for doctors in patient_doctors:
            for left, right in combinations(doctors, 2):
                shared_weights[(left, right)] += 1

        for (left, right), weight in shared_weights.items():
            graph.add_edge(left, right, weight=weight, distance=1.0 / weight)
        return graph

    def build_claim_features(self, claims: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        doctor_projection = self.build_doctor_projection(claims)

        degree_dict = dict(doctor_projection.degree())
        weighted_degree_dict = dict(doctor_projection.degree(weight="weight"))
        pagerank = nx.pagerank(doctor_projection, weight="weight") if doctor_projection.number_of_nodes() else {}
        if doctor_projection.number_of_nodes() <= 200:
            betweenness = nx.betweenness_centrality(doctor_projection, weight="distance", normalized=True)
        else:
            sample_size = min(40, doctor_projection.number_of_nodes())
            betweenness = nx.betweenness_centrality(
                doctor_projection,
                k=sample_size,
                weight="distance",
                normalized=True,
                seed=self.config["model"]["random_state"],
            )

        if hasattr(nx.community, "louvain_communities") and doctor_projection.number_of_edges() > 0:
            communities = nx.community.louvain_communities(
                doctor_projection,
                seed=self.config["model"]["random_state"],
                weight="weight",
            )
        else:
            communities = list(nx.community.greedy_modularity_communities(doctor_projection)) if doctor_projection.number_of_edges() else []

        community_lookup: Dict[str, int] = {}
        community_size_lookup: Dict[int, int] = {}
        community_density_lookup: Dict[int, float] = {}
        for community_id, members in enumerate(communities):
            member_list = sorted(members)
            for member in member_list:
                community_lookup[member] = community_id
            community_size_lookup[community_id] = len(member_list)
            community_density_lookup[community_id] = nx.density(doctor_projection.subgraph(member_list)) if len(member_list) > 1 else 0.0

        doctor_patients = claims.groupby("doctor_id")["patient_id"].agg(lambda values: set(values))
        patient_doctors = claims.groupby("patient_id")["doctor_id"].agg(lambda values: set(values))
        shared_patient_ratio = {}
        for doctor_id, patients in doctor_patients.items():
            shared = sum(1 for patient in patients if len(patient_doctors.get(patient, set())) > 1)
            shared_patient_ratio[doctor_id] = shared / max(len(patients), 1)

        doctor_claim_count = claims.groupby("doctor_id")["claim_event_id"].size()
        doctor_patient_degree = claims.groupby("doctor_id")["patient_id"].nunique()
        patient_claim_count = claims.groupby("patient_id")["claim_event_id"].size()
        patient_provider_overlap = claims.groupby("patient_id")["doctor_id"].nunique().sub(1).clip(lower=0)

        doctor_features = pd.DataFrame({"doctor_id": sorted(claims["doctor_id"].unique())})
        doctor_features["doctor_claim_count"] = doctor_features["doctor_id"].map(doctor_claim_count).fillna(0)
        doctor_features["doctor_patient_degree"] = doctor_features["doctor_id"].map(doctor_patient_degree).fillna(0)
        doctor_features["doctor_degree"] = doctor_features["doctor_id"].map(degree_dict).fillna(0)
        doctor_features["doctor_weighted_degree"] = doctor_features["doctor_id"].map(weighted_degree_dict).fillna(0)
        doctor_features["doctor_pagerank"] = doctor_features["doctor_id"].map(pagerank).fillna(0.0)
        doctor_features["doctor_betweenness"] = doctor_features["doctor_id"].map(betweenness).fillna(0.0)
        doctor_features["louvain_community"] = doctor_features["doctor_id"].map(community_lookup).fillna(-1).astype(int)
        doctor_features["provider_cluster_size"] = doctor_features["louvain_community"].map(community_size_lookup).fillna(1).astype(int)
        doctor_features["provider_network_density"] = doctor_features["louvain_community"].map(community_density_lookup).fillna(0.0)
        doctor_features["shared_patient_ratio"] = doctor_features["doctor_id"].map(shared_patient_ratio).fillna(0.0)

        doctor_features["doctor_centrality"] = (
            minmax_scale(doctor_features["doctor_degree"])
            + minmax_scale(doctor_features["doctor_pagerank"])
            + minmax_scale(doctor_features["doctor_betweenness"])
        ) / 3.0

        claim_graph_features = claims[["claim_event_id", "doctor_id", "patient_id"]].copy()
        claim_graph_features = claim_graph_features.merge(doctor_features, on="doctor_id", how="left")
        claim_graph_features["patient_claim_count"] = claim_graph_features["patient_id"].map(patient_claim_count).fillna(0)
        claim_graph_features["patient_provider_overlap"] = (
            claim_graph_features["patient_id"].map(patient_provider_overlap).fillna(0).astype(int)
        )
        claim_graph_features["community_risk_score"] = 0.0

        return claim_graph_features, doctor_features

    def run(self) -> Dict[str, pd.DataFrame]:
        claims = self.load_claims()
        nodes, edges = self.build_graph_tables(claims)
        claim_graph_features, doctor_graph_features = self.build_claim_features(claims)

        doctor_projection_edges = []
        projection = self.build_doctor_projection(claims)
        for source, target, attributes in projection.edges(data=True):
            doctor_projection_edges.append(
                {
                    "doctor_id_source": source,
                    "doctor_id_target": target,
                    "shared_patient_count": attributes.get("weight", 1),
                }
            )

        outputs = {
            "graph_nodes": nodes,
            "graph_edges": edges,
            "claim_graph_features": claim_graph_features,
            "doctor_graph_features": doctor_graph_features,
            "doctor_projection_edges": pd.DataFrame(doctor_projection_edges),
        }

        for name, frame in outputs.items():
            frame.to_csv(self.processed_dir / f"{name}.csv", index=False)
        return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate graph features for medical fraud detection.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = FraudGraphFeatureBuilder(args.config)
    outputs = builder.run()
    print(
        "Saved graph assets:",
        ", ".join(f"{name}={frame.shape}" for name, frame in outputs.items()),
    )


if __name__ == "__main__":
    main()
