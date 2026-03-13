from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.common import load_config, set_matplotlib_cache  # noqa: E402

config = load_config(PROJECT_ROOT / "config" / "config.yaml")
set_matplotlib_cache(config)

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

try:  # noqa: E402
    from neo4j import GraphDatabase
    from neo4j.graph import Node, Path as Neo4jPath, Relationship
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    GraphDatabase = None
    Node = Neo4jPath = Relationship = None


NEO4J_GRAPH_QUERIES = [
    {
        "title": "Schema Visualization",
        "kind": "table",
        "query": "CALL db.schema.visualization()",
        "caption": "High-level node label and relationship type overview.",
    },
    {
        "title": "Doctor Claim Patient Network",
        "kind": "graph",
        "query": """
MATCH (d:Doctor)-[:SUBMITTED]->(c:Claim)<-[:HAS_CLAIM]-(p:Patient)
RETURN d, c, p
LIMIT 50
""".strip(),
        "caption": "Core project graph connecting doctors, claims, and patients.",
    },
    {
        "title": "Flagged Claims by Doctor",
        "kind": "graph",
        "query": """
MATCH (d:Doctor)-[:SUBMITTED]->(c:Claim)
WHERE c.fraud_prediction = 1
RETURN d, c
ORDER BY c.fraud_probability DESC
LIMIT 30
""".strip(),
        "caption": "Doctors connected to high-risk claims.",
    },
    {
        "title": "Flagged Claim Investigation Paths",
        "kind": "graph",
        "query": """
MATCH path = (d:Doctor)-[:SUBMITTED]->(c:Claim)<-[:HAS_CLAIM]-(p:Patient)
WHERE c.fraud_prediction = 1
RETURN path
LIMIT 30
""".strip(),
        "caption": "End-to-end suspicious claim investigation paths.",
    },
    {
        "title": "Specialty Mismatch Graph",
        "kind": "graph",
        "query": """
MATCH (d:Doctor)-[:SUBMITTED]->(c:Claim)
WHERE c.specialty_mismatch = 1
RETURN d, c
LIMIT 25
""".strip(),
        "caption": "Claims whose diagnosis pattern does not align with provider specialty.",
    },
    {
        "title": "Shared Patient Provider Network",
        "kind": "graph",
        "query": """
MATCH (d1:Doctor)-[:TREATED]->(p:Patient)<-[:TREATED]-(d2:Doctor)
WHERE d1 <> d2
RETURN d1, p, d2
LIMIT 50
""".strip(),
        "caption": "Doctors linked through shared patients.",
    },
    {
        "title": "Specialty Hierarchy Graph",
        "kind": "graph",
        "query": """
MATCH (d:Doctor)-[:SPECIALIZED_IN]->(s:Specialty)-[:HAS_SUBSPECIALTY]->(ss:Subspecialty)
RETURN d, s, ss
LIMIT 50
""".strip(),
        "caption": "Doctor to specialty to subspecialty lineage.",
    },
]

NEO4J_TABLE_QUERIES = [
    {
        "title": "Node Counts by Label",
        "query": """
MATCH (n)
RETURN labels(n) AS labels, count(*) AS count
ORDER BY count DESC
""".strip(),
    },
    {
        "title": "Relationship Counts by Type",
        "query": """
MATCH ()-[r]->()
RETURN type(r) AS relationship, count(*) AS count
ORDER BY count DESC
""".strip(),
    },
    {
        "title": "Top Suspicious Doctors",
        "query": """
MATCH (d:Doctor)-[:SUBMITTED]->(c:Claim)
WITH d, count(c) AS total_claims,
     sum(CASE WHEN c.fraud_prediction = 1 THEN 1 ELSE 0 END) AS suspicious_claims
RETURN d.doctor_id AS doctor_id, total_claims, suspicious_claims,
       round(toFloat(suspicious_claims) / total_claims, 3) AS suspicious_rate
ORDER BY suspicious_claims DESC, suspicious_rate DESC
LIMIT 20
""".strip(),
    },
    {
        "title": "Patients with Highest Claim Volume",
        "query": """
MATCH (p:Patient)-[:HAS_CLAIM]->(c:Claim)
WITH p, count(c) AS claim_count
ORDER BY claim_count DESC
LIMIT 10
RETURN p.patient_id AS patient_id, claim_count
""".strip(),
    },
    {
        "title": "Top Fraudulent Claims",
        "query": """
MATCH (c:Claim)
WHERE c.fraud_prediction = 1
RETURN c.claim_event_id AS claim_event_id,
       c.fraud_probability AS fraud_probability,
       c.fraud_reason_summary AS fraud_reason_summary
ORDER BY c.fraud_probability DESC
LIMIT 10
""".strip(),
    },
]


st.set_page_config(page_title="Medical Fraud Investigation Dashboard", layout="wide")
st.markdown(
    """
    <style>
      [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #f4ede1 0%, #fbfaf6 52%, #ffffff 100%);
      }
      .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
      }
      .metric-card {
        background: rgba(255, 255, 255, 0.76);
        border: 1px solid rgba(115, 79, 48, 0.14);
        border-radius: 16px;
        padding: 0.9rem 1rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_assets():
    processed_dir = Path(config["paths"]["processed_dir"])
    logs_dir = Path(config["paths"]["logs_dir"])
    predictions = pd.read_csv(processed_dir / "predictions.csv", parse_dates=["claim_time"])
    provider_summary = pd.read_csv(processed_dir / "provider_summary.csv")
    graph_nodes = pd.read_csv(processed_dir / "graph_nodes.csv")
    graph_edges = pd.read_csv(processed_dir / "graph_edges.csv")
    global_importance = pd.read_csv(processed_dir / "global_feature_importance.csv")
    evaluation_metrics = {}
    training_summary = {}
    evaluation_path = logs_dir / "evaluation_metrics.json"
    training_path = logs_dir / "training_summary.json"
    if evaluation_path.exists():
        evaluation_metrics = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if training_path.exists():
        training_summary = json.loads(training_path.read_text(encoding="utf-8"))
    return predictions, provider_summary, graph_nodes, graph_edges, global_importance, evaluation_metrics, training_summary


@st.cache_resource
def get_neo4j_driver():
    if GraphDatabase is None:
        return None
    neo4j_config = config["neo4j"]
    return GraphDatabase.driver(
        neo4j_config["uri"],
        auth=(neo4j_config["user"], neo4j_config["password"]),
    )


def build_claim_subgraph(claim_id: str, predictions: pd.DataFrame, graph_edges: pd.DataFrame) -> nx.Graph:
    row = predictions.loc[predictions["claim_event_id"] == claim_id].iloc[0]
    focus_nodes = {
        f"claim::{row['claim_event_id']}",
        f"doctor::{row['doctor_id']}",
        f"patient::{row['patient_id']}",
        f"specialty::{row['doctor_specialty']}",
        f"subspecialty::{row['doctor_subspecialty']}",
    }
    same_patient_claims = predictions.loc[predictions["patient_id"] == row["patient_id"], "claim_event_id"].head(5)
    same_doctor_claims = predictions.loc[predictions["doctor_id"] == row["doctor_id"], "claim_event_id"].head(5)
    focus_nodes.update({f"claim::{value}" for value in same_patient_claims})
    focus_nodes.update({f"claim::{value}" for value in same_doctor_claims})

    sub_edges = graph_edges.loc[
        graph_edges["source"].isin(focus_nodes) | graph_edges["target"].isin(focus_nodes)
    ].copy()
    graph = nx.Graph()
    for edge in sub_edges.itertuples(index=False):
        graph.add_edge(edge.source, edge.target, relationship=edge.relationship)
    return graph


def draw_network(graph: nx.Graph):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_facecolor("#fbfaf6")
    node_colors = []
    labels = {}
    color_map = {
        "doctor": "#ad5c2f",
        "patient": "#2b6f78",
        "claim": "#c28b20",
        "specialty": "#537d36",
        "subspecialty": "#8b5e83",
        "Doctor": "#ad5c2f",
        "Patient": "#2b6f78",
        "Claim": "#c28b20",
        "Specialty": "#537d36",
        "Subspecialty": "#8b5e83",
    }
    for node, attributes in graph.nodes(data=True):
        labels[node] = attributes.get("label", str(node).split("::", 1)[-1])
        kind = attributes.get("kind") or str(node).split("::", 1)[0]
        node_colors.append(color_map.get(kind, "#6b7280"))

    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No graph data returned.", ha="center", va="center")
        ax.axis("off")
        return fig

    positions = nx.spring_layout(graph, seed=42)
    nx.draw_networkx(
        graph,
        pos=positions,
        ax=ax,
        labels=labels,
        with_labels=True,
        node_color=node_colors,
        node_size=900,
        font_size=8,
        edge_color="#d4c8b6",
    )
    edge_labels = {}
    for left, right, attributes in graph.edges(data=True):
        if attributes.get("relationship"):
            edge_labels[(left, right)] = attributes["relationship"]
    if edge_labels:
        nx.draw_networkx_edge_labels(graph, pos=positions, edge_labels=edge_labels, ax=ax, font_size=7)
    ax.axis("off")
    return fig


def _neo4j_node_id(node: Node) -> str:
    labels = sorted(node.labels)
    label = labels[0] if labels else "Node"
    properties = dict(node.items())
    preferred_keys = [
        "doctor_id",
        "patient_id",
        "claim_event_id",
        "id",
        "name",
    ]
    for key in preferred_keys:
        if properties.get(key):
            return f"{label}:{properties[key]}"
    return f"{label}:{node.element_id}"


def _neo4j_node_label(node: Node) -> str:
    properties = dict(node.items())
    for key in ("doctor_id", "patient_id", "claim_event_id", "name", "id"):
        if properties.get(key):
            return str(properties[key])
    labels = sorted(node.labels)
    return labels[0] if labels else "node"


def _add_neo4j_node(graph: nx.Graph, node: Node) -> str:
    node_id = _neo4j_node_id(node)
    graph.add_node(
        node_id,
        label=_neo4j_node_label(node),
        kind=sorted(node.labels)[0] if node.labels else "Node",
    )
    return node_id


def _extract_graph_value(graph: nx.Graph, value: Any) -> None:
    if Node is not None and isinstance(value, Node):
        _add_neo4j_node(graph, value)
        return

    if Relationship is not None and isinstance(value, Relationship):
        start_id = _add_neo4j_node(graph, value.start_node)
        end_id = _add_neo4j_node(graph, value.end_node)
        graph.add_edge(start_id, end_id, relationship=value.type)
        return

    if Neo4jPath is not None and isinstance(value, Neo4jPath):
        for node in value.nodes:
            _add_neo4j_node(graph, node)
        for relationship in value.relationships:
            start_id = _add_neo4j_node(graph, relationship.start_node)
            end_id = _add_neo4j_node(graph, relationship.end_node)
            graph.add_edge(start_id, end_id, relationship=relationship.type)
        return

    if isinstance(value, (list, tuple)):
        for item in value:
            _extract_graph_value(graph, item)


def _make_display_value(value: Any) -> Any:
    if Node is not None and isinstance(value, Node):
        return {"labels": sorted(value.labels), **dict(value.items())}
    if Relationship is not None and isinstance(value, Relationship):
        return {
            "type": value.type,
            "start": _neo4j_node_label(value.start_node),
            "end": _neo4j_node_label(value.end_node),
        }
    if Neo4jPath is not None and isinstance(value, Neo4jPath):
        return " -> ".join(_neo4j_node_label(node) for node in value.nodes)
    if isinstance(value, list):
        return [_make_display_value(item) for item in value]
    return value


def run_neo4j_query(query: str) -> tuple[nx.Graph, pd.DataFrame]:
    driver = get_neo4j_driver()
    if driver is None:
        raise RuntimeError("Neo4j driver is not installed in this environment.")

    graph = nx.Graph()
    neo4j_config = config["neo4j"]
    with driver.session(database=neo4j_config["database"]) as session:
        result = session.run(query)
        records = list(result)

    rows = []
    for record in records:
        row = {}
        for key in record.keys():
            value = record[key]
            _extract_graph_value(graph, value)
            row[key] = _make_display_value(value)
        rows.append(row)
    return graph, pd.DataFrame(rows)


def render_neo4j_section():
    st.subheader("Neo4j Graph Explorer")
    st.caption("Live Cypher results from the local Neo4j database loaded for this project.")

    driver = get_neo4j_driver()
    if driver is None:
        st.warning("Neo4j driver is not installed, so live graph queries are unavailable.")
        return

    try:
        with driver.session(database=config["neo4j"]["database"]) as session:
            counts = {
                "nodes": session.run("MATCH (n) RETURN count(n) AS c").single()["c"],
                "relationships": session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"],
            }
    except Exception as exc:  # pragma: no cover - depends on local service
        st.error(f"Neo4j connection failed: {exc}")
        return

    metric_1, metric_2 = st.columns(2)
    metric_1.metric("Neo4j Nodes", f"{counts['nodes']:,}")
    metric_2.metric("Neo4j Relationships", f"{counts['relationships']:,}")

    graph_tab, table_tab = st.tabs(["Graph Queries", "Summary Tables"])

    with graph_tab:
        for query_info in NEO4J_GRAPH_QUERIES:
            with st.expander(query_info["title"], expanded=query_info["title"] == "Doctor Claim Patient Network"):
                st.caption(query_info["caption"])
                st.code(query_info["query"], language="cypher")
                graph, table = run_neo4j_query(query_info["query"])
                if query_info["kind"] == "graph":
                    st.pyplot(draw_network(graph), use_container_width=True)
                if not table.empty:
                    st.dataframe(table.head(50), use_container_width=True)

    with table_tab:
        for query_info in NEO4J_TABLE_QUERIES:
            with st.expander(query_info["title"], expanded=query_info["title"] == "Top Suspicious Doctors"):
                st.code(query_info["query"], language="cypher")
                _, table = run_neo4j_query(query_info["query"])
                if table.empty:
                    st.info("This query returned no rows.")
                else:
                    st.dataframe(table, use_container_width=True)


def render_model_metrics(evaluation_metrics: dict[str, Any], training_summary: dict[str, Any]):
    st.subheader("Model Performance")
    if not evaluation_metrics:
        st.info("Run model evaluation to display accuracy, F1-score, recall, and related metrics.")
        return

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Accuracy", f"{evaluation_metrics.get('accuracy', 0.0):.2%}")
    metric_2.metric("Precision", f"{evaluation_metrics.get('precision', 0.0):.2%}")
    metric_3.metric("Recall", f"{evaluation_metrics.get('recall', 0.0):.2%}")
    metric_4.metric("F1 Score", f"{evaluation_metrics.get('f1_score', 0.0):.2%}")

    metric_5, metric_6, metric_7, metric_8 = st.columns(4)
    metric_5.metric("ROC AUC", f"{evaluation_metrics.get('roc_auc', 0.0):.4f}")
    metric_6.metric("Log Loss", f"{evaluation_metrics.get('log_loss', 0.0):.4f}")
    metric_7.metric("Threshold", f"{evaluation_metrics.get('threshold', 0.0):.4f}")
    metric_8.metric("Validation Rows", f"{int(evaluation_metrics.get('validation_rows', 0)):,}")

    details = {
        "target_column": evaluation_metrics.get("target_column", "unknown"),
        "training_backend": training_summary.get("backend", "unknown"),
    }
    if training_summary.get("random_forest_metrics"):
        details["random_forest_roc_auc"] = round(
            float(training_summary["random_forest_metrics"].get("roc_auc", 0.0)), 4
        )
    st.caption(
        "Evaluation metrics are loaded from the saved pipeline outputs so the dashboard reflects the latest trained model."
    )
    st.json(details, expanded=False)


st.title("Medical Insurance Fraud Investigation Dashboard")
st.caption("Hybrid risk scoring that combines claim features, provider behavior, and graph-network signals.")

processed_dir = Path(config["paths"]["processed_dir"])
required_files = [
    processed_dir / "predictions.csv",
    processed_dir / "provider_summary.csv",
    processed_dir / "graph_nodes.csv",
    processed_dir / "graph_edges.csv",
    processed_dir / "global_feature_importance.csv",
]
missing = [path.name for path in required_files if not path.exists()]
if missing:
    st.error(
        "Run the pipeline first. Missing generated assets: "
        + ", ".join(missing)
    )
    st.stop()

predictions, provider_summary, graph_nodes, graph_edges, global_importance, evaluation_metrics, training_summary = load_assets()
top_n = st.sidebar.slider("Top suspicious doctors", min_value=5, max_value=50, value=config["dashboard"]["default_top_n"])
claim_choices = predictions.sort_values("fraud_probability", ascending=False)["claim_event_id"].head(200).tolist()
selected_claim = st.sidebar.selectbox("Investigate claim", claim_choices)

top_risk = predictions.sort_values("fraud_probability", ascending=False)
positive_rate = top_risk["fraud_prediction"].mean()

metric_1, metric_2, metric_3 = st.columns(3)
metric_1.metric("Average Fraud Probability", f"{top_risk['fraud_probability'].mean():.2%}")
metric_2.metric("Flagged Claims", f"{int(top_risk['fraud_prediction'].sum())}")
metric_3.metric("Flagged Rate", f"{positive_rate:.2%}")

render_model_metrics(evaluation_metrics, training_summary)

selected_row = predictions.loc[predictions["claim_event_id"] == selected_claim].iloc[0]
summary_col, network_col = st.columns([1.1, 1.2])

with summary_col:
    st.subheader("Claim Summary")
    st.write(
        {
            "claim_event_id": selected_row["claim_event_id"],
            "claim_group_id": selected_row["claim_group_id"],
            "patient_id": selected_row["patient_id"],
            "doctor_id": selected_row["doctor_id"],
            "doctor_specialty": selected_row["doctor_specialty"],
            "code_group_title": selected_row["code_group_title"],
            "fraud_probability": round(float(selected_row["fraud_probability"]), 4),
            "triggered_rules": selected_row["triggered_rules"],
        }
    )
    st.info(selected_row["fraud_reason_summary"])

    st.subheader("Top Fraud Features")
    top_feature_slice = global_importance.head(10).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top_feature_slice["feature_name"], top_feature_slice["importance"], color="#ad5c2f")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    ax.set_facecolor("#fbfaf6")
    st.pyplot(fig, use_container_width=True)

with network_col:
    st.subheader("Claim Network")
    claim_graph = build_claim_subgraph(selected_claim, predictions, graph_edges)
    st.pyplot(draw_network(claim_graph), use_container_width=True)

st.subheader("Suspicious Doctors")
display_doctors = provider_summary.head(top_n).copy()
display_doctors["suspicious_rate"] = (
    display_doctors["suspicious_claims"] / display_doctors["claim_count"].clip(lower=1)
).round(3)
st.dataframe(display_doctors, use_container_width=True)

st.subheader("High-Risk Claims")
display_claims = top_risk[
    [
        "claim_event_id",
        "doctor_id",
        "patient_id",
        "doctor_specialty",
        "code_group_title",
        "fraud_probability",
        "triggered_rules",
    ]
].head(25)
st.dataframe(display_claims, use_container_width=True)

render_neo4j_section()
