"""
Healthcare Provider Fraud Detection — Streamlit Dashboard
One-click runnable: python -m streamlit run app.py

Features:
  - Dataset overview & class distribution
  - Model training with live progress
  - Overfitting diagnosis (train vs val gaps)
  - Feature importance visualization
  - Interactive provider fraud lookup
  - Neo4j graph visualization (live or offline)
  - Threshold tuning simulator
  - Full metrics comparison
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Add src to path
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config import (
    MODEL_DIR, OUTPUT_DIR, RAW_DATA_DIR, RANDOM_SEED,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE,
)
from utils import load_artifact, compute_metrics

st.set_page_config(
    page_title="Healthcare Fraud Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 5px;
    }
    .metric-card h2 { margin: 0; font-size: 2em; }
    .metric-card p { margin: 5px 0 0 0; font-size: 0.9em; opacity: 0.9; }
    .good { color: #00c853; font-weight: bold; }
    .warn { color: #ff9100; font-weight: bold; }
    .bad { color: #ff1744; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 5px 5px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ─────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    """Load raw datasets."""
    train_labels = pd.read_csv(RAW_DATA_DIR / "Train-1542865627584.csv")
    test_labels = pd.read_csv(RAW_DATA_DIR / "Test-1542969243754.csv")
    train_bene = pd.read_csv(RAW_DATA_DIR / "Train_Beneficiarydata-1542865627584.csv")
    train_inp = pd.read_csv(RAW_DATA_DIR / "Train_Inpatientdata-1542865627584.csv")
    train_out = pd.read_csv(RAW_DATA_DIR / "Train_Outpatientdata-1542865627584.csv")
    return train_labels, test_labels, train_bene, train_inp, train_out


@st.cache_data
def load_oof_predictions():
    """Load OOF predictions."""
    path = OUTPUT_DIR / "oof_predictions_improved.csv"
    if path.exists():
        return pd.read_csv(path)
    path = OUTPUT_DIR / "oof_predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_test_predictions():
    """Load test set predictions."""
    path = OUTPUT_DIR / "test_predictions_ensemble.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_training_summary():
    """Load training summary."""
    for name in ["training_summary_improved.json", "training_summary.json"]:
        path = OUTPUT_DIR / name
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def get_neo4j_connection():
    """Try to connect to Neo4j."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        return driver
    except Exception:
        return None


def query_neo4j(driver, query, params=None):
    """Run a Cypher query and return results as DataFrame."""
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(query, params or {})
        return pd.DataFrame(result.data())


# ─── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("🏥 Fraud Detection")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "🔬 Model Analysis", "🔍 Provider Lookup",
     "🕸️ Neo4j Graph", "⚙️ Train Model", "📄 About"],
    index=0,
)

# Check Neo4j status
neo4j_driver = get_neo4j_connection()
if neo4j_driver:
    st.sidebar.success("✅ Neo4j Connected")
else:
    st.sidebar.warning("⚠️ Neo4j Offline — using cached data")

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset**: CMS Medicare Claims")
st.sidebar.markdown("**Providers**: 5,410 train / 1,353 test")
st.sidebar.markdown("**Fraud Rate**: 9.35%")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("Healthcare Provider Fraud Detection Dashboard")
    st.markdown("*Graph-enhanced ensemble learning with Neo4j-augmented features*")

    # Load data
    train_labels, test_labels, train_bene, train_inp, train_out = load_data()
    summary = load_training_summary()
    oof = load_oof_predictions()

    # ─── Top metrics ──────────────────────────────────────────────────────
    st.markdown("### Model Performance (Out-of-Fold Validation)")

    if summary:
        metrics = summary.get("improved_oof_metrics", summary.get("ensemble_oof_metrics", {}))
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
        c2.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.4f}")
        c3.metric("F1 Score", f"{metrics.get('f1', 0):.4f}")
        c4.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        c5.metric("Recall", f"{metrics.get('recall', 0):.4f}")

    st.markdown("---")

    # ─── Dataset Overview ─────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Class Distribution")
        fraud_counts = train_labels["PotentialFraud"].value_counts()
        fig = px.pie(
            values=fraud_counts.values,
            names=["Legitimate", "Fraudulent"],
            color_discrete_sequence=["#2ecc71", "#e74c3c"],
            hole=0.4,
        )
        fig.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Dataset Sizes")
        data_sizes = pd.DataFrame({
            "Dataset": ["Train Labels", "Test Labels", "Beneficiaries",
                        "Inpatient Claims", "Outpatient Claims"],
            "Rows": [len(train_labels), len(test_labels), len(train_bene),
                     len(train_inp), len(train_out)],
        })
        fig = px.bar(data_sizes, x="Dataset", y="Rows",
                     color_discrete_sequence=["#3498db"])
        fig.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # ─── OOF Distribution ────────────────────────────────────────────────
    if oof is not None:
        st.markdown("### Predicted Probability Distribution (Out-of-Fold)")
        col_name = "oof_ensemble" if "oof_ensemble" in oof.columns else oof.columns[-1]
        fig = px.histogram(
            oof, x=col_name, color="PotentialFraud",
            nbins=50, barmode="overlay", opacity=0.7,
            color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
            labels={col_name: "Predicted Fraud Probability", "PotentialFraud": "Actual Label"},
        )
        fig.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Model Analysis
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Model Analysis":
    st.title("Model Analysis & Overfitting Diagnosis")

    oof = load_oof_predictions()
    summary = load_training_summary()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overfitting Gap", "🎯 Threshold Tuning", "📊 Feature Importance",
        "🔄 Model Comparison", "📉 Calibration"
    ])

    # ─── Tab 1: Overfitting Gap ──────────────────────────────────────────
    with tab1:
        st.markdown("### Train vs Validation Gap Analysis")
        st.markdown("*A healthy model has small gaps. Large gaps = overfitting.*")

        gap_data = pd.DataFrame({
            "Model": ["XGBoost", "XGBoost", "LightGBM", "LightGBM", "CatBoost", "CatBoost"] * 2,
            "Version": ["Initial"] * 6 + ["Improved"] * 6,
            "Metric": ["ROC-AUC", "PR-AUC"] * 6,
            "Gap": [
                0.055, 0.295, 0.042, 0.194, 0.034, 0.158,
                0.048, 0.238, 0.020, 0.054, 0.018, 0.057,
            ]
        })

        fig = px.bar(
            gap_data, x="Model", y="Gap", color="Version",
            facet_col="Metric", barmode="group",
            color_discrete_map={"Initial": "#e74c3c", "Improved": "#2ecc71"},
        )
        fig.add_hline(y=0.05, line_dash="dash", line_color="orange",
                      annotation_text="Healthy threshold (ROC)")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        # Gap reduction table
        st.markdown("### Gap Reduction Summary")
        reduction_df = pd.DataFrame({
            "Model": ["XGBoost", "LightGBM", "CatBoost"],
            "Initial ROC Gap": [0.055, 0.042, 0.034],
            "Improved ROC Gap": [0.048, 0.020, 0.018],
            "ROC Reduction": ["13%", "53%", "47%"],
            "Initial PR Gap": [0.295, 0.194, 0.158],
            "Improved PR Gap": [0.238, 0.054, 0.057],
            "PR Reduction": ["19%", "72%", "64%"],
        })
        st.dataframe(reduction_df, use_container_width=True, hide_index=True)

    # ─── Tab 2: Threshold Tuning ─────────────────────────────────────────
    with tab2:
        st.markdown("### Interactive Threshold Tuning")

        if oof is not None:
            col_name = "oof_ensemble" if "oof_ensemble" in oof.columns else oof.columns[-1]
            y_true = oof["PotentialFraud"].values
            y_prob = oof[col_name].values

            threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.53, 0.01)

            m = compute_metrics(y_true, y_prob, threshold)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Precision", f"{m['precision']:.3f}")
            c2.metric("Recall", f"{m['recall']:.3f}")
            c3.metric("F1", f"{m['f1']:.3f}")
            c4.metric("Flagged Providers", f"{(y_prob >= threshold).sum()}")

            # PR curve
            from sklearn.metrics import precision_recall_curve, roc_curve
            prec, rec, thresholds_pr = precision_recall_curve(y_true, y_prob)

            col_a, col_b = st.columns(2)
            with col_a:
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                                            name="PR Curve", line=dict(color="#3498db")))
                # Mark current threshold
                curr_prec = m["precision"]
                curr_rec = m["recall"]
                fig_pr.add_trace(go.Scatter(
                    x=[curr_rec], y=[curr_prec], mode="markers",
                    marker=dict(size=15, color="red"), name=f"τ={threshold:.2f}"
                ))
                fig_pr.update_layout(title="Precision-Recall Curve",
                                     xaxis_title="Recall", yaxis_title="Precision",
                                     height=400)
                st.plotly_chart(fig_pr, use_container_width=True)

            with col_b:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                             name="ROC Curve", line=dict(color="#9b59b6")))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                             line=dict(dash="dash", color="gray"), name="Random"))
                fig_roc.update_layout(title=f"ROC Curve (AUC={m['roc_auc']:.4f})",
                                      xaxis_title="FPR", yaxis_title="TPR", height=400)
                st.plotly_chart(fig_roc, use_container_width=True)

            # Confusion matrix
            cm = np.array(m["confusion_matrix"])
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Predicted", y="Actual"),
                x=["Legitimate", "Fraud"], y=["Legitimate", "Fraud"],
                color_continuous_scale="RdYlGn_r",
            )
            fig_cm.update_layout(title=f"Confusion Matrix (τ={threshold:.2f})", height=400)
            st.plotly_chart(fig_cm, use_container_width=True)

    # ─── Tab 3: Feature Importance ────────────────────────────────────────
    with tab3:
        st.markdown("### Feature Importance (XGBoost Gain)")

        # Try loading from improved model first
        for model_name in ["improved_ensemble.pkl", "ensemble.pkl", "baseline_xgboost.pkl"]:
            model_path = MODEL_DIR / model_name
            if model_path.exists():
                artifact = load_artifact(model_path)
                importance = artifact.get("importance", pd.DataFrame())
                if not importance.empty:
                    break

        if not importance.empty:
            top_n = st.slider("Show top N features", 10, len(importance), 20)
            imp_top = importance.head(top_n).sort_values("importance")

            # Color graph features differently
            imp_top["type"] = imp_top["feature"].apply(
                lambda x: "Graph Feature" if "graph_" in str(x) else "Tabular Feature"
            )

            fig = px.bar(
                imp_top, x="importance", y="feature", orientation="h",
                color="type",
                color_discrete_map={"Tabular Feature": "#3498db", "Graph Feature": "#e74c3c"},
            )
            fig.update_layout(height=max(400, top_n * 25), yaxis_title="",
                              xaxis_title="Importance (Gain)")
            st.plotly_chart(fig, use_container_width=True)

    # ─── Tab 4: Model Comparison ──────────────────────────────────────────
    with tab4:
        st.markdown("### Initial vs Improved Ensemble")

        comp_data = pd.DataFrame({
            "Metric": ["ROC-AUC", "PR-AUC", "F1", "Precision", "Recall"],
            "Initial": [0.9533, 0.7010, 0.6808, 0.6384, 0.7292],
            "Improved": [0.9527, 0.7042, 0.6772, 0.6075, 0.7648],
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Initial", x=comp_data["Metric"],
                             y=comp_data["Initial"], marker_color="#e74c3c"))
        fig.add_trace(go.Bar(name="Improved", x=comp_data["Metric"],
                             y=comp_data["Improved"], marker_color="#2ecc71"))
        fig.update_layout(barmode="group", height=450, yaxis_range=[0.5, 1.0],
                          title="OOF Validation Metrics")
        st.plotly_chart(fig, use_container_width=True)

        st.info("""
        **Key Insight**: The improved model trades 3pp precision for 3.6pp recall gain.
        PR-AUC (the primary metric) improved from 0.7010 → 0.7042. The overfitting
        gap was reduced by up to 72% (LightGBM PR gap: 0.194 → 0.054).
        """)

    # ─── Tab 5: Calibration ──────────────────────────────────────────────
    with tab5:
        st.markdown("### Probability Calibration")

        if oof is not None:
            col_name = "oof_ensemble" if "oof_ensemble" in oof.columns else oof.columns[-1]
            y_true = oof["PotentialFraud"].values
            y_prob = oof[col_name].values

            from sklearn.calibration import calibration_curve
            try:
                frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode="lines+markers",
                                         name="Model", line=dict(color="#3498db")))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         line=dict(dash="dash", color="gray"),
                                         name="Perfect Calibration"))
                fig.update_layout(title="Calibration Curve",
                                  xaxis_title="Mean Predicted Probability",
                                  yaxis_title="Fraction Positive", height=450)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not compute calibration curve: {e}")

            cal_data = pd.DataFrame({
                "Model": ["XGBoost (base)", "Ensemble (calibrated)"],
                "Brier Score ↓": [0.0590, 0.0624],
                "ECE ↓": [0.0628, 0.1493],
            })
            st.dataframe(cal_data, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Provider Lookup
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Provider Lookup":
    st.title("Provider Fraud Risk Lookup")

    oof = load_oof_predictions()
    test_preds = load_test_predictions()

    # Combine available predictions
    all_preds = []
    if oof is not None:
        oof_display = oof.copy()
        oof_display["split"] = "Train (OOF)"
        all_preds.append(oof_display)
    if test_preds is not None:
        test_display = test_preds.copy()
        test_display["split"] = "Test"
        all_preds.append(test_display)

    if all_preds:
        combined = pd.concat(all_preds, ignore_index=True)

        # Search
        search = st.text_input("🔎 Search Provider ID (e.g., PRV51001)", "")

        if search:
            results = combined[combined["Provider"].str.contains(search, case=False, na=False)]
            if len(results) > 0:
                for _, row in results.iterrows():
                    prob_col = "oof_ensemble" if "oof_ensemble" in row.index else "fraud_probability"
                    prob = row.get(prob_col, row.get("oof_xgb", 0))

                    color = "🟢" if prob < 0.3 else "🟡" if prob < 0.6 else "🔴"
                    risk = "LOW" if prob < 0.3 else "MEDIUM" if prob < 0.6 else "HIGH"

                    st.markdown(f"### {color} Provider: {row['Provider']}")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Fraud Probability", f"{prob:.1%}")
                    c2.metric("Risk Level", risk)
                    actual = row.get("PotentialFraud", "N/A")
                    if actual != "N/A":
                        c3.metric("Actual Label", "FRAUD" if actual == 1 else "Legitimate")
                    else:
                        c3.metric("Split", row.get("split", "Unknown"))

                    st.markdown("---")
            else:
                st.warning(f"No provider found matching '{search}'")
        else:
            # Show top fraud risks
            st.markdown("### Top 20 Highest Fraud Risk Providers")
            prob_col = "oof_ensemble" if "oof_ensemble" in combined.columns else "fraud_probability"
            if prob_col in combined.columns:
                top_risk = combined.nlargest(20, prob_col)[
                    ["Provider", prob_col, "split"] +
                    ([c for c in ["PotentialFraud"] if c in combined.columns])
                ]
                top_risk.columns = [c.replace("oof_ensemble", "Fraud Probability").replace(
                    "fraud_probability", "Fraud Probability") for c in top_risk.columns]
                st.dataframe(top_risk, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Neo4j Graph
# ════════════════════════════════════════════════════════════════════════════
elif page == "🕸️ Neo4j Graph":
    st.title("Neo4j Provider Network Graph")

    if neo4j_driver:
        st.success("Connected to Neo4j — querying live graph data")

        tab_g1, tab_g2, tab_g3, tab_g4 = st.tabs([
            "📊 Graph Stats", "🔗 Provider Network", "🔍 Provider Detail", "📈 Graph Features"
        ])

        with tab_g1:
            st.markdown("### Graph Database Statistics")

            stats_query = """
            CALL {
                MATCH (p:Provider) RETURN 'Provider' AS label, count(p) AS count
                UNION ALL
                MATCH (b:Beneficiary) RETURN 'Beneficiary' AS label, count(b) AS count
                UNION ALL
                MATCH (ph:Physician) RETURN 'Physician' AS label, count(ph) AS count
                UNION ALL
                MATCH (d:DiagnosisCode) RETURN 'DiagnosisCode' AS label, count(d) AS count
                UNION ALL
                MATCH (pr:ProcedureCode) RETURN 'ProcedureCode' AS label, count(pr) AS count
            }
            RETURN label, count
            """
            try:
                stats = query_neo4j(neo4j_driver, stats_query)
                if not stats.empty:
                    fig = px.bar(stats, x="label", y="count",
                                 color_discrete_sequence=["#3498db"])
                    fig.update_layout(title="Node Counts by Type", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Graph is empty. Use the 'Train Model' page to ingest data into Neo4j.")
            except Exception as e:
                st.error(f"Query error: {e}")

        with tab_g2:
            st.markdown("### Shared-Beneficiary Provider Network")
            st.markdown("*Providers connected if they share patients*")

            limit = st.slider("Max providers to show", 20, 200, 50)

            network_query = """
            MATCH (p1:Provider)-[:TREATED]->(b:Beneficiary)<-[:TREATED]-(p2:Provider)
            WHERE p1.id < p2.id
            WITH p1.id AS prov1, p2.id AS prov2, count(DISTINCT b) AS shared
            WHERE shared >= 3
            RETURN prov1, prov2, shared
            ORDER BY shared DESC
            LIMIT $limit
            """
            try:
                edges = query_neo4j(neo4j_driver, network_query, {"limit": limit})
                if not edges.empty:
                    # Build network with pyvis
                    import streamlit.components.v1 as components
                    from pyvis.network import Network

                    net = Network(height="600px", width="100%", bgcolor="#0e1117",
                                  font_color="white")
                    net.barnes_hut(gravity=-3000, spring_length=200)

                    nodes = set(edges["prov1"].tolist() + edges["prov2"].tolist())
                    for n in nodes:
                        net.add_node(n, label=n[-6:], title=n, color="#3498db", size=15)

                    for _, row in edges.iterrows():
                        net.add_edge(row["prov1"], row["prov2"],
                                     value=int(row["shared"]),
                                     title=f"Shared: {row['shared']} patients")

                    html_path = str(OUTPUT_DIR / "neo4j_network.html")
                    net.save_graph(html_path)
                    with open(html_path, "r") as f:
                        components.html(f.read(), height=620)
                else:
                    st.info("No shared-beneficiary edges found. Ingest data first.")
            except Exception as e:
                st.error(f"Network query error: {e}")

        with tab_g3:
            st.markdown("### Provider Detail (Neo4j)")
            provider_id = st.text_input("Provider ID", "PRV51001")

            if provider_id:
                detail_query = """
                MATCH (p:Provider {id: $pid})
                OPTIONAL MATCH (p)-[:TREATED]->(b:Beneficiary)
                OPTIONAL MATCH (p)-[:EMPLOYED]->(ph:Physician)
                OPTIONAL MATCH (p)-[:USED_DIAGNOSIS]->(d:DiagnosisCode)
                OPTIONAL MATCH (p)-[:USED_PROCEDURE]->(pr:ProcedureCode)
                RETURN p.id AS Provider,
                       count(DISTINCT b) AS Beneficiaries,
                       count(DISTINCT ph) AS Physicians,
                       count(DISTINCT d) AS DiagnosisCodes,
                       count(DISTINCT pr) AS ProcedureCodes
                """
                try:
                    detail = query_neo4j(neo4j_driver, detail_query, {"pid": provider_id})
                    if not detail.empty:
                        row = detail.iloc[0]
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Beneficiaries", int(row["Beneficiaries"]))
                        c2.metric("Physicians", int(row["Physicians"]))
                        c3.metric("Diagnosis Codes", int(row["DiagnosisCodes"]))
                        c4.metric("Procedure Codes", int(row["ProcedureCodes"]))
                    else:
                        st.warning(f"Provider '{provider_id}' not found in graph")
                except Exception as e:
                    st.error(f"Query error: {e}")

        with tab_g4:
            st.markdown("### Graph-Derived Features Distribution")

            graph_query = """
            MATCH (p:Provider)
            OPTIONAL MATCH (p)-[:TREATED]->(b:Beneficiary)
            OPTIONAL MATCH (p)-[:EMPLOYED]->(ph:Physician)
            WITH p.id AS Provider,
                 count(DISTINCT b) AS bene_degree,
                 count(DISTINCT ph) AS phys_degree
            RETURN Provider, bene_degree, phys_degree
            ORDER BY bene_degree DESC
            LIMIT 500
            """
            try:
                graph_feats = query_neo4j(neo4j_driver, graph_query)
                if not graph_feats.empty:
                    fig = px.scatter(
                        graph_feats, x="bene_degree", y="phys_degree",
                        hover_data=["Provider"],
                        color_discrete_sequence=["#3498db"],
                        labels={"bene_degree": "Beneficiary Count",
                                "phys_degree": "Physician Count"},
                    )
                    fig.update_layout(title="Provider Degree Distribution", height=450)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Query error: {e}")

    else:
        st.warning("⚠️ Neo4j is not running. Showing offline graph visualization.")
        st.markdown("### Offline Provider Network (from cached features)")

        # Load offline graph features
        oof = load_oof_predictions()
        if oof is not None:
            st.markdown("#### Provider Risk Network")
            st.markdown("*Node size = fraud probability, Color = fraud risk level*")

            # Build a simple network from OOF data
            import streamlit.components.v1 as components

            try:
                from pyvis.network import Network

                # Load the claims data to build a mini network
                train_inp = pd.read_csv(RAW_DATA_DIR / "Train_Inpatientdata-1542865627584.csv")
                # Find top fraud-risk providers
                prob_col = "oof_ensemble" if "oof_ensemble" in oof.columns else oof.columns[-1]
                top_fraud = oof.nlargest(30, prob_col)

                # Find shared beneficiaries among top providers
                top_provs = set(top_fraud["Provider"].values)
                claims_top = train_inp[train_inp["Provider"].isin(top_provs)]

                net = Network(height="500px", width="100%", bgcolor="#0e1117",
                              font_color="white")
                net.barnes_hut(gravity=-2000)

                for _, row in top_fraud.iterrows():
                    prob = row[prob_col]
                    color = "#e74c3c" if prob > 0.7 else "#f39c12" if prob > 0.4 else "#2ecc71"
                    size = 10 + prob * 30
                    net.add_node(
                        row["Provider"], label=row["Provider"][-6:],
                        title=f"{row['Provider']}\nFraud Prob: {prob:.1%}",
                        color=color, size=size
                    )

                # Add edges for shared beneficiaries
                prov_bene = claims_top[["Provider", "BeneID"]].drop_duplicates()
                if len(prov_bene) > 0:
                    merged = prov_bene.merge(prov_bene, on="BeneID", suffixes=("_1", "_2"))
                    merged = merged[merged["Provider_1"] < merged["Provider_2"]]
                    edge_counts = merged.groupby(["Provider_1", "Provider_2"]).size().reset_index(name="shared")
                    edge_counts = edge_counts[edge_counts["shared"] >= 2]

                    for _, row in edge_counts.iterrows():
                        if row["Provider_1"] in top_provs and row["Provider_2"] in top_provs:
                            net.add_edge(
                                row["Provider_1"], row["Provider_2"],
                                value=int(row["shared"]),
                                title=f"Shared: {row['shared']} patients"
                            )

                html_path = str(OUTPUT_DIR / "offline_network.html")
                net.save_graph(html_path)
                with open(html_path, "r") as f:
                    components.html(f.read(), height=520)

                st.caption("🔴 High risk  🟡 Medium risk  🟢 Low risk | Edge thickness = shared patients")

            except Exception as e:
                st.error(f"Error building network: {e}")

        st.markdown("---")
        st.markdown("### Start Neo4j to enable live queries")
        st.code("""
# Install Neo4j (pick one):
brew install neo4j          # macOS
choco install neo4j         # Windows
docker run -p 7474:7474 -p 7687:7687 neo4j  # Docker

# Then set environment variables:
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your_password
        """, language="bash")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Train Model
# ════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Train Model":
    st.title("Train / Retrain Model")

    st.markdown("""
    Click below to run the full training pipeline:
    1. Load & merge 558K claims → 5,410 provider features
    2. Compute 12 graph features (offline or Neo4j)
    3. 3×5-fold CV with XGBoost, LightGBM, CatBoost
    4. Stacking meta-learner + calibration
    5. Save models and predictions
    """)

    col1, col2 = st.columns(2)

    with col1:
        model_type = st.radio("Model Version", ["Improved (recommended)", "Original"])

    with col2:
        use_neo4j = st.checkbox("Use Neo4j (requires running instance)", value=False)
        use_optuna = st.checkbox("Optuna tuning (adds ~60s)", value=False)

    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        progress = st.progress(0, "Initializing...")
        log_area = st.empty()

        import subprocess

        if model_type == "Improved (recommended)":
            cmd = [sys.executable, "run_train_improved.py"]
            if use_optuna:
                cmd.append("--optuna")
        else:
            cmd = [sys.executable, "run_train.py"]

        if use_neo4j:
            cmd.append("--neo4j")

        progress.progress(10, "Starting training pipeline...")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
                cwd=str(Path(__file__).resolve().parent)
            )

            progress.progress(100, "Training complete!")

            if result.returncode == 0:
                st.success("✅ Training completed successfully!")
                # Show output
                with st.expander("Training Log", expanded=True):
                    st.code(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)

                # Reload summary
                st.cache_data.clear()
                summary = load_training_summary()
                if summary:
                    st.markdown("### Results")
                    metrics = summary.get("improved_oof_metrics",
                                         summary.get("ensemble_oof_metrics", {}))
                    c1, c2, c3 = st.columns(3)
                    c1.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.4f}")
                    c2.metric("F1", f"{metrics.get('f1', 0):.4f}")
                    c3.metric("Training Time", f"{summary.get('training_time_seconds', 0):.1f}s")
            else:
                st.error("❌ Training failed")
                with st.expander("Error Log"):
                    st.code(result.stderr[-2000:])
        except subprocess.TimeoutExpired:
            st.error("Training timed out (>300s)")

    # Neo4j Ingestion
    st.markdown("---")
    st.markdown("### Neo4j Data Ingestion")

    if neo4j_driver:
        if st.button("📥 Ingest Data into Neo4j", use_container_width=True):
            with st.spinner("Ingesting claims data into Neo4j..."):
                try:
                    from data_loader import load_dataset
                    from neo4j_graph import Neo4jGraphBuilder

                    train_claims, _ = load_dataset("train")
                    test_claims, _ = load_dataset("test")
                    all_claims = pd.concat([train_claims, test_claims], ignore_index=True)

                    builder = Neo4jGraphBuilder()
                    builder.connect()
                    builder.setup_schema()
                    builder.clear_graph()
                    builder.ingest_data(all_claims)
                    builder.close()

                    st.success("✅ Data ingested into Neo4j successfully!")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
    else:
        st.info("Start Neo4j to enable data ingestion.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: About
# ════════════════════════════════════════════════════════════════════════════
elif page == "📄 About":
    st.title("About This Project")

    st.markdown("""
    ## Healthcare Provider Fraud Detection

    **Capstone Project** — Graph-Enhanced Ensemble Learning for Medicare Fraud Detection

    ### Architecture

    ```
    558K Claims → Provider Aggregation → 58 Tabular Features
                                          + 12 Graph Features (Neo4j)
                                          → Feature Selection (49 features)
                                          → XGBoost + LightGBM + CatBoost
                                          → Logistic Regression Meta-Learner
                                          → Isotonic Calibration
                                          → Fraud Probability
    ```

    ### Key Results

    | Metric | Improved Model |
    |--------|---------------|
    | ROC-AUC | 0.9527 |
    | **PR-AUC** | **0.7042** |
    | F1 | 0.6772 |
    | Precision | 60.8% |
    | **Recall** | **76.5%** |

    ### Anti-Overfitting Measures
    - 3×5 Repeated Stratified CV (15 evaluations)
    - Aggressive regularization (depth=4, L1=2.0, L2=5.0)
    - Feature selection: 70 → 49 features
    - Train-vs-validation gap tracking
    - Adversarial validation (AUC=0.488 ✓)
    - Class weights instead of SMOTE
    - Early stopping on PR-AUC

    ### References
    - Herland et al. 2018, *Journal of Big Data*
    - Johnson & Khoshgoftaar 2023, *SN Computer Science*
    - Bauder & Khoshgoftaar 2017, *IEEE ICMLA*
    """)

    st.markdown("---")
    st.markdown("### Project Structure")
    st.code("""
finaldataproject/
├── app.py                  ← This dashboard
├── run_train.py            ← Original training
├── run_train_improved.py   ← Improved training
├── run_predict.py          ← Inference
├── src/                    ← 8 source modules
├── data/raw/               ← 8 CSV files
├── models/                 ← Trained models (.pkl)
├── outputs/                ← Predictions & metrics
├── paper/                  ← IEEE LaTeX paper
└── logs/                   ← Pipeline logs
    """)
