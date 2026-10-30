"""
Neo4j graph construction and feature extraction.

Graph Schema:
  (:Provider {id})
  (:Beneficiary {id, age, chronicCount, isDead})
  (:Physician {id})
  (:DiagnosisCode {code})
  (:ProcedureCode {code})

  (Provider)-[:SUBMITTED_CLAIM]->(Claim)   — not materialized as node, implicit
  (Provider)-[:TREATED]->(Beneficiary)
  (Provider)-[:EMPLOYED]->(Physician)
  (Beneficiary)-[:VISITED]->(Provider)
  (Physician)-[:WORKS_AT]->(Provider)
  (Provider)-[:USED_DIAGNOSIS]->(DiagnosisCode)
  (Provider)-[:USED_PROCEDURE]->(ProcedureCode)

Graph Features Computed:
  - Provider degree (beneficiary, physician, diagnosis, procedure counts)
  - Shared-beneficiary overlap between providers (Jaccard similarity)
  - Provider-to-provider shared-physician count
  - Concentration metrics (how focused a provider's diagnosis codes are)
  - Beneficiary "loyalty" (fraction of beneficiaries seen by only this provider)
  - Neighborhood density proxies
"""
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from config import (
    DIAGNOSIS_COLS, NEO4J_BATCH_SIZE, NEO4J_DATABASE, NEO4J_PASSWORD,
    NEO4J_URI, NEO4J_USER, PHYSICIAN_COLS, PROCEDURE_COLS,
)

logger = logging.getLogger("fraud_detection")


class Neo4jGraphBuilder:
    """Manages Neo4j connection, ingestion, and feature extraction."""

    def __init__(self, uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD,
                 database=NEO4J_DATABASE):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None

    def connect(self):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        # Verify connectivity
        self.driver.verify_connectivity()
        logger.info(f"Connected to Neo4j at {self.uri}")

    def close(self):
        if self.driver:
            self.driver.close()

    def _run(self, query: str, params: dict = None):
        with self.driver.session(database=self.database) as session:
            return session.run(query, params or {}).data()

    def _run_write(self, query: str, params: dict = None):
        with self.driver.session(database=self.database) as session:
            session.execute_write(lambda tx: tx.run(query, params or {}))

    def setup_schema(self):
        """Create constraints and indexes for efficient ingestion and queries."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Provider) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (b:Beneficiary) REQUIRE b.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (ph:Physician) REQUIRE ph.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:DiagnosisCode) REQUIRE d.code IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (pr:ProcedureCode) REQUIRE pr.code IS UNIQUE",
        ]
        for q in constraints:
            self._run_write(q)
        logger.info("Neo4j schema constraints created")

    def clear_graph(self):
        """Remove all nodes and relationships. Use with caution."""
        self._run_write("MATCH (n) DETACH DELETE n")
        logger.info("Neo4j graph cleared")

    def ingest_data(self, claims: pd.DataFrame):
        """
        Batch-ingest claim data into Neo4j graph.
        Builds provider-beneficiary-physician-code relationships.
        """
        logger.info(f"Ingesting {len(claims):,} claim rows into Neo4j...")

        # ─── Extract unique entities ─────────────────────────────────────
        providers = claims["Provider"].dropna().unique().tolist()
        beneficiaries = claims["BeneID"].dropna().unique().tolist()

        physicians = set()
        for col in PHYSICIAN_COLS:
            if col in claims.columns:
                physicians.update(claims[col].dropna().unique())
        physicians = list(physicians)

        diag_codes = set()
        for col in DIAGNOSIS_COLS:
            if col in claims.columns:
                diag_codes.update(claims[col].dropna().unique())
        diag_codes = list(diag_codes)

        proc_codes = set()
        for col in PROCEDURE_COLS:
            if col in claims.columns:
                proc_codes.update(claims[col].dropna().unique())
        proc_codes = list(proc_codes)

        # ─── Batch insert nodes ──────────────────────────────────────────
        self._batch_create_nodes("Provider", "id", providers)
        self._batch_create_nodes("Beneficiary", "id", beneficiaries)
        self._batch_create_nodes("Physician", "id", physicians)
        self._batch_create_nodes("DiagnosisCode", "code", diag_codes)
        self._batch_create_nodes("ProcedureCode", "code", proc_codes)

        # ─── Batch insert relationships ──────────────────────────────────
        # Provider -> Beneficiary (TREATED)
        prov_bene = (
            claims[["Provider", "BeneID"]].dropna()
            .drop_duplicates()
            .rename(columns={"Provider": "from", "BeneID": "to"})
        )
        self._batch_create_rels(
            "Provider", "id", "Beneficiary", "id", "TREATED",
            prov_bene[["from", "to"]].values.tolist()
        )

        # Provider -> Physician (EMPLOYED)
        for col in PHYSICIAN_COLS:
            if col not in claims.columns:
                continue
            prov_phys = (
                claims[["Provider", col]].dropna()
                .drop_duplicates()
                .rename(columns={"Provider": "from", col: "to"})
            )
            self._batch_create_rels(
                "Provider", "id", "Physician", "id", "EMPLOYED",
                prov_phys[["from", "to"]].values.tolist()
            )

        # Provider -> DiagnosisCode (USED_DIAGNOSIS)
        for col in DIAGNOSIS_COLS:
            if col not in claims.columns:
                continue
            prov_diag = (
                claims[["Provider", col]].dropna()
                .drop_duplicates()
                .rename(columns={"Provider": "from", col: "to"})
            )
            self._batch_create_rels(
                "Provider", "id", "DiagnosisCode", "code", "USED_DIAGNOSIS",
                prov_diag[["from", "to"]].values.tolist()
            )

        # Provider -> ProcedureCode (USED_PROCEDURE)
        for col in PROCEDURE_COLS:
            if col not in claims.columns:
                continue
            prov_proc = (
                claims[["Provider", col]].dropna()
                .drop_duplicates()
                .rename(columns={"Provider": "from", col: "to"})
            )
            self._batch_create_rels(
                "Provider", "id", "ProcedureCode", "code", "USED_PROCEDURE",
                prov_proc[["from", "to"]].values.tolist()
            )

        logger.info("Neo4j ingestion complete")

    def _batch_create_nodes(self, label: str, key: str, values: list):
        """UNWIND-based batch node creation."""
        query = f"""
        UNWIND $batch AS val
        MERGE (n:{label} {{{key}: val}})
        """
        for i in range(0, len(values), NEO4J_BATCH_SIZE):
            batch = values[i:i + NEO4J_BATCH_SIZE]
            self._run_write(query, {"batch": batch})
        logger.info(f"  Created {len(values):,} {label} nodes")

    def _batch_create_rels(self, from_label, from_key, to_label, to_key,
                           rel_type, pairs):
        """UNWIND-based batch relationship creation."""
        query = f"""
        UNWIND $batch AS pair
        MATCH (a:{from_label} {{{from_key}: pair[0]}})
        MATCH (b:{to_label} {{{to_key}: pair[1]}})
        MERGE (a)-[:{rel_type}]->(b)
        """
        for i in range(0, len(pairs), NEO4J_BATCH_SIZE):
            batch = pairs[i:i + NEO4J_BATCH_SIZE]
            self._run_write(query, {"batch": batch})
        logger.info(f"  Created {len(pairs):,} {rel_type} relationships")

    def compute_graph_features(self) -> pd.DataFrame:
        """
        Extract provider-level graph features using Cypher queries.
        Returns a DataFrame indexed by Provider.
        """
        features = {}

        # 1. Degree-based features
        degree_query = """
        MATCH (p:Provider)
        OPTIONAL MATCH (p)-[:TREATED]->(b:Beneficiary)
        OPTIONAL MATCH (p)-[:EMPLOYED]->(ph:Physician)
        OPTIONAL MATCH (p)-[:USED_DIAGNOSIS]->(d:DiagnosisCode)
        OPTIONAL MATCH (p)-[:USED_PROCEDURE]->(pr:ProcedureCode)
        RETURN p.id AS Provider,
               count(DISTINCT b) AS graph_beneficiary_degree,
               count(DISTINCT ph) AS graph_physician_degree,
               count(DISTINCT d) AS graph_diagnosis_degree,
               count(DISTINCT pr) AS graph_procedure_degree
        """
        result = self._run(degree_query)
        if result:
            df = pd.DataFrame(result)
            features["degree"] = df
            logger.info(f"  Extracted degree features for {len(df)} providers")

        # 2. Shared-beneficiary overlap (provider-to-provider similarity)
        #    For each provider, count how many other providers share beneficiaries
        shared_bene_query = """
        MATCH (p1:Provider)-[:TREATED]->(b:Beneficiary)<-[:TREATED]-(p2:Provider)
        WHERE p1.id < p2.id
        WITH p1.id AS prov1, p2.id AS prov2, count(DISTINCT b) AS shared_benes
        WITH prov1, collect({partner: prov2, shared: shared_benes}) AS overlaps
        RETURN prov1 AS Provider,
               size(overlaps) AS graph_shared_bene_partners,
               reduce(s = 0, o IN overlaps | s + o.shared) AS graph_total_shared_benes,
               reduce(s = 0, o IN overlaps | CASE WHEN o.shared > s THEN o.shared ELSE s END) AS graph_max_shared_benes
        """
        result = self._run(shared_bene_query)
        if result:
            features["shared_bene"] = pd.DataFrame(result)
            logger.info(f"  Extracted shared-beneficiary features")

        # Also get from p2 side
        shared_bene_query2 = """
        MATCH (p1:Provider)-[:TREATED]->(b:Beneficiary)<-[:TREATED]-(p2:Provider)
        WHERE p1.id > p2.id
        WITH p1.id AS prov1, p2.id AS prov2, count(DISTINCT b) AS shared_benes
        WITH prov1, collect({partner: prov2, shared: shared_benes}) AS overlaps
        RETURN prov1 AS Provider,
               size(overlaps) AS graph_shared_bene_partners,
               reduce(s = 0, o IN overlaps | s + o.shared) AS graph_total_shared_benes,
               reduce(s = 0, o IN overlaps | CASE WHEN o.shared > s THEN o.shared ELSE s END) AS graph_max_shared_benes
        """
        result2 = self._run(shared_bene_query2)
        if result2:
            df2 = pd.DataFrame(result2)
            if "shared_bene" in features:
                # Combine both sides
                combined = pd.concat([features["shared_bene"], df2], ignore_index=True)
                combined = combined.groupby("Provider").sum().reset_index()
                features["shared_bene"] = combined

        # 3. Shared-physician overlap
        shared_phys_query = """
        MATCH (p1:Provider)-[:EMPLOYED]->(ph:Physician)<-[:EMPLOYED]-(p2:Provider)
        WHERE p1.id <> p2.id
        WITH p1.id AS Provider, count(DISTINCT p2) AS graph_shared_phys_partners,
             count(DISTINCT ph) AS graph_shared_physicians
        RETURN Provider, graph_shared_phys_partners, graph_shared_physicians
        """
        result = self._run(shared_phys_query)
        if result:
            features["shared_phys"] = pd.DataFrame(result)
            logger.info(f"  Extracted shared-physician features")

        # 4. Beneficiary loyalty (fraction who visit only this provider)
        loyalty_query = """
        MATCH (p:Provider)-[:TREATED]->(b:Beneficiary)
        WITH p.id AS Provider, b,
             size([(b)<-[:TREATED]-(p2:Provider) | p2]) AS num_providers
        WITH Provider, count(b) AS total_benes,
             sum(CASE WHEN num_providers = 1 THEN 1 ELSE 0 END) AS exclusive_benes
        RETURN Provider,
               toFloat(exclusive_benes) / total_benes AS graph_bene_exclusivity,
               total_benes AS graph_total_benes
        """
        result = self._run(loyalty_query)
        if result:
            features["loyalty"] = pd.DataFrame(result)
            logger.info(f"  Extracted beneficiary loyalty features")

        # 5. Diagnosis concentration (Herfindahl index of diagnosis codes)
        diag_conc_query = """
        MATCH (p:Provider)-[:USED_DIAGNOSIS]->(d:DiagnosisCode)
        WITH p.id AS Provider, d.code AS diag, count(*) AS cnt
        WITH Provider, collect(cnt) AS counts, sum(cnt) AS total
        WITH Provider, total,
             reduce(s = 0.0, c IN counts | s + (toFloat(c)/total)*(toFloat(c)/total)) AS hhi
        RETURN Provider, hhi AS graph_diag_concentration, total AS graph_diag_total_edges
        """
        result = self._run(diag_conc_query)
        if result:
            features["diag_conc"] = pd.DataFrame(result)
            logger.info(f"  Extracted diagnosis concentration features")

        # 6. Procedure concentration
        proc_conc_query = """
        MATCH (p:Provider)-[:USED_PROCEDURE]->(pr:ProcedureCode)
        WITH p.id AS Provider, pr.code AS proc, count(*) AS cnt
        WITH Provider, collect(cnt) AS counts, sum(cnt) AS total
        WITH Provider, total,
             reduce(s = 0.0, c IN counts | s + (toFloat(c)/total)*(toFloat(c)/total)) AS hhi
        RETURN Provider, hhi AS graph_proc_concentration, total AS graph_proc_total_edges
        """
        result = self._run(proc_conc_query)
        if result:
            features["proc_conc"] = pd.DataFrame(result)
            logger.info(f"  Extracted procedure concentration features")

        # ─── Merge all feature DataFrames ────────────────────────────────
        if not features:
            logger.warning("No graph features extracted!")
            return pd.DataFrame(columns=["Provider"])

        merged = features["degree"] if "degree" in features else pd.DataFrame()
        for key, df in features.items():
            if key == "degree":
                continue
            if not df.empty:
                merged = merged.merge(df, on="Provider", how="left")

        merged = merged.fillna(0)
        logger.info(f"Graph features: {merged.shape[1]-1} features for {len(merged)} providers")
        return merged


def compute_graph_features_offline(claims: pd.DataFrame) -> pd.DataFrame:
    """
    Compute graph-equivalent features using pure Python/pandas when Neo4j
    is unavailable. This is the fallback path.

    Uses vectorized pandas operations for efficiency instead of per-provider loops.
    """
    logger.info("Computing graph features offline (no Neo4j)...")

    providers = claims["Provider"].unique()
    feats = pd.DataFrame({"Provider": providers})

    # ─── Degree features (vectorized) ────────────────────────────────────
    bene_degree = claims.groupby("Provider")["BeneID"].nunique().reset_index()
    bene_degree.columns = ["Provider", "graph_beneficiary_degree"]
    feats = feats.merge(bene_degree, on="Provider", how="left")

    # Physician degree: count unique across all physician columns
    phys_dfs = []
    for col in PHYSICIAN_COLS:
        if col in claims.columns:
            sub = claims[["Provider", col]].dropna(subset=[col])
            sub = sub.rename(columns={col: "_phys"})
            phys_dfs.append(sub)
    if phys_dfs:
        all_phys = pd.concat(phys_dfs, ignore_index=True)
        phys_degree = all_phys.groupby("Provider")["_phys"].nunique().reset_index()
        phys_degree.columns = ["Provider", "graph_physician_degree"]
    else:
        phys_degree = pd.DataFrame({"Provider": providers, "graph_physician_degree": 0})
    feats = feats.merge(phys_degree, on="Provider", how="left")

    # Diagnosis degree
    diag_dfs = []
    for col in DIAGNOSIS_COLS:
        if col in claims.columns:
            sub = claims[["Provider", col]].dropna(subset=[col])
            sub = sub.rename(columns={col: "_code"})
            diag_dfs.append(sub)
    if diag_dfs:
        all_diag = pd.concat(diag_dfs, ignore_index=True)
        diag_degree = all_diag.groupby("Provider")["_code"].nunique().reset_index()
        diag_degree.columns = ["Provider", "graph_diagnosis_degree"]
    else:
        diag_degree = pd.DataFrame({"Provider": providers, "graph_diagnosis_degree": 0})
    feats = feats.merge(diag_degree, on="Provider", how="left")

    # Procedure degree
    proc_dfs = []
    for col in PROCEDURE_COLS:
        if col in claims.columns:
            sub = claims[["Provider", col]].dropna(subset=[col])
            sub = sub.rename(columns={col: "_code"})
            proc_dfs.append(sub)
    if proc_dfs:
        all_proc = pd.concat(proc_dfs, ignore_index=True)
        proc_degree = all_proc.groupby("Provider")["_code"].nunique().reset_index()
        proc_degree.columns = ["Provider", "graph_procedure_degree"]
    else:
        proc_degree = pd.DataFrame({"Provider": providers, "graph_procedure_degree": 0})
    feats = feats.merge(proc_degree, on="Provider", how="left")

    # ─── Shared-beneficiary features (vectorized via co-occurrence) ──────
    logger.info("  Computing shared-beneficiary overlap (vectorized)...")
    prov_bene = claims[["Provider", "BeneID"]].drop_duplicates()

    # Build beneficiary -> provider count (how many providers per bene)
    bene_prov_count = prov_bene.groupby("BeneID")["Provider"].nunique().reset_index()
    bene_prov_count.columns = ["BeneID", "_n_provs"]

    # For shared-bene: join prov_bene with itself on BeneID
    # Only keep beneficiaries seen by >1 provider (shared)
    shared_benes = bene_prov_count.loc[bene_prov_count["_n_provs"] > 1, "BeneID"]
    prov_bene_shared = prov_bene[prov_bene["BeneID"].isin(shared_benes)]

    if len(prov_bene_shared) > 0:
        # Self-join on BeneID to find provider pairs sharing beneficiaries
        pairs = prov_bene_shared.merge(prov_bene_shared, on="BeneID", suffixes=("_1", "_2"))
        pairs = pairs[pairs["Provider_1"] != pairs["Provider_2"]]
        pair_counts = pairs.groupby(["Provider_1", "Provider_2"])["BeneID"].nunique().reset_index()
        pair_counts.columns = ["Provider", "Partner", "SharedBenes"]

        shared_stats = pair_counts.groupby("Provider").agg(
            graph_shared_bene_partners=("Partner", "nunique"),
            graph_total_shared_benes=("SharedBenes", "sum"),
            graph_max_shared_benes=("SharedBenes", "max"),
        ).reset_index()
    else:
        shared_stats = pd.DataFrame({
            "Provider": providers,
            "graph_shared_bene_partners": 0,
            "graph_total_shared_benes": 0,
            "graph_max_shared_benes": 0,
        })
    feats = feats.merge(shared_stats, on="Provider", how="left")

    # ─── Shared-physician features (vectorized) ─────────────────────────
    logger.info("  Computing shared-physician overlap...")
    if phys_dfs:
        all_phys_prov = pd.concat(phys_dfs, ignore_index=True).drop_duplicates()
        # Physicians at >1 provider
        phys_count = all_phys_prov.groupby("_phys")["Provider"].nunique().reset_index()
        shared_phys_ids = phys_count.loc[phys_count["Provider"] > 1, "_phys"]
        phys_shared = all_phys_prov[all_phys_prov["_phys"].isin(shared_phys_ids)]

        if len(phys_shared) > 0:
            phys_pairs = phys_shared.merge(phys_shared, on="_phys", suffixes=("_1", "_2"))
            phys_pairs = phys_pairs[phys_pairs["Provider_1"] != phys_pairs["Provider_2"]]
            phys_stats = phys_pairs.groupby("Provider_1").agg(
                graph_shared_phys_partners=("Provider_2", "nunique"),
                graph_shared_physicians=("_phys", "nunique"),
            ).reset_index().rename(columns={"Provider_1": "Provider"})
        else:
            phys_stats = pd.DataFrame({
                "Provider": providers,
                "graph_shared_phys_partners": 0,
                "graph_shared_physicians": 0,
            })
    else:
        phys_stats = pd.DataFrame({
            "Provider": providers,
            "graph_shared_phys_partners": 0,
            "graph_shared_physicians": 0,
        })
    feats = feats.merge(phys_stats, on="Provider", how="left")

    # ─── Beneficiary exclusivity (vectorized) ────────────────────────────
    prov_bene_ex = prov_bene.merge(bene_prov_count, on="BeneID", how="left")
    prov_bene_ex["_exclusive"] = (prov_bene_ex["_n_provs"] == 1).astype(int)
    excl = prov_bene_ex.groupby("Provider").agg(
        _total=("BeneID", "count"),
        _excl=("_exclusive", "sum"),
    ).reset_index()
    excl["graph_bene_exclusivity"] = excl["_excl"] / excl["_total"]
    feats = feats.merge(excl[["Provider", "graph_bene_exclusivity"]], on="Provider", how="left")

    # ─── Diagnosis concentration (Herfindahl index, vectorized) ──────────
    if diag_dfs:
        all_diag_counts = all_diag.groupby(["Provider", "_code"]).size().reset_index(name="cnt")
        diag_totals = all_diag_counts.groupby("Provider")["cnt"].sum().reset_index(name="total")
        all_diag_counts = all_diag_counts.merge(diag_totals, on="Provider")
        all_diag_counts["share_sq"] = (all_diag_counts["cnt"] / all_diag_counts["total"]) ** 2
        diag_hhi = all_diag_counts.groupby("Provider")["share_sq"].sum().reset_index()
        diag_hhi.columns = ["Provider", "graph_diag_concentration"]
    else:
        diag_hhi = pd.DataFrame({"Provider": providers, "graph_diag_concentration": 0.0})
    feats = feats.merge(diag_hhi, on="Provider", how="left")

    # ─── Procedure concentration ─────────────────────────────────────────
    if proc_dfs:
        all_proc_counts = all_proc.groupby(["Provider", "_code"]).size().reset_index(name="cnt")
        proc_totals = all_proc_counts.groupby("Provider")["cnt"].sum().reset_index(name="total")
        all_proc_counts = all_proc_counts.merge(proc_totals, on="Provider")
        all_proc_counts["share_sq"] = (all_proc_counts["cnt"] / all_proc_counts["total"]) ** 2
        proc_hhi = all_proc_counts.groupby("Provider")["share_sq"].sum().reset_index()
        proc_hhi.columns = ["Provider", "graph_proc_concentration"]
    else:
        proc_hhi = pd.DataFrame({"Provider": providers, "graph_proc_concentration": 0.0})
    feats = feats.merge(proc_hhi, on="Provider", how="left")

    feats = feats.fillna(0)
    logger.info(f"Offline graph features: {feats.shape[1]-1} features for {len(feats)} providers")
    return feats
