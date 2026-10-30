"""
Feature engineering orchestrator: builds the final provider-level feature
matrix by combining tabular aggregates with graph features.

Anti-leakage safeguards:
  - No target information is used during feature construction.
  - Train and test go through identical code paths.
  - Graph features are built from ALL claims (train+test) to avoid
    information asymmetry, but NEVER use fraud labels.
  - Feature transformations are fit on train only, then applied to test.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import MODEL_DIR, RANDOM_SEED
from data_loader import build_provider_features, load_dataset
from utils import save_artifact, setup_logging

logger = setup_logging()


def _try_neo4j_features(claims: pd.DataFrame) -> pd.DataFrame | None:
    """Attempt to compute graph features via Neo4j; return None on failure."""
    try:
        from neo4j_graph import Neo4jGraphBuilder
        builder = Neo4jGraphBuilder()
        builder.connect()
        builder.setup_schema()
        builder.clear_graph()
        builder.ingest_data(claims)
        graph_feats = builder.compute_graph_features()
        builder.close()
        return graph_feats
    except Exception as e:
        logger.warning(f"Neo4j unavailable ({e}), falling back to offline graph features")
        return None


def build_features(use_neo4j: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Build the complete feature matrix for train and test.

    Returns:
        train_df: Provider-level features + target for train providers
        test_df: Provider-level features for test providers (no target)
        feature_names: Array of feature column names
    """
    # Load raw data
    train_claims, train_labels = load_dataset("train")
    test_claims, test_labels = load_dataset("test")

    # Build tabular provider-level features (independently for train/test)
    logger.info("Building tabular provider features...")
    train_tab = build_provider_features(train_claims)
    test_tab = build_provider_features(test_claims)

    # Build graph features from ALL claims to capture full network structure.
    # This is safe because graph features only capture structural relationships
    # (who treated whom, which codes were used) — NO fraud labels involved.
    logger.info("Building graph features...")
    all_claims = pd.concat([train_claims, test_claims], ignore_index=True)

    graph_feats = None
    if use_neo4j:
        graph_feats = _try_neo4j_features(all_claims)

    if graph_feats is None:
        from neo4j_graph import compute_graph_features_offline
        graph_feats = compute_graph_features_offline(all_claims)

    # Merge tabular + graph features
    train_df = train_tab.merge(graph_feats, on="Provider", how="left")
    test_df = test_tab.merge(graph_feats, on="Provider", how="left")

    # Attach labels to train
    train_df = train_df.merge(train_labels, on="Provider", how="inner")

    # Ensure test has no target column
    if "PotentialFraud" in test_df.columns:
        test_df = test_df.drop(columns=["PotentialFraud"])

    # Align columns: test must have same features as train (minus target)
    feature_cols = [c for c in train_df.columns if c not in ("Provider", "PotentialFraud")]

    # Add any columns missing in test (with 0)
    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = 0

    # Remove any extra columns from test
    test_df = test_df[["Provider"] + feature_cols]

    # Fill remaining NaN
    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    test_df[feature_cols] = test_df[feature_cols].fillna(0)

    # Replace inf with large finite values
    for df in [train_df, test_df]:
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    logger.info(
        f"Final feature matrix: {len(feature_cols)} features, "
        f"{len(train_df)} train providers, {len(test_df)} test providers"
    )

    return train_df, test_df, np.array(feature_cols)


def build_scaled_features(use_neo4j: bool = True):
    """
    Build features and fit a StandardScaler on train data.
    Returns scaled train/test arrays plus metadata.

    The scaler is fit ONLY on train data to prevent leakage.
    """
    train_df, test_df, feature_names = build_features(use_neo4j=use_neo4j)

    X_train = train_df[feature_names].values.astype(np.float32)
    y_train = train_df["PotentialFraud"].values.astype(np.int8)
    X_test = test_df[feature_names].values.astype(np.float32)
    train_providers = train_df["Provider"].values
    test_providers = test_df["Provider"].values

    # Fit scaler on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler for inference
    save_artifact(scaler, MODEL_DIR / "scaler.pkl")
    save_artifact(feature_names, MODEL_DIR / "feature_names.pkl")

    return {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_test": X_test_scaled,
        "train_providers": train_providers,
        "test_providers": test_providers,
        "feature_names": feature_names,
        "scaler": scaler,
        "train_df": train_df,
        "test_df": test_df,
    }
