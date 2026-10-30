#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Healthcare Provider Fraud Detection — Unix Launcher
#  Usage:
#    ./run.sh              → Full pipeline (train + neo4j + predict + dashboard)
#    ./run.sh --train      → Train only
#    ./run.sh --neo4j      → Neo4j setup + ingest only
#    ./run.sh --predict    → Predict only
#    ./run.sh --dashboard  → Dashboard only
#    ./run.sh --no-dashboard → Everything except dashboard
# ─────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")"

# Check Python
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo "ERROR: Python not found. Install Python 3.9+."
    exit 1
fi

PYTHON=$(command -v python3 || command -v python)

# Check key dependencies
if ! $PYTHON -c "import xgboost, lightgbm, catboost, streamlit" 2>/dev/null; then
    echo "Installing required packages..."
    $PYTHON -m pip install xgboost lightgbm catboost streamlit neo4j scikit-learn pandas pyvis
fi

# Run the pipeline
$PYTHON run.py "$@"
