"""
Configuration for Healthcare Provider Fraud Detection Pipeline.
All paths, hyperparameters, and Neo4j settings in one place.
"""
import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # finaldataproject/
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Raw data files (CMS-style)
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_LABELS = RAW_DATA_DIR / "Train-1542865627584.csv"
TRAIN_BENEFICIARY = RAW_DATA_DIR / "Train_Beneficiarydata-1542865627584.csv"
TRAIN_INPATIENT = RAW_DATA_DIR / "Train_Inpatientdata-1542865627584.csv"
TRAIN_OUTPATIENT = RAW_DATA_DIR / "Train_Outpatientdata-1542865627584.csv"
TEST_LABELS = RAW_DATA_DIR / "Test-1542969243754.csv"
TEST_BENEFICIARY = RAW_DATA_DIR / "Test_Beneficiarydata-1542969243754.csv"
TEST_INPATIENT = RAW_DATA_DIR / "Test_Inpatientdata-1542969243754.csv"
TEST_OUTPATIENT = RAW_DATA_DIR / "Test_Outpatientdata-1542969243754.csv"

# ─── Neo4j ───────────────────────────────────────────────────────────────────
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "neo4jLocal123!")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")
NEO4J_BATCH_SIZE = 5000

# ─── Reproducibility ────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ─── Feature Engineering ────────────────────────────────────────────────────
# Chronic condition columns in beneficiary data (1=Yes, 2=No in raw data)
CHRONIC_COLS = [
    "ChronicCond_Alzheimer", "ChronicCond_Heartfailure",
    "ChronicCond_KidneyDisease", "ChronicCond_Cancer",
    "ChronicCond_ObstrPulmonary", "ChronicCond_Depression",
    "ChronicCond_Diabetes", "ChronicCond_IschemicHeart",
    "ChronicCond_Osteoporasis", "ChronicCond_rheumatoidarthritis",
    "ChronicCond_stroke",
]

DIAGNOSIS_COLS = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
PROCEDURE_COLS = [f"ClmProcedureCode_{i}" for i in range(1, 7)]
PHYSICIAN_COLS = ["AttendingPhysician", "OperatingPhysician", "OtherPhysician"]

DATE_COLS_INPATIENT = ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt"]
DATE_COLS_OUTPATIENT = ["ClaimStartDt", "ClaimEndDt"]

# ─── Model Hyperparameters ──────────────────────────────────────────────────
# XGBoost (primary) – conservative to prevent overfitting
XGBOOST_PARAMS = {
    "n_estimators": 1000,       # high, but early stopping controls actual count
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "colsample_bylevel": 0.7,
    "reg_alpha": 0.5,           # L1
    "reg_lambda": 2.0,          # L2
    "gamma": 0.3,               # min split loss
    "objective": "binary:logistic",
    "eval_metric": "aucpr",     # PR-AUC for imbalanced data
    "random_state": RANDOM_SEED,
    "verbosity": 0,
}

# LightGBM
LIGHTGBM_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 6,
    "num_leaves": 31,           # conservative (< 2^max_depth)
    "learning_rate": 0.05,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "min_split_gain": 0.1,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}

# CatBoost
CATBOOST_PARAMS = {
    "iterations": 1000,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
    "min_data_in_leaf": 20,
    "subsample": 0.8,
    "rsm": 0.7,                 # column sampling
    "random_seed": RANDOM_SEED,
    "verbose": 0,
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
}

# ─── Cross Validation ───────────────────────────────────────────────────────
CV_FOLDS = 5
EARLY_STOPPING_ROUNDS = 50

# ─── Threshold Tuning ───────────────────────────────────────────────────────
# Optimize threshold for best F1, but also report at various recall targets
RECALL_TARGETS = [0.70, 0.80, 0.90]

# ─── GPU Detection ──────────────────────────────────────────────────────────
def detect_gpu():
    """Detect NVIDIA GPU availability for XGBoost."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False

USE_GPU = detect_gpu()
