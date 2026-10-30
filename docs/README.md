# Healthcare Provider Fraud Detection

Production-grade provider-level fraud prediction pipeline using CMS-style claims data
with optional Neo4j graph features.

## Architecture

```
Raw CSVs → Data Loader → Feature Engineering → Training Pipeline → Predictions
                              ↑
                     Neo4j Graph Features
                     (or offline fallback)
```

**Models**: XGBoost (GPU) + LightGBM + CatBoost → Logistic Regression meta-learner
**Validation**: Stratified 5-fold CV with out-of-fold predictions
**Anti-overfitting**: Early stopping, L1/L2 regularization, subsampling, correlation pruning,
threshold tuning on OOF predictions only

## Setup (Windows + NVIDIA GPU)

### 1. Create environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install CUDA-compatible XGBoost

```powershell
# Requires CUDA 11.8+ installed (https://developer.nvidia.com/cuda-downloads)
pip install xgboost  # pip wheels include GPU support automatically since v2.0
```

### 3. Install remaining dependencies

```powershell
pip install -r requirements.txt
```

### 4. (Optional) Set up Neo4j

```powershell
# Download Neo4j Community: https://neo4j.com/download/
# Set environment variables:
$env:NEO4J_URI = "bolt://localhost:7687"
$env:NEO4J_USER = "neo4j"
$env:NEO4J_PASSWORD = "your-password"
```

### 5. Place data files

Ensure the `new/` folder at the project root contains:
- `Train-1542865627584.csv`
- `Train_Beneficiarydata-1542865627584.csv`
- `Train_Inpatientdata-1542865627584.csv`
- `Train_Outpatientdata-1542865627584.csv`
- `Test-1542969243754.csv`
- `Test_Beneficiarydata-1542969243754.csv`
- `Test_Inpatientdata-1542969243754.csv`
- `Test_Outpatientdata-1542969243754.csv`

## Running

### Train (CPU fallback if no GPU)

```powershell
cd healthcare_fraud_detection
python train.py              # Without Neo4j (uses offline graph features)
python train.py --neo4j      # With Neo4j
```

### Predict

```powershell
python predict.py            # Ensemble predictions
python predict.py --baseline # XGBoost-only predictions
```

### Outputs

- `models/baseline_xgboost.pkl` — Baseline model artifact
- `models/ensemble.pkl` — Ensemble model artifact
- `outputs/oof_predictions.csv` — Out-of-fold training predictions
- `outputs/test_predictions_ensemble.csv` — Test set predictions
- `outputs/training_summary.json` — Training metrics summary
- `logs/pipeline.log` — Full training log

## Anti-Overfitting Measures

| Safeguard | How |
|-----------|-----|
| No target leakage | Features built from claims only, never from labels |
| No train/test contamination | Separate data loading paths; scaler fit on train only |
| Early stopping | 50-round patience on validation PR-AUC |
| Regularization | L1=0.5, L2=2.0, gamma=0.3, max_depth=6 |
| Subsampling | row=0.8, column=0.7 per tree and per level |
| Conservative depth | max_depth=6, num_leaves=31 (< 2^6) |
| Correlation pruning | Drop features with |r| > 0.95 |
| Threshold tuning | Optimized on OOF predictions, not training data |
| No SMOTE | 506 fraud / 4904 legitimate — SMOTE would manufacture unrealistic providers |
| Calibration | Isotonic calibration of meta-learner probabilities |
| Stratified CV | Class balance maintained across all 5 folds |

## Adapting to Different Schemas

- **Different column names**: Edit `config.py` CHRONIC_COLS, DIAGNOSIS_COLS, etc.
- **Different date formats**: The loader uses `pd.to_datetime(errors="coerce")` — most formats work
- **Missing columns**: Features are computed with `.get()` / `if col in df.columns` checks
- **Additional claim types**: Add to `_load_claims()` in `data_loader.py`
- **Different label encoding**: Adjust the label encoding line in `load_dataset()`
