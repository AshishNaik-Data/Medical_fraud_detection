# Healthcare Provider Fraud Detection

Graph-enhanced ensemble learning pipeline for Medicare provider fraud detection using Neo4j graph features and calibrated stacking of XGBoost, LightGBM, and CatBoost.

## Results

| Metric    | Baseline (XGBoost) | Ensemble (Stacking) |
|-----------|-------------------|---------------------|
| ROC-AUC   | 0.9452            | **0.9533**          |
| PR-AUC    | 0.6995            | **0.7010**          |
| F1        | 0.6661            | **0.6808**          |
| Precision | 0.6075            | **0.6384**          |
| Recall    | **0.7372**        | 0.7292              |

## Project Structure

```
finaldataproject/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── run_train.py                 # Training entry point
├── run_predict.py               # Inference entry point
├── run_train.bat                # Windows training script
├── run_predict.bat              # Windows inference script
├── src/                         # Source code
│   ├── config.py                # All paths, hyperparameters, settings
│   ├── data_loader.py           # Load and merge CMS CSV files
│   ├── feature_engineering.py   # 58 tabular + 12 graph features
│   ├── neo4j_graph.py           # Neo4j graph construction & feature extraction
│   ├── train.py                 # Training pipeline (CV, ensemble, calibration)
│   ├── predict.py               # Inference pipeline
│   └── utils.py                 # Logging, GPU detection, utilities
├── data/
│   └── raw/                     # 8 CMS-style CSV files
│       ├── Train-1542865627584.csv
│       ├── Train_Beneficiarydata-1542865627584.csv
│       ├── Train_Inpatientdata-1542865627584.csv
│       ├── Train_Outpatientdata-1542865627584.csv
│       ├── Test-1542969243754.csv
│       ├── Test_Beneficiarydata-1542969243754.csv
│       ├── Test_Inpatientdata-1542969243754.csv
│       └── Test_Outpatientdata-1542969243754.csv
├── models/                      # Trained model artifacts
│   ├── baseline_xgboost.pkl
│   ├── ensemble.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── outputs/                     # Predictions and metrics
│   ├── training_summary.json
│   ├── oof_predictions.csv
│   ├── test_predictions_ensemble.csv
│   └── prediction_summary_ensemble.json
├── logs/
│   └── pipeline.log
├── paper/                       # IEEE conference paper
│   ├── ieee_fraud_detection.tex
│   ├── build.sh
│   └── build.bat
└── docs/
    └── README.md                # Detailed technical documentation
```

## Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the model
```bash
python run_train.py
```

### Run predictions on test set
```bash
python run_predict.py
```

### Windows (double-click)
- `run_train.bat` - train
- `run_predict.bat` - predict

## Key Design Decisions

- **No SMOTE**: Uses class weights instead — synthetic oversampling on provider aggregates creates unrealistic profiles
- **PR-AUC as primary metric**: More informative than ROC-AUC at 9.35% fraud rate
- **Neo4j optional**: Graph features computed offline if Neo4j unavailable
- **7 anti-overfitting safeguards**: Stratified CV, early stopping, L1/L2, subsampling, correlation pruning, class weights, OOF threshold tuning
