# Medical Insurance Fraud Detection System

Hybrid fraud detection project that combines claim-level machine learning with graph analytics for explainable medical insurance investigations.

## Run On A Fresh Windows Machine

This project now bootstraps itself for Windows:

- checks whether Python 3 is available
- downloads and installs Python 3.12 if it is missing
- creates or repairs `.venv`
- checks whether required libraries are already installed
- installs only the missing or changed dependencies
- starts the Streamlit dashboard

### Simplest command

From the project folder:

```cmd
start_project.cmd
```

If you prefer PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_project.ps1
```

### Install optional extras too

```cmd
start_project.cmd -InstallOptional
```

This includes:

- LightGBM
- SHAP
- Neo4j Python drivers

### Rebuild the environment from scratch

```cmd
start_project.cmd -ForceReinstall
```

## Rebuild The Full Pipeline

```powershell
powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -InstallOptional
```

To also load Neo4j tables into a running local Neo4j instance:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -InstallOptional -LoadNeo4j
```

## Main Windows Scripts

- `setup_local.ps1`: bootstraps Python, `.venv`, and dependency checks
- `run_dashboard.ps1`: ensures setup is valid, then starts Streamlit
- `run_pipeline.ps1`: ensures setup is valid, then runs the full ML pipeline
- `start_project.ps1`: one-command entry point for fresh Windows machines
- `start_project.cmd`: double-click or command prompt launcher

## Neo4j Notes

Neo4j settings can be overridden with environment variables:

- `NEO4J_URI`
- `NEO4J_USER` or `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`

## Default Runtime Layout

```text
medical_fraud_detection/
|-- config/
|-- dashboard/
|-- data/
|   |-- raw/
|   `-- processed/
|-- docs/
|-- logs/
|-- models/
|   `-- checkpoints/
|-- notebooks/
|-- src/
|-- requirements.txt
|-- requirements-optional.txt
|-- setup_local.ps1
|-- run_dashboard.ps1
|-- run_pipeline.ps1
|-- start_project.ps1
`-- start_project.cmd
```
