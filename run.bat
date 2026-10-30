@echo off
REM ─────────────────────────────────────────────────────────────
REM  Healthcare Provider Fraud Detection — Windows Launcher
REM  Usage:
REM    run.bat              → Full pipeline (train + neo4j + predict + dashboard)
REM    run.bat --train      → Train only
REM    run.bat --neo4j      → Neo4j setup + ingest only
REM    run.bat --predict    → Predict only
REM    run.bat --dashboard  → Dashboard only
REM    run.bat --no-dashboard → Everything except dashboard
REM ─────────────────────────────────────────────────────────────

cd /d "%~dp0"

REM Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python not found in PATH.
    echo Install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

REM Check key dependencies
python -c "import xgboost, lightgbm, catboost, streamlit" 2>nul
if %errorlevel% neq 0 (
    echo Installing required packages...
    pip install xgboost lightgbm catboost streamlit neo4j scikit-learn pandas pyvis
)

REM Run the pipeline with all passed arguments
python run.py %*

if %errorlevel% neq 0 (
    echo.
    echo Pipeline encountered an error. Check logs\pipeline.log for details.
    pause
)
