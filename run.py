"""
Healthcare Provider Fraud Detection — One-Click Full Pipeline

Usage:
    python run.py              # Run everything: train + neo4j + predict + dashboard
    python run.py --train      # Train only
    python run.py --neo4j      # Neo4j ingest only
    python run.py --predict    # Predict only
    python run.py --dashboard  # Dashboard only
    python run.py --no-dashboard  # Train + Neo4j + Predict, skip dashboard

This script:
  1. Runs the improved training pipeline (3x5 CV, 3 models, stacking)
  2. Sets up Neo4j (auto-detects, auto-resets password if needed)
  3. Ingests all claims data into Neo4j graph
  4. Runs predictions on test set
  5. Launches Streamlit dashboard in browser
"""
import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


def print_banner():
    print("\n" + "=" * 65)
    print("  Healthcare Provider Fraud Detection — Full Pipeline")
    print("  Graph-Enhanced Ensemble with Neo4j Integration")
    print("=" * 65 + "\n")


def print_step(step_num, total, msg):
    print(f"\n{'─' * 65}")
    print(f"  Step {step_num}/{total}: {msg}")
    print(f"{'─' * 65}\n")


# ─── Neo4j Helpers ────────────────────────────────────────────────────────────

def find_neo4j_binary():
    """Find neo4j executable on the system."""
    # Check PATH
    for cmd in ["neo4j", "neo4j.bat"]:
        try:
            result = subprocess.run(
                ["which", cmd] if platform.system() != "Windows" else ["where", cmd],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return cmd
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # Common install locations
    common_paths = []
    if platform.system() == "Windows":
        common_paths = [
            Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Neo4j" / "bin" / "neo4j.bat",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Neo4j Desktop",
        ]
    elif platform.system() == "Darwin":
        common_paths = [
            Path("/opt/homebrew/bin/neo4j"),
            Path("/usr/local/bin/neo4j"),
        ]
    else:
        common_paths = [
            Path("/usr/bin/neo4j"),
            Path("/usr/local/bin/neo4j"),
        ]

    for p in common_paths:
        if p.exists():
            return str(p)

    return None


def is_neo4j_running():
    """Check if Neo4j is responding."""
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:7474")
        resp = urllib.request.urlopen(req, timeout=3)
        return resp.status == 200
    except Exception:
        return False


def start_neo4j(neo4j_bin):
    """Start Neo4j if not running."""
    if is_neo4j_running():
        print("  ✓ Neo4j is already running")
        return True

    print(f"  Starting Neo4j ({neo4j_bin})...")
    try:
        result = subprocess.run(
            [neo4j_bin, "start"], capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            print(f"  ⚠ Neo4j start returned: {result.stderr.strip()}")

        # Wait for Neo4j to be ready
        for i in range(20):
            time.sleep(2)
            if is_neo4j_running():
                print("  ✓ Neo4j started successfully")
                return True
            print(f"  Waiting for Neo4j... ({i+1}/20)")

        print("  ✗ Neo4j failed to start within 40 seconds")
        return False
    except Exception as e:
        print(f"  ✗ Could not start Neo4j: {e}")
        return False


def test_neo4j_auth(uri, user, password):
    """Test if Neo4j credentials work."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


def reset_neo4j_password(neo4j_bin, new_password):
    """
    Reset Neo4j password when it's a fresh install or password expired.
    Tries multiple approaches.
    """
    print("  Attempting to reset Neo4j password...")

    # Approach 1: neo4j-admin set-initial-password (works on fresh install)
    admin_bin = neo4j_bin.replace("neo4j", "neo4j-admin") if neo4j_bin else "neo4j-admin"

    # Stop Neo4j first
    if neo4j_bin:
        subprocess.run([neo4j_bin, "stop"], capture_output=True, text=True, timeout=30)
        time.sleep(3)

    # Try set-initial-password
    try:
        result = subprocess.run(
            [admin_bin, "dbms", "set-initial-password", new_password],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            print("  ✓ Password set via neo4j-admin")
    except Exception:
        pass

    # Try server password command (newer Neo4j versions)
    try:
        result = subprocess.run(
            [admin_bin, "server", "password", new_password],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            print("  ✓ Password set via neo4j-admin server")
    except Exception:
        pass

    # Restart Neo4j
    if neo4j_bin:
        subprocess.run([neo4j_bin, "start"], capture_output=True, text=True, timeout=30)
        time.sleep(10)

    # Verify
    from config import NEO4J_URI, NEO4J_USER
    if test_neo4j_auth(NEO4J_URI, NEO4J_USER, new_password):
        print("  ✓ Neo4j password verified")
        return True

    # Approach 2: Try cypher-shell ALTER USER
    for old_pw in ["neo4j", "password", ""]:
        try:
            cmd = f'ALTER CURRENT USER SET PASSWORD FROM "{old_pw}" TO "{new_password}";'
            result = subprocess.run(
                ["cypher-shell", "-u", "neo4j", "-p", old_pw, cmd],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0:
                print(f"  ✓ Password changed from '{old_pw}' to configured password")
                return True
        except Exception:
            pass

    print("  ⚠ Could not reset password automatically")
    return False


def setup_neo4j():
    """Full Neo4j setup: find, start, authenticate, reset password if needed."""
    from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

    print("  Looking for Neo4j installation...")
    neo4j_bin = find_neo4j_binary()

    if not neo4j_bin:
        print("  ⚠ Neo4j not found on this system")
        print("    Install with: brew install neo4j (macOS)")
        print("                  choco install neo4j (Windows)")
        print("                  docker run -p 7474:7474 -p 7687:7687 neo4j (Docker)")
        print("  → Continuing without Neo4j (offline graph features will be used)")
        return False

    print(f"  ✓ Found Neo4j: {neo4j_bin}")

    # Start if not running
    if not start_neo4j(neo4j_bin):
        return False

    # Test authentication
    if test_neo4j_auth(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD):
        print("  ✓ Neo4j authentication successful")
        return True

    # Password doesn't work — try reset
    print("  ⚠ Neo4j authentication failed, attempting password reset...")
    if reset_neo4j_password(neo4j_bin, NEO4J_PASSWORD):
        # Wait and re-test
        time.sleep(5)
        if test_neo4j_auth(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD):
            print("  ✓ Neo4j password reset and verified")
            return True

    print("  ⚠ Could not authenticate with Neo4j")
    print(f"    Set password manually: neo4j-admin dbms set-initial-password {NEO4J_PASSWORD}")
    print("  → Continuing without Neo4j")
    return False


def ingest_neo4j():
    """Ingest claims data into Neo4j graph."""
    from data_loader import load_dataset
    from neo4j_graph import Neo4jGraphBuilder
    import pandas as pd

    print("  Loading claims data...")
    train_claims, _ = load_dataset("train")
    test_claims, _ = load_dataset("test")
    all_claims = pd.concat([train_claims, test_claims], ignore_index=True)
    print(f"  Total claims: {len(all_claims):,}")

    print("  Connecting to Neo4j...")
    builder = Neo4jGraphBuilder()
    builder.connect()
    builder.setup_schema()
    builder.clear_graph()

    print("  Ingesting into graph (this takes ~60s)...")
    builder.ingest_data(all_claims)

    # Compute and export graph features
    print("  Computing graph features from Neo4j...")
    graph_feats = builder.compute_graph_features()
    print(f"  ✓ Graph features: {graph_feats.shape[1]-1} features for {len(graph_feats)} providers")

    builder.close()
    print("  ✓ Neo4j ingestion complete")


# ─── Pipeline Steps ───────────────────────────────────────────────────────────

def run_training():
    """Run the improved training pipeline."""
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "run_train_improved.py")],
        cwd=str(PROJECT_ROOT),
        timeout=300,
    )
    if result.returncode != 0:
        print("  ✗ Training failed!")
        sys.exit(1)
    print("  ✓ Training complete")


def run_prediction():
    """Run prediction on test set."""
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "run_predict.py")],
        cwd=str(PROJECT_ROOT),
        timeout=120,
    )
    if result.returncode != 0:
        print("  ⚠ Prediction had issues (non-critical)")
    else:
        print("  ✓ Predictions saved")


def run_dashboard():
    """Launch Streamlit dashboard."""
    print("\n  Opening dashboard in browser...")
    print("  URL: http://localhost:8501")
    print("  Press Ctrl+C to stop the dashboard\n")

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run",
             str(PROJECT_ROOT / "app.py"),
             "--server.port", "8501",
             "--server.headless", "false",
             "--browser.gatherUsageStats", "false"],
            cwd=str(PROJECT_ROOT),
        )
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Healthcare Fraud Detection — Full Pipeline")
    parser.add_argument("--train", action="store_true", help="Run training only")
    parser.add_argument("--neo4j", action="store_true", help="Neo4j setup + ingest only")
    parser.add_argument("--predict", action="store_true", help="Run prediction only")
    parser.add_argument("--dashboard", action="store_true", help="Launch dashboard only")
    parser.add_argument("--no-dashboard", action="store_true", help="Run all steps except dashboard")
    parser.add_argument("--optuna", action="store_true", help="Enable Optuna hyperparameter tuning")
    args = parser.parse_args()

    print_banner()

    # If specific flags given, run only those steps
    run_specific = args.train or args.neo4j or args.predict or args.dashboard

    if run_specific:
        if args.train:
            print_step(1, 1, "Training Improved Model")
            run_training()
        if args.neo4j:
            print_step(1, 1, "Neo4j Setup & Data Ingestion")
            neo4j_ok = setup_neo4j()
            if neo4j_ok:
                ingest_neo4j()
        if args.predict:
            print_step(1, 1, "Running Predictions")
            run_prediction()
        if args.dashboard:
            run_dashboard()
        return

    # ─── Full Pipeline ────────────────────────────────────────────────────
    total_steps = 4 if not args.no_dashboard else 3
    start_time = time.time()

    # Step 1: Train
    print_step(1, total_steps, "Training Improved Model (3×5-fold CV)")
    run_training()

    # Step 2: Neo4j
    print_step(2, total_steps, "Neo4j Setup & Data Ingestion")
    neo4j_ok = setup_neo4j()
    if neo4j_ok:
        ingest_neo4j()
    else:
        print("  → Skipping Neo4j ingestion (offline features already computed)")

    # Step 3: Predict
    print_step(3, total_steps, "Running Predictions on Test Set")
    run_prediction()

    elapsed = time.time() - start_time
    print(f"\n{'=' * 65}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"{'=' * 65}")
    print(f"\n  Models saved:      {PROJECT_ROOT / 'models'}")
    print(f"  Predictions saved: {PROJECT_ROOT / 'outputs'}")
    print(f"  Logs:              {PROJECT_ROOT / 'logs' / 'pipeline.log'}")

    if neo4j_ok:
        print(f"  Neo4j:             http://localhost:7474")

    # Step 4: Dashboard
    if not args.no_dashboard:
        print_step(4, total_steps, "Launching Dashboard")
        run_dashboard()


if __name__ == "__main__":
    main()
