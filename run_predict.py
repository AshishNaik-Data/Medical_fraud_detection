"""Entry point to run inference on test data."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from predict import main

if __name__ == "__main__":
    main()
