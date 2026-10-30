"""Entry point for improved training pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from train_improved import main

if __name__ == "__main__":
    main()
