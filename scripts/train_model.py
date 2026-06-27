"""Train the Talon ML win-probability model from accumulated history.

Usage:
    python -m scripts.train_model [--min-samples 150]

Trains on resolved alert outcomes + counterfactuals and persists
data/ml_model.json. Activates only if there are enough samples and the
model beats a coin flip on the training set.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.db.database import init_database
from src.learning.ml_model import MIN_SAMPLES, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Talon ML model")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    args = parser.parse_args()
    init_database()
    result = train(min_samples=args.min_samples)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
