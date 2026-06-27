"""Wipe poisoned learning data after the outcome-tracker fix.

The pre-fix tracker compared underlying-ticker prices to option-contract
target/stop prices, so every outcome resolved as a fake "target hit". The
source_scores and performance_reviews tables were updated from those fakes.
This script clears all three so the new tracker starts from a clean slate.

Preserves: scans, setups, alerts, feedback, cost_log (real history).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.db.database import get_session, init_database
from src.db.models import Outcome, PerformanceReview, SourceScore


def main():
    init_database()
    with get_session() as session:
        n_outcomes = session.query(Outcome).count()
        n_sources = session.query(SourceScore).count()
        n_reviews = session.query(PerformanceReview).count()

        session.query(Outcome).delete()
        session.query(SourceScore).delete()
        session.query(PerformanceReview).delete()
        session.commit()

    print("Wiped poisoned learning data:")
    print(f"  outcomes:            {n_outcomes} rows deleted")
    print(f"  source_scores:       {n_sources} rows deleted")
    print(f"  performance_reviews: {n_reviews} rows deleted")
    print("Preserved: scans, setups, alerts, feedback, cost_log.")


if __name__ == "__main__":
    main()
