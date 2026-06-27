"""Backfill outcome entries for alerts that don't have one yet."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.db.database import get_session, init_database
from src.db.models import Alert, Outcome


def main():
    init_database()
    with get_session() as session:
        alerts_without_outcomes = (
            session.query(Alert)
            .outerjoin(Outcome)
            .filter(Outcome.id.is_(None))
            .all()
        )

        count = 0
        for alert in alerts_without_outcomes:
            mid = ((alert.entry_price_low or 0) + (alert.entry_price_high or 0)) / 2
            outcome = Outcome(
                alert_id=alert.id,
                entry_filled=True,
                actual_entry_price=mid if mid > 0 else None,
            )
            session.add(outcome)
            count += 1

        session.commit()
        print(f"Backfilled {count} outcome entries.")


if __name__ == "__main__":
    main()
