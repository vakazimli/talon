"""Initialize the Talon database with all required tables."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.db.database import init_database


def main():
    session_factory = init_database()
    from src.learning.source_evaluator import seed_source_scores
    seeded = seed_source_scores()
    with session_factory() as session:
        tables = session.execute(
            __import__("sqlalchemy").text(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ).fetchall()
        print(f"Database ready. Tables created: {[t[0] for t in tables]}")
        print(f"Seeded {seeded} source_scores rows.")


if __name__ == "__main__":
    main()
