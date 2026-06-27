import logging
import os
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionFactory = None


def get_db_path() -> str:
    env_path = os.environ.get("TALON_DB_PATH", "")
    if env_path:
        return str(Path(env_path).resolve())
    return str(Path(__file__).resolve().parent.parent.parent / "data" / "talon.db")


def init_database(db_path: str | None = None) -> sessionmaker[Session]:
    """Initialize SQLite database, create tables if they don't exist."""
    global _engine, _SessionFactory

    db_path = db_path or get_db_path()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    _engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

    Base.metadata.create_all(_engine)
    _apply_lightweight_migrations(_engine)
    _SessionFactory = sessionmaker(bind=_engine)

    logger.info("Database initialized at %s", db_path)
    return _SessionFactory


def _apply_lightweight_migrations(engine) -> None:
    """Add new columns to existing tables when models gain fields.

    SQLAlchemy's create_all only creates missing tables; it does not ALTER
    existing ones. We do a single ADD COLUMN per known addition.
    """
    with engine.begin() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(source_scores)")).fetchall()}
        if "enabled" not in cols:
            conn.execute(text("ALTER TABLE source_scores ADD COLUMN enabled BOOLEAN DEFAULT 1"))
            logger.info("Migrated source_scores: added 'enabled' column")
        if "disabled_reason" not in cols:
            conn.execute(text("ALTER TABLE source_scores ADD COLUMN disabled_reason VARCHAR"))
            logger.info("Migrated source_scores: added 'disabled_reason' column")


def get_session() -> Session:
    """Get a new database session."""
    if _SessionFactory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _SessionFactory()


def get_engine():
    return _engine


def checkpoint_wal() -> None:
    """Force a WAL checkpoint and truncate, keeping the WAL file from growing
    unboundedly. Safe to call periodically; a no-op if the engine isn't ready.
    """
    if _engine is None:
        return
    try:
        with _engine.connect() as conn:
            conn.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
        logger.info("WAL checkpoint (TRUNCATE) complete.")
    except Exception:
        logger.exception("WAL checkpoint failed.")
