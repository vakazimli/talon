"""Updates source reliability scores based on trade outcomes.

Uses exponential moving average:
  new_score = old_score * 0.85 + outcome_score * 0.15
  outcome_score: 1.0 for wins, 0.0 for losses, 0.5 for breakeven
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from src.db.database import get_session
from src.db.models import Alert, Outcome, Setup, SourceScore

logger = logging.getLogger(__name__)


def seed_source_scores() -> int:
    """Seed source_scores rows from config/sources.yaml (idempotent).

    Creates one row per top-level source plus one per declared subtype so
    /sources is meaningful from day one and the auto-disable/enable
    machinery has rows to update. Existing rows are left untouched.
    Returns the number of rows inserted.
    """
    import yaml

    cfg_path = Path(__file__).resolve().parent.parent.parent / "config" / "sources.yaml"
    if not cfg_path.exists():
        return 0
    try:
        data = yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        logger.exception("Failed to read sources.yaml for seeding.")
        return 0

    sources = data.get("sources", {}) or {}
    now = datetime.utcnow().isoformat()
    inserted = 0
    with get_session() as session:
        for name, meta in sources.items():
            meta = meta or {}
            rel = float(meta.get("reliability_score", 0.50))
            enabled = bool(meta.get("enabled", True))
            subtypes = meta.get("subtypes") or []
            targets = [(name, None)] + [(name, st) for st in subtypes]
            for src_name, src_sub in targets:
                exists = (
                    session.query(SourceScore)
                    .filter(SourceScore.source_name == src_name)
                    .filter(
                        SourceScore.source_subtype.is_(None)
                        if src_sub is None
                        else SourceScore.source_subtype == src_sub
                    )
                    .first()
                )
                if exists:
                    continue
                session.add(
                    SourceScore(
                        source_name=src_name,
                        source_subtype=src_sub,
                        total_signals=0,
                        winning_signals=0,
                        avg_pnl_pct=0.0,
                        reliability_score=rel,
                        enabled=enabled,
                        last_updated=now,
                    )
                )
                inserted += 1
        session.commit()
    if inserted:
        logger.info("Seeded %d source_scores rows from sources.yaml.", inserted)
    return inserted

EMA_ALPHA = 0.15

# Auto-disable thresholds: a signal source is taken offline if its reliability
# drops below DISABLE_THRESHOLD with at least DISABLE_MIN_SAMPLES observations.
DISABLE_THRESHOLD = 0.25
DISABLE_MIN_SAMPLES = 20
RECOVER_THRESHOLD = 0.40


class SourceEvaluator:
    def update_from_outcome(self, alert: Alert, outcome: Outcome) -> None:
        """Update source scores based on a resolved outcome."""
        pnl_pct = outcome.pnl_pct or 0

        if pnl_pct > 1.0:
            outcome_score = 1.0
        elif pnl_pct < -1.0:
            outcome_score = 0.0
        else:
            outcome_score = 0.5

        with get_session() as session:
            setup = session.query(Setup).filter(Setup.id == alert.setup_id).first()
            if not setup:
                return

            sources = []
            try:
                sources = json.loads(setup.sources_used or "[]")
            except (json.JSONDecodeError, TypeError):
                pass

            signals = []
            try:
                signals = json.loads(setup.raw_signals or "[]")
            except (json.JSONDecodeError, TypeError):
                pass

            all_sources = set(sources + signals)
            for source_name in all_sources:
                self._update_single_score(session, source_name, outcome_score, pnl_pct)

            session.commit()

    def _update_single_score(
        self, session, source_name: str, outcome_score: float, pnl_pct: float
    ) -> None:
        row = (
            session.query(SourceScore)
            .filter(
                (SourceScore.source_name == source_name)
                | (SourceScore.source_subtype == source_name)
            )
            .first()
        )

        if row is None:
            row = SourceScore(
                source_name=source_name,
                source_subtype=source_name,
                total_signals=0,
                winning_signals=0,
                avg_pnl_pct=0.0,
                reliability_score=0.50,
            )
            session.add(row)

        row.total_signals = (row.total_signals or 0) + 1
        if outcome_score >= 0.8:
            row.winning_signals = (row.winning_signals or 0) + 1

        old_avg = row.avg_pnl_pct or 0
        n = row.total_signals
        row.avg_pnl_pct = old_avg + (pnl_pct - old_avg) / n

        old_reliability = row.reliability_score or 0.50
        row.reliability_score = old_reliability * (1 - EMA_ALPHA) + outcome_score * EMA_ALPHA

        row.last_updated = datetime.utcnow().isoformat()

        # Auto-disable / re-enable based on track record.
        was_enabled = bool(row.enabled) if row.enabled is not None else True
        if (
            was_enabled
            and (row.total_signals or 0) >= DISABLE_MIN_SAMPLES
            and (row.reliability_score or 0) < DISABLE_THRESHOLD
        ):
            row.enabled = False
            row.disabled_reason = (
                f"auto: reliability {row.reliability_score:.2f} < "
                f"{DISABLE_THRESHOLD} after {row.total_signals} signals"
            )
            logger.warning("Auto-disabled source '%s': %s", source_name, row.disabled_reason)
        elif (not was_enabled) and (row.reliability_score or 0) >= RECOVER_THRESHOLD:
            row.enabled = True
            row.disabled_reason = None
            logger.info("Auto-re-enabled source '%s' (reliability=%.2f)",
                        source_name, row.reliability_score)

        logger.info(
            "Source '%s' updated: reliability=%.3f (was %.3f), total=%d, wins=%d, enabled=%s",
            source_name, row.reliability_score, old_reliability,
            row.total_signals, row.winning_signals, bool(row.enabled),
        )

    def apply_feedback_multiplier(
        self, alert_id: int, feedback_type: str
    ) -> None:
        """Adjust source scores based on user feedback."""
        multipliers = {
            "good": 1.05,
            "bad": 0.90,
            "too_late": 0.95,
            "late": 0.95,
            "wrong_direction": 0.85,
            "good_idea_bad_timing": 1.0,
            "skip": 1.0,
        }
        mult = multipliers.get(feedback_type, 1.0)
        if mult == 1.0:
            return

        with get_session() as session:
            alert = session.query(Alert).filter(Alert.id == alert_id).first()
            if not alert:
                return

            setup = session.query(Setup).filter(Setup.id == alert.setup_id).first()
            if not setup:
                return

            signals = []
            try:
                signals = json.loads(setup.raw_signals or "[]")
            except (json.JSONDecodeError, TypeError):
                pass

            for sig in signals:
                row = (
                    session.query(SourceScore)
                    .filter(
                        (SourceScore.source_name == sig)
                        | (SourceScore.source_subtype == sig)
                    )
                    .first()
                )
                if row:
                    old = row.reliability_score or 0.50
                    row.reliability_score = max(0.0, min(1.0, old * mult))
                    row.last_updated = datetime.utcnow().isoformat()

            session.commit()
