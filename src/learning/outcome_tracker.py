"""Tracks outcomes of every alert — shadow, paper, or live.

Records a hypothetical entry at the alert's mid price when sent,
polls the *option contract's* live price periodically (Tradier first,
yfinance options chain fallback), and records P&L at target/stop/expiry.
"""

import logging
from datetime import datetime

from src.db.database import get_session
from src.db.models import Alert, Outcome
from src.learning.source_evaluator import SourceEvaluator
from src.scanner.option_quotes import get_option_quote

logger = logging.getLogger(__name__)

# Max hold for a swing trade. Past this many calendar days without a
# target/stop hit, we close the outcome at the current price ("time_exit")
# so it matches the 1-3 day-hold thesis and feeds the learners promptly
# instead of sitting open until the contract expires.
MAX_HOLD_DAYS = 3


class OutcomeTracker:
    def __init__(self, config: dict):
        self.config = config
        self.source_evaluator = SourceEvaluator()

    async def record_entry(self, alert: Alert) -> None:
        """Record hypothetical entry when an alert is first sent."""
        mid = ((alert.entry_price_low or 0) + (alert.entry_price_high or 0)) / 2
        if mid <= 0:
            mid = alert.entry_price_low or alert.entry_price_high or 0

        with get_session() as session:
            existing = session.query(Outcome).filter(Outcome.alert_id == alert.id).first()
            if existing:
                return

            outcome = Outcome(
                alert_id=alert.id,
                entry_filled=True,
                actual_entry_price=mid if mid > 0 else None,
            )
            session.add(outcome)
            session.commit()
            logger.info("Entry recorded for alert #%d at $%.2f", alert.id, mid)

    async def check_all_open_outcomes(self) -> None:
        """Check option-contract prices for all unresolved outcomes."""
        with get_session() as session:
            open_outcomes = (
                session.query(Outcome)
                .filter(Outcome.resolved_at.is_(None))
                .filter(Outcome.entry_filled.is_(True))
                .all()
            )

            for outcome in open_outcomes:
                alert = session.query(Alert).filter(Alert.id == outcome.alert_id).first()
                if not alert:
                    continue

                try:
                    self._check_outcome(session, outcome, alert)
                except Exception:
                    logger.exception("Failed to check outcome for alert #%d", alert.id)

            session.commit()

    async def check_all_open_counterfactuals(self) -> None:
        """Resolve shadow (counterfactual) outcomes for non-alerted setups,
        using the same target/stop/expiry/time-exit logic as real outcomes."""
        from src.db.models import CounterfactualOutcome
        with get_session() as session:
            rows = (
                session.query(CounterfactualOutcome)
                .filter(CounterfactualOutcome.resolved_at.is_(None))
                .all()
            )
            for cf in rows:
                try:
                    self._check_counterfactual(cf)
                except Exception:
                    logger.exception("Failed to check counterfactual #%s", cf.id)
            session.commit()

    def _check_counterfactual(self, cf) -> None:
        entry = cf.entry_price or 0
        if entry <= 0:
            return
        current_price = self._get_option_price(cf.contract)
        expired = self._contract_expired(cf.contract)

        if current_price is None:
            if expired:
                cf.exit_reason = "expiry_unknown"
                cf.resolved_at = datetime.utcnow().isoformat()
            return

        if cf.max_favorable is None or current_price > cf.max_favorable:
            cf.max_favorable = current_price
        if cf.max_adverse is None or current_price < cf.max_adverse:
            cf.max_adverse = current_price

        target = cf.target_price or 0
        stop = cf.stop_price or 0
        resolved = False
        reason = None
        exit_price = current_price
        if target > 0 and current_price >= target:
            resolved, reason, exit_price = True, "target", target
        elif stop > 0 and current_price <= stop:
            resolved, reason, exit_price = True, "stop", stop
        elif expired:
            resolved, reason = True, "expiry"
        elif self._iso_older_than(cf.created_at, MAX_HOLD_DAYS):
            resolved, reason, exit_price = True, "time_exit", current_price

        if resolved:
            cf.actual_exit_price = exit_price
            cf.exit_reason = reason
            cf.pnl_pct = ((exit_price / entry) - 1) * 100 if entry > 0 else 0
            cf.resolved_at = datetime.utcnow().isoformat()
            logger.info(
                "Counterfactual #%s resolved: %s, P&L=%.1f%%",
                cf.id, reason, cf.pnl_pct or 0,
            )

    def _check_outcome(self, session, outcome: Outcome, alert: Alert) -> None:
        entry = outcome.actual_entry_price or 0
        if entry <= 0:
            return

        current_price = self._get_option_price(alert.contract)
        expired = self._is_expired(alert)

        if current_price is None:
            if expired:
                # No quote available and the contract has expired —
                # record an unknown-resolution row so we stop polling it.
                outcome.exit_reason = "expiry_unknown"
                outcome.resolved_at = datetime.utcnow().isoformat()
                self._set_hold_duration(outcome, alert)
            return

        if outcome.max_favorable is None or current_price > outcome.max_favorable:
            outcome.max_favorable = current_price
        if outcome.max_adverse is None or current_price < outcome.max_adverse:
            outcome.max_adverse = current_price

        target = alert.target_price or 0
        stop = alert.stop_price or 0

        resolved = False
        exit_reason = None
        exit_price = current_price

        if target > 0 and current_price >= target:
            resolved = True
            exit_reason = "target"
            exit_price = target
        elif stop > 0 and current_price <= stop:
            resolved = True
            exit_reason = "stop"
            exit_price = stop
        elif expired:
            resolved = True
            exit_reason = "expiry"
        elif self._held_longer_than(alert, MAX_HOLD_DAYS):
            # Swing window elapsed without hitting target/stop — close at
            # the current mark so the trade feeds learning on schedule.
            resolved = True
            exit_reason = "time_exit"
            exit_price = current_price

        if resolved:
            outcome.actual_exit_price = exit_price
            outcome.exit_reason = exit_reason
            outcome.pnl_dollars = exit_price - entry
            outcome.pnl_pct = ((exit_price / entry) - 1) * 100 if entry > 0 else 0
            outcome.resolved_at = datetime.utcnow().isoformat()
            self._set_hold_duration(outcome, alert)

            logger.info(
                "Outcome resolved for alert #%d: %s, P&L=%.1f%%",
                alert.id, exit_reason, outcome.pnl_pct or 0,
            )

            self.source_evaluator.update_from_outcome(alert, outcome)

    def _get_option_price(self, contract: str) -> float | None:
        """Live quote for the specific option contract (Tradier preferred)."""
        if not contract:
            return None
        quote = get_option_quote(contract)
        if quote is None:
            return None
        return quote.mid if quote.mid > 0 else (quote.last if quote.last > 0 else None)

    def _is_expired(self, alert: Alert) -> bool:
        """Check if the option contract has expired."""
        return self._contract_expired(alert.contract or "")

    def _contract_expired(self, contract: str) -> bool:
        """True if the contract's expiry date is in the past."""
        contract = contract or ""
        parts = contract.split()
        if len(parts) < 3:
            return False
        date_part = parts[-1]
        try:
            from datetime import date as dt_date
            exp_parts = date_part.split("/")
            if len(exp_parts) == 2:
                month, day = int(exp_parts[0]), int(exp_parts[1])
                year = datetime.utcnow().year
                exp_date = dt_date(year, month, day)
                if exp_date < dt_date.today():
                    # Roll forward only if we're still well within the same year window
                    exp_date = dt_date(year + 1, month, day)
                return dt_date.today() > exp_date
        except (ValueError, IndexError):
            pass
        return False

    def _held_longer_than(self, alert: Alert, days: int) -> bool:
        """True if more than `days` calendar days have elapsed since the
        alert was sent."""
        return self._iso_older_than(alert.sent_at, days)

    @staticmethod
    def _iso_older_than(iso_str: str | None, days: int) -> bool:
        """True if `iso_str` is more than `days` calendar days in the past."""
        if not iso_str:
            return False
        try:
            then = datetime.fromisoformat(iso_str)
        except (ValueError, TypeError):
            return False
        return (datetime.utcnow() - then).total_seconds() > days * 86400

    def _set_hold_duration(self, outcome: Outcome, alert: Alert) -> None:
        if outcome.hold_duration_minutes is None and alert.sent_at:
            try:
                sent = datetime.fromisoformat(alert.sent_at)
                outcome.hold_duration_minutes = int(
                    (datetime.utcnow() - sent).total_seconds() / 60
                )
            except (ValueError, TypeError):
                pass
