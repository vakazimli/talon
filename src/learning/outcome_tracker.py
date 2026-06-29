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
        """Resolve unresolved alert outcomes.

        Pattern: snapshot open rows -> fetch quotes with NO DB session open
        -> apply updates in one short write session -> run source-reliability
        updates afterwards. This avoids holding a write transaction across
        network calls (SQLite lock contention) and avoids nested sessions.
        """
        with get_session() as session:
            rows = (
                session.query(Outcome, Alert)
                .join(Alert, Alert.id == Outcome.alert_id)
                .filter(Outcome.resolved_at.is_(None))
                .filter(Outcome.entry_filled.is_(True))
                .all()
            )
            snaps = [
                {
                    "oid": o.id, "alert_id": a.id,
                    "entry": o.actual_entry_price or 0, "contract": a.contract,
                    "target": a.target_price or 0, "stop": a.stop_price or 0,
                    "hold_iso": a.sent_at, "max_fav": o.max_favorable,
                    "max_adv": o.max_adverse,
                }
                for (o, a) in rows
            ]

        # Network + compute, no session held.
        computed = []
        for s in snaps:
            if s["entry"] <= 0:
                continue
            try:
                fields, resolved = self._evaluate(s)
            except Exception:
                logger.exception("Failed to evaluate outcome #%s", s["oid"])
                continue
            if not fields:
                continue
            if "resolved_at" in fields:
                fields["hold_duration_minutes"] = self._hold_minutes(s["hold_iso"])
            if "actual_exit_price" in fields:
                fields["pnl_dollars"] = fields["actual_exit_price"] - s["entry"]
            computed.append((s["oid"], s["alert_id"], fields, resolved))

        # Apply in one short write session.
        resolved_alert_ids = []
        with get_session() as session:
            for oid, alert_id, fields, resolved in computed:
                o = session.query(Outcome).filter(Outcome.id == oid).first()
                if not o:
                    continue
                for k, v in fields.items():
                    setattr(o, k, v)
                if resolved:
                    resolved_alert_ids.append(alert_id)
            session.commit()

        # Source-reliability updates (each opens its own short session).
        for alert_id in resolved_alert_ids:
            try:
                with get_session() as session:
                    alert = session.query(Alert).filter(Alert.id == alert_id).first()
                    outcome = (
                        session.query(Outcome)
                        .filter(Outcome.alert_id == alert_id).first()
                    )
                if alert is not None and outcome is not None:
                    self.source_evaluator.update_from_outcome(alert, outcome)
            except Exception:
                logger.exception("Source update failed for alert #%s", alert_id)

    async def check_all_open_counterfactuals(self) -> None:
        """Resolve shadow (counterfactual) outcomes — same snapshot/network/
        write discipline as real outcomes."""
        from src.db.models import CounterfactualOutcome
        with get_session() as session:
            rows = (
                session.query(CounterfactualOutcome)
                .filter(CounterfactualOutcome.resolved_at.is_(None))
                .all()
            )
            snaps = [
                {
                    "oid": cf.id, "entry": cf.entry_price or 0, "contract": cf.contract,
                    "target": cf.target_price or 0, "stop": cf.stop_price or 0,
                    "hold_iso": cf.created_at, "max_fav": cf.max_favorable,
                    "max_adv": cf.max_adverse,
                }
                for cf in rows
            ]

        computed = []
        for s in snaps:
            if s["entry"] <= 0:
                continue
            try:
                fields, _resolved = self._evaluate(s)
            except Exception:
                logger.exception("Failed to evaluate counterfactual #%s", s["oid"])
                continue
            if fields:
                computed.append((s["oid"], fields))

        with get_session() as session:
            for oid, fields in computed:
                cf = session.query(CounterfactualOutcome).filter(
                    CounterfactualOutcome.id == oid
                ).first()
                if not cf:
                    continue
                for k, v in fields.items():
                    setattr(cf, k, v)
            session.commit()

    def _evaluate(self, snap: dict) -> tuple[dict, bool]:
        """Given a position snapshot, fetch the live option price and return
        (fields_to_update, resolved). Shared by real + counterfactual outcomes;
        the returned column names exist on both models. Does network I/O but
        holds NO DB session."""
        entry = snap["entry"]
        contract = snap["contract"]
        price = self._get_option_price(contract)
        expired = self._contract_expired(contract)

        if price is None:
            if expired:
                return {
                    "exit_reason": "expiry_unknown",
                    "resolved_at": datetime.utcnow().isoformat(),
                }, False
            return {}, False

        prev_fav = snap.get("max_fav")
        prev_adv = snap.get("max_adv")
        fields: dict = {
            "max_favorable": price if prev_fav is None else max(prev_fav, price),
            "max_adverse": price if prev_adv is None else min(prev_adv, price),
        }

        target = snap["target"]
        stop = snap["stop"]
        resolved = False
        reason = None
        exit_price = price
        if target > 0 and price >= target:
            resolved, reason, exit_price = True, "target", target
        elif stop > 0 and price <= stop:
            resolved, reason, exit_price = True, "stop", stop
        elif expired:
            resolved, reason = True, "expiry"
        elif self._iso_older_than(snap.get("hold_iso"), MAX_HOLD_DAYS):
            resolved, reason, exit_price = True, "time_exit", price

        if resolved:
            fields["actual_exit_price"] = exit_price
            fields["exit_reason"] = reason
            fields["pnl_pct"] = ((exit_price / entry) - 1) * 100 if entry > 0 else 0
            fields["resolved_at"] = datetime.utcnow().isoformat()
        return fields, resolved

    @staticmethod
    def _hold_minutes(iso_str: str | None) -> int | None:
        if not iso_str:
            return None
        try:
            sent = datetime.fromisoformat(iso_str)
        except (ValueError, TypeError):
            return None
        return int((datetime.utcnow() - sent).total_seconds() / 60)

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
