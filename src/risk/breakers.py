"""Risk circuit breakers.

Talon never executes trades, so these breakers gate ALERTING (and
counterfactual tracking), never money. They pause new alerts when recent
results are poor or open exposure is too high, and auto-reset at the start
of each new trading day (America/New_York).

Breakers:
  * exposure       — too many open (unresolved) outcomes right now.
  * daily_loss     — cumulative resolved P&L today <= daily_loss_halt_pct.
  * loss_streak    — last N resolved trades are all losers.
  * weekly_drawdown— cumulative resolved P&L over 7d <= weekly_drawdown_halt_pct.

The exposure breaker is evaluated live every cycle (not persisted). The
loss/streak/drawdown breakers latch for the rest of the ET day and are
persisted in data/learned_state.json so a restart remembers them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.db.database import get_session
from src.db.models import Outcome
from src.learning.adaptive_thresholds import load_state, save_state

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

DEFAULTS = {
    "max_open_positions": 6,
    "daily_loss_halt_pct": -40.0,      # sum of today's resolved pnl_pct
    "consecutive_loss_halt": 4,        # N losers in a row
    "weekly_drawdown_halt_pct": -80.0,  # sum of 7d resolved pnl_pct
}


@dataclass
class BreakerStatus:
    halted: bool
    reason: str
    open_positions: int
    daily_pnl_pct: float
    consecutive_losses: int
    weekly_pnl_pct: float


def _cfg(config: dict) -> dict:
    risk = (config or {}).get("risk", {}) if isinstance(config, dict) else {}
    out = dict(DEFAULTS)
    for k in DEFAULTS:
        if k in risk and risk[k] is not None:
            out[k] = risk[k]
    return out


def _utc_cutoff(days_back: int) -> str:
    """UTC ISO timestamp for ET-midnight `days_back` days ago."""
    now_et = datetime.now(ET)
    start_et = (now_et - timedelta(days=days_back)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return start_et.astimezone(ZoneInfo("UTC")).replace(tzinfo=None).isoformat()


def evaluate(config: dict) -> tuple[int, float, int, float]:
    """Return (open_positions, daily_pnl_pct, consecutive_losses, weekly_pnl_pct)."""
    today_cutoff = _utc_cutoff(0)
    week_cutoff = _utc_cutoff(7)
    with get_session() as session:
        open_positions = (
            session.query(Outcome)
            .filter(Outcome.resolved_at.is_(None))
            .filter(Outcome.entry_filled.is_(True))
            .count()
        )
        daily_pnl = (
            session.query(Outcome.pnl_pct)
            .filter(Outcome.resolved_at.isnot(None))
            .filter(Outcome.resolved_at >= today_cutoff)
            .filter(Outcome.pnl_pct.isnot(None))
            .all()
        )
        weekly_pnl = (
            session.query(Outcome.pnl_pct)
            .filter(Outcome.resolved_at.isnot(None))
            .filter(Outcome.resolved_at >= week_cutoff)
            .filter(Outcome.pnl_pct.isnot(None))
            .all()
        )
        recent = (
            session.query(Outcome.pnl_pct)
            .filter(Outcome.resolved_at.isnot(None))
            .filter(Outcome.pnl_pct.isnot(None))
            .order_by(Outcome.resolved_at.desc())
            .limit(20)
            .all()
        )

    daily_sum = round(sum(float(r[0]) for r in daily_pnl), 2)
    weekly_sum = round(sum(float(r[0]) for r in weekly_pnl), 2)
    streak = 0
    for (pnl,) in recent:
        if pnl is not None and float(pnl) < 0:
            streak += 1
        else:
            break
    return open_positions, daily_sum, streak, weekly_sum


def check_and_update(config: dict) -> BreakerStatus:
    """Evaluate breakers, latch day-level trips into learned_state, and
    return the current halt status. Safe to call every cycle."""
    cfg = _cfg(config)
    open_positions, daily_pnl, streak, weekly_pnl = evaluate(config)

    state = load_state()
    cb = state.get("circuit_breaker") or {}
    today = datetime.now(ET).date().isoformat()

    # Auto-reset a stale latch from a previous day.
    if cb.get("tripped_for_date") and cb.get("tripped_for_date") != today:
        cb = {}

    # Latch a persistent (day-long) trip on poor results.
    persistent_reason = None
    if daily_pnl <= cfg["daily_loss_halt_pct"]:
        persistent_reason = (
            f"daily loss {daily_pnl:.0f}% <= {cfg['daily_loss_halt_pct']:.0f}%"
        )
    elif streak >= cfg["consecutive_loss_halt"]:
        persistent_reason = f"{streak} consecutive losses"
    elif weekly_pnl <= cfg["weekly_drawdown_halt_pct"]:
        persistent_reason = (
            f"weekly drawdown {weekly_pnl:.0f}% <= {cfg['weekly_drawdown_halt_pct']:.0f}%"
        )

    if persistent_reason and cb.get("tripped_for_date") != today:
        cb = {
            "tripped_for_date": today,
            "reason": persistent_reason,
            "at": datetime.utcnow().isoformat(),
        }
        logger.warning("Circuit breaker tripped for %s: %s", today, persistent_reason)

    state["circuit_breaker"] = cb
    save_state(state)

    latched = cb.get("tripped_for_date") == today
    exposure_halt = open_positions >= cfg["max_open_positions"]
    halted = latched or exposure_halt
    if latched:
        reason = cb.get("reason", "tripped")
    elif exposure_halt:
        reason = f"exposure cap: {open_positions} open >= {cfg['max_open_positions']}"
    else:
        reason = ""

    return BreakerStatus(
        halted=halted,
        reason=reason,
        open_positions=open_positions,
        daily_pnl_pct=daily_pnl,
        consecutive_losses=streak,
        weekly_pnl_pct=weekly_pnl,
    )


def manual_reset() -> None:
    """Clear a latched day-trip (exposure is recomputed live)."""
    state = load_state()
    state["circuit_breaker"] = {}
    save_state(state)
    logger.info("Circuit breaker manually reset.")


def status_text(config: dict) -> str:
    st = check_and_update(config)
    cfg = _cfg(config)
    head = "HALTED" if st.halted else "OK"
    lines = [
        f"Risk breakers: {head}",
    ]
    if st.reason:
        lines.append(f"Reason: {st.reason}")
    lines.append(
        f"Open positions: {st.open_positions}/{cfg['max_open_positions']}"
    )
    lines.append(
        f"Today P&L: {st.daily_pnl_pct:+.0f}% (halt <= {cfg['daily_loss_halt_pct']:.0f}%)"
    )
    lines.append(
        f"Loss streak: {st.consecutive_losses} (halt >= {cfg['consecutive_loss_halt']})"
    )
    lines.append(
        f"7d P&L: {st.weekly_pnl_pct:+.0f}% (halt <= {cfg['weekly_drawdown_halt_pct']:.0f}%)"
    )
    return "\n".join(lines)
