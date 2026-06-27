"""Hour-of-day performance tracking.

Builds a per-(scan-hour) profile of win rate + average P&L from resolved
trades, then exposes a multiplicative score modifier so the scoring
engine can favour scan times that historically produce winners and
penalise the bad ones.

Bucketing is by scan hour in America/New_York (08, 09, 10, ...). We
look up each Alert.sent_at, convert to ET, and group.

Sample-size guards: a bucket needs >= MIN_SAMPLES_PER_HOUR resolved
trades before its modifier deviates from 1.0; otherwise it returns 1.0.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from src.db.database import get_session
from src.db.models import Alert, Outcome
from src.learning.adaptive_thresholds import load_state, save_state

logger = logging.getLogger(__name__)

# Stats policy
MIN_SAMPLES_PER_HOUR = 8     # below this, use neutral modifier
MAX_MODIFIER = 1.10          # cap bonus
MIN_MODIFIER = 0.85          # cap penalty
EXPECTED_WIN_RATE = 0.50     # baseline; deviations above/below scale modifier

ET = ZoneInfo("America/New_York")


def compute_hour_stats() -> dict[int, dict[str, Any]]:
    """Returns {hour: {samples, wins, win_rate, avg_pnl_pct}}.

    Uses sent_at converted to America/New_York. Only resolved alerts
    with a clear target/stop/expiry exit count.
    """
    with get_session() as session:
        rows = (
            session.query(Alert.sent_at, Outcome.pnl_pct, Outcome.exit_reason)
            .join(Outcome, Outcome.alert_id == Alert.id)
            .filter(Outcome.resolved_at.isnot(None))
            .filter(Outcome.exit_reason.in_(("target", "stop", "expiry", "time_exit")))
            .filter(Outcome.pnl_pct.isnot(None))
            .all()
        )

    buckets: dict[int, dict[str, Any]] = {}
    for sent_at, pnl_pct, exit_reason in rows:
        if not sent_at:
            continue
        try:
            dt = datetime.fromisoformat(sent_at)
        except (ValueError, TypeError):
            continue
        # Treat naive as UTC.
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        hour = dt.astimezone(ET).hour
        b = buckets.setdefault(hour, {
            "samples": 0, "wins": 0, "total_pnl": 0.0,
        })
        b["samples"] += 1
        b["total_pnl"] += float(pnl_pct or 0)
        if (pnl_pct or 0) > 0:
            b["wins"] += 1

    out: dict[int, dict[str, Any]] = {}
    for h, b in buckets.items():
        n = b["samples"]
        out[h] = {
            "samples": n,
            "wins": b["wins"],
            "win_rate": round(b["wins"] / n, 3) if n else 0.0,
            "avg_pnl_pct": round(b["total_pnl"] / n, 2) if n else 0.0,
        }
    return out


def update_hour_stats(reason: str = "weekly") -> dict:
    """Recompute hour stats and persist to learned_state.json."""
    stats = compute_hour_stats()
    state = load_state()
    state["hour_stats"] = {str(h): v for h, v in stats.items()}
    state["hour_stats_updated_at"] = datetime.utcnow().isoformat()
    state.setdefault("hour_stats_history", []).append({
        "at": state["hour_stats_updated_at"],
        "trigger": reason,
        "stats": dict(state["hour_stats"]),
    })
    state["hour_stats_history"] = state["hour_stats_history"][-12:]
    save_state(state)
    if stats:
        best = max(stats.items(), key=lambda kv: kv[1]["win_rate"])
        worst = min(stats.items(), key=lambda kv: kv[1]["win_rate"])
        logger.info(
            "Hour-of-day stats updated: best %02d:00 ET (%d samples, %.0f%% win rate); "
            "worst %02d:00 ET (%d samples, %.0f%% win rate)",
            best[0], best[1]["samples"], best[1]["win_rate"] * 100,
            worst[0], worst[1]["samples"], worst[1]["win_rate"] * 100,
        )
    return state


def hour_score_modifier(hour: int | None = None) -> float:
    """Multiplicative modifier in [MIN_MODIFIER, MAX_MODIFIER] for the
    given scan hour (America/New_York). Defaults to current ET hour."""
    if hour is None:
        hour = datetime.now(ET).hour

    state = load_state()
    raw = (state.get("hour_stats") or {}).get(str(hour))
    if not raw:
        return 1.0
    samples = int(raw.get("samples") or 0)
    if samples < MIN_SAMPLES_PER_HOUR:
        return 1.0

    win_rate = float(raw.get("win_rate") or EXPECTED_WIN_RATE)
    # +0.20 win-rate -> max bonus, -0.20 -> max penalty (linear).
    delta = win_rate - EXPECTED_WIN_RATE
    if delta >= 0:
        modifier = 1.0 + min(MAX_MODIFIER - 1.0, delta * (MAX_MODIFIER - 1.0) / 0.20)
    else:
        modifier = 1.0 + max(MIN_MODIFIER - 1.0, delta * (1.0 - MIN_MODIFIER) / 0.20)
    return round(modifier, 3)
