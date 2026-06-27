"""Adaptive scoring thresholds — closes the self-improvement loop.

The weekly postmortem reads the rolling 7-day win rate and nudges the
shadow/live alert thresholds. Persisted to disk so a restart doesn't
forget what the system has learned.

Rules:
  win_rate < 40%  -> raise threshold by 5 (bot is being too generous)
  win_rate > 60%  -> lower threshold by 5 (bot is being too conservative)
  in between      -> no change
Bounded to a safe range so adversarial outcomes can't push it absurdly.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from src.db.database import get_session
from src.db.models import Alert, Outcome

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
STATE_FILE = DATA_DIR / "learned_state.json"

DEFAULT_STATE = {
    "min_score_to_alert_shadow": 62,
    "min_score_to_alert": 72,
    "last_adjusted_at": None,
    "win_rate_7d": None,
    "samples_7d": 0,
    "history": [],
}

# Adjustment policy
WIN_RATE_LOW = 0.40
WIN_RATE_HIGH = 0.60
ADJUST_STEP = 5
MIN_SAMPLES_REQUIRED = 5
SHADOW_BOUNDS = (35, 80)
LIVE_BOUNDS = (50, 90)


def load_state() -> dict:
    if not STATE_FILE.exists():
        return dict(DEFAULT_STATE)
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
        # Preserve ALL persisted keys (tuned_weights, hour_stats,
        # circuit_breaker, counterfactual_7d, ...), filling any missing
        # DEFAULT_STATE keys. Previously this dropped non-default keys, which
        # silently broke cross-run persistence of learned state.
        merged = dict(DEFAULT_STATE)
        merged.update(data)
        return merged
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to read learned_state.json; using defaults.")
        return dict(DEFAULT_STATE)


def save_state(state: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_threshold(mode: str, fallback: float) -> float:
    """Read the live shadow/live threshold. Falls back to config value
    if the state file is absent or missing the key."""
    state = load_state()
    key = "min_score_to_alert_shadow" if mode == "shadow" else "min_score_to_alert"
    val = state.get(key)
    return float(val) if isinstance(val, (int, float)) else float(fallback)


CF_MIN_SAMPLES = 10
CF_STRICT_WIN_RATE = 0.55   # filtered setups winning this often => gates too strict


def calibrate_from_counterfactuals(reason: str = "weekly") -> dict:
    """Nudge thresholds DOWN when setups we filtered out have been winning a
    lot over the last 7 days (a sign the gates are too strict and we're
    leaving money on the table). Conservative: only lowers, bounded, and
    requires a minimum sample size.
    """
    from src.db.models import CounterfactualOutcome

    cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()
    with get_session() as session:
        rows = (
            session.query(CounterfactualOutcome.pnl_pct)
            .filter(CounterfactualOutcome.resolved_at.isnot(None))
            .filter(CounterfactualOutcome.created_at >= cutoff)
            .filter(CounterfactualOutcome.pnl_pct.isnot(None))
            .all()
        )
    samples = len(rows)
    wins = sum(1 for (p,) in rows if (p or 0) > 0)
    win_rate = (wins / samples) if samples else None

    state = load_state()
    state["counterfactual_7d"] = {
        "samples": samples,
        "win_rate": round(win_rate, 3) if win_rate is not None else None,
    }

    if samples >= CF_MIN_SAMPLES and win_rate is not None and win_rate >= CF_STRICT_WIN_RATE:
        old_shadow = state["min_score_to_alert_shadow"]
        old_live = state["min_score_to_alert"]
        state["min_score_to_alert_shadow"] = max(
            SHADOW_BOUNDS[0], old_shadow - ADJUST_STEP
        )
        state["min_score_to_alert"] = max(LIVE_BOUNDS[0], old_live - ADJUST_STEP)
        logger.info(
            "Counterfactual calibration: filtered win rate %.0f%% over %d samples; "
            "lowering floors shadow %d->%d, live %d->%d.",
            win_rate * 100, samples, old_shadow, state["min_score_to_alert_shadow"],
            old_live, state["min_score_to_alert"],
        )
    else:
        logger.info(
            "Counterfactual calibration: %s samples, win_rate=%s; no change.",
            samples, state["counterfactual_7d"]["win_rate"],
        )
    save_state(state)
    return state["counterfactual_7d"]


def adjust_thresholds(reason: str = "weekly") -> dict:
    """Compute rolling 7-day win rate and nudge thresholds. Returns the
    updated state dict. Idempotent — safe to call multiple times."""
    cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()
    with get_session() as session:
        outcomes = (
            session.query(Outcome)
            .join(Alert)
            .filter(Alert.sent_at >= cutoff)
            .filter(Outcome.resolved_at.isnot(None))
            .all()
        )
    resolved = [o for o in outcomes if o.exit_reason in ("target", "stop", "expiry", "time_exit")]
    samples = len(resolved)
    wins = sum(1 for o in resolved if (o.pnl_pct or 0) > 0)
    win_rate = (wins / samples) if samples > 0 else None

    state = load_state()
    state["samples_7d"] = samples
    state["win_rate_7d"] = round(win_rate, 3) if win_rate is not None else None

    if samples < MIN_SAMPLES_REQUIRED or win_rate is None:
        logger.info(
            "Adaptive thresholds: only %d samples in last 7d (need >= %d). "
            "Holding thresholds steady.",
            samples, MIN_SAMPLES_REQUIRED,
        )
        save_state(state)
        return state

    delta = 0
    if win_rate < WIN_RATE_LOW:
        delta = ADJUST_STEP
        rationale = f"win_rate {win_rate:.0%} < {WIN_RATE_LOW:.0%}; raise"
    elif win_rate > WIN_RATE_HIGH:
        delta = -ADJUST_STEP
        rationale = f"win_rate {win_rate:.0%} > {WIN_RATE_HIGH:.0%}; lower"
    else:
        rationale = f"win_rate {win_rate:.0%} in dead band; hold"

    if delta != 0:
        old_shadow = state["min_score_to_alert_shadow"]
        old_live = state["min_score_to_alert"]
        state["min_score_to_alert_shadow"] = max(
            SHADOW_BOUNDS[0], min(SHADOW_BOUNDS[1], old_shadow + delta)
        )
        state["min_score_to_alert"] = max(
            LIVE_BOUNDS[0], min(LIVE_BOUNDS[1], old_live + delta)
        )
        logger.info(
            "Adaptive thresholds: %s. shadow %d -> %d, live %d -> %d",
            rationale, old_shadow, state["min_score_to_alert_shadow"],
            old_live, state["min_score_to_alert"],
        )
    else:
        logger.info("Adaptive thresholds: %s.", rationale)

    state["last_adjusted_at"] = datetime.utcnow().isoformat()
    history = state.get("history", []) or []
    history.append({
        "at": state["last_adjusted_at"],
        "trigger": reason,
        "samples": samples,
        "win_rate": state["win_rate_7d"],
        "delta": delta,
        "shadow_threshold": state["min_score_to_alert_shadow"],
        "live_threshold": state["min_score_to_alert"],
        "rationale": rationale,
    })
    state["history"] = history[-50:]  # cap history
    save_state(state)
    return state
