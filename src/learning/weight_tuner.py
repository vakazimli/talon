"""Adaptive scoring-weight tuner.

Once we have enough resolved trades (default >= 100), search the weight
space for combinations that produce a higher correlation between final
score and realized P&L. Persists best weights to learned_state.json so
ScoringEngine can read them at scoring time.

Method: random search with a sum-to-one constraint. We score each
candidate weighting by Spearman rank correlation between the *would-be
score* (recomputed from each setup's stored breakdown) and *actual P&L*.
Spearman is preferred over Pearson because it tolerates the heavy-tailed
P&L distribution of options trades.

Bounds & guards:
  * Each weight in [0.05, 0.55].
  * Sum to 1.0 (renormalized after sampling).
  * Refuse to apply weights with rank correlation < 0.10 (no real signal).
  * Step size capped at +/- 0.10 from current weights so it doesn't
    swing wildly between weekly runs.
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from typing import Any

from src.db.database import get_session
from src.db.models import Alert, Outcome, Setup
from src.learning.adaptive_thresholds import STATE_FILE, load_state, save_state

logger = logging.getLogger(__name__)

# Tuning policy
MIN_RESOLVED_SAMPLES = 100
N_TRIALS = 400
MIN_CORRELATION_TO_APPLY = 0.10
WEIGHT_BOUNDS = (0.05, 0.55)
MAX_DELTA_PER_RUN = 0.10  # smooth: no key shifts > 0.10 from current

# Default weight set (also matches config defaults — fallback if no prior)
DEFAULT_WEIGHTS = {
    "technical_setup": 0.40,
    "source_reliability": 0.15,
    "risk_reward": 0.30,
    "news_catalyst": 0.15,
}
WEIGHT_KEYS = tuple(DEFAULT_WEIGHTS.keys())


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation. Returns 0 on degenerate input."""
    n = len(xs)
    if n < 5 or len(ys) != n:
        return 0.0

    def _rank(values: list[float]) -> list[float]:
        # Average rank for ties
        sorted_idx = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[sorted_idx[j + 1]] == values[sorted_idx[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg
            i = j + 1
        return ranks

    rx = _rank(xs)
    ry = _rank(ys)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    num = sum((rx[i] - mean_x) * (ry[i] - mean_y) for i in range(n))
    dx = sum((rx[i] - mean_x) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - mean_y) ** 2 for i in range(n)) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def _load_samples() -> list[dict]:
    """Load (breakdown, pnl_pct) pairs for resolved trades.

    Only setups that produced an alert and had the alert resolved are
    eligible. Samples include the per-component scores from the time of
    the original alert so we can re-score with new weights.
    """
    samples: list[dict] = []
    with get_session() as session:
        rows = (
            session.query(Setup, Alert, Outcome)
            .join(Alert, Alert.setup_id == Setup.id)
            .join(Outcome, Outcome.alert_id == Alert.id)
            .filter(Outcome.resolved_at.isnot(None))
            .filter(Outcome.exit_reason.in_(("target", "stop", "expiry", "time_exit")))
            .filter(Outcome.pnl_pct.isnot(None))
            .all()
        )
    for setup, _alert, outcome in rows:
        try:
            breakdown = json.loads(setup.score_breakdown or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if not all(k in breakdown for k in WEIGHT_KEYS):
            continue
        samples.append({
            "breakdown": {k: float(breakdown[k]) for k in WEIGHT_KEYS},
            "pnl_pct": float(outcome.pnl_pct or 0),
        })
    return samples


def _sample_weights(rng: random.Random, anchor: dict) -> dict:
    """Random weights near `anchor`, respecting bounds & MAX_DELTA_PER_RUN
    on each key, then renormalized to sum to 1.0."""
    candidate = {}
    for k in WEIGHT_KEYS:
        a = anchor.get(k, DEFAULT_WEIGHTS[k])
        delta = rng.uniform(-MAX_DELTA_PER_RUN, MAX_DELTA_PER_RUN)
        v = max(WEIGHT_BOUNDS[0], min(WEIGHT_BOUNDS[1], a + delta))
        candidate[k] = v
    total = sum(candidate.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {k: v / total for k, v in candidate.items()}


def _score_weight_set(weights: dict, samples: list[dict]) -> tuple[float, list[float], list[float]]:
    scores = [
        sum(s["breakdown"].get(k, 0) * weights.get(k, 0) for k in WEIGHT_KEYS)
        for s in samples
    ]
    pnls = [s["pnl_pct"] for s in samples]
    return _spearman(scores, pnls), scores, pnls


def tune_weights(reason: str = "weekly", seed: int | None = None) -> dict:
    """Search the weight space and persist the best find. Returns the
    updated learned-state dict (always — even when no change is made)."""
    samples = _load_samples()
    state = load_state()
    state.setdefault("tuned_weights", {})
    state.setdefault("tuned_weights_history", [])

    if len(samples) < MIN_RESOLVED_SAMPLES:
        logger.info(
            "Weight tuner: only %d resolved samples (need >= %d). Skipping.",
            len(samples), MIN_RESOLVED_SAMPLES,
        )
        state["tuned_weights"] = state.get("tuned_weights") or {
            "weights": dict(DEFAULT_WEIGHTS),
            "samples": len(samples),
            "correlation": 0.0,
            "active": False,
        }
        save_state(state)
        return state

    rng = random.Random(seed)
    current = (state.get("tuned_weights") or {}).get("weights") or dict(DEFAULT_WEIGHTS)

    best_weights = dict(current)
    best_corr, _, _ = _score_weight_set(current, samples)

    for _ in range(N_TRIALS):
        cand = _sample_weights(rng, current)
        corr, _, _ = _score_weight_set(cand, samples)
        if corr > best_corr:
            best_corr = corr
            best_weights = cand

    active = best_corr >= MIN_CORRELATION_TO_APPLY
    if active:
        logger.info(
            "Weight tuner: rank corr %.3f over %d samples. Applying weights: %s",
            best_corr, len(samples),
            {k: round(v, 3) for k, v in best_weights.items()},
        )
    else:
        logger.info(
            "Weight tuner: best rank corr %.3f below threshold %.2f. Holding existing weights.",
            best_corr, MIN_CORRELATION_TO_APPLY,
        )

    state["tuned_weights"] = {
        "weights": best_weights,
        "samples": len(samples),
        "correlation": round(best_corr, 4),
        "active": active,
        "updated_at": datetime.utcnow().isoformat(),
    }
    history = state.get("tuned_weights_history") or []
    history.append({
        "at": state["tuned_weights"]["updated_at"],
        "trigger": reason,
        "samples": len(samples),
        "correlation": state["tuned_weights"]["correlation"],
        "weights": best_weights,
        "active": active,
    })
    state["tuned_weights_history"] = history[-30:]
    save_state(state)
    return state


def get_active_weights(fallback: dict) -> dict:
    """Return tuned weights if active and present, otherwise `fallback`.
    `fallback` is what the ScoringEngine has from config."""
    state = load_state()
    block = state.get("tuned_weights") or {}
    if not block.get("active"):
        return dict(fallback)
    weights = block.get("weights")
    if not isinstance(weights, dict):
        return dict(fallback)
    # Defensive: only return weights for known keys, fallback for any missing
    out = dict(fallback)
    for k in WEIGHT_KEYS:
        if k in weights:
            out[k] = float(weights[k])
    # Renormalize
    total = sum(out.values())
    if total > 0:
        return {k: v / total for k, v in out.items()}
    return dict(fallback)
