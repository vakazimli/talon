"""Self-contained ML win-probability model.

A small, interpretable logistic-regression classifier trained on Talon's own
accumulated history (resolved alert outcomes + counterfactuals) to estimate
the probability a scored setup becomes a winner. Implemented with numpy +
scipy.optimize only (no scikit-learn / heavy deps; local-first).

It is a NO-OP until trained on enough samples, at which point it can apply a
small, bounded multiplicative nudge to the heuristic score. Features come
from the persisted score breakdown (always available historically) plus the
signal count, so no schema change is needed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.db.database import get_session

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
MODEL_FILE = DATA_DIR / "ml_model.json"

# Persisted score-breakdown components (0-100) used as features.
FEATURE_KEYS = ["technical_setup", "source_reliability", "risk_reward", "news_catalyst"]

MIN_SAMPLES = 150
MIN_TRAIN_ACC = 0.55     # must beat a coin flip to activate
L2 = 1.0
# Bounded modifier applied to the heuristic score when active.
MIN_MOD = 0.90
MAX_MOD = 1.10


def features_from(breakdown: dict, signal_count: int) -> list[float]:
    """Feature vector from a setup's score breakdown + number of signals.
    Components are scaled to ~0-1; signal_count capped at 5."""
    vec = [float(breakdown.get(k, 50.0)) / 100.0 for k in FEATURE_KEYS]
    vec.append(min(int(signal_count or 0), 5) / 5.0)
    return vec


def _load_samples() -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) from resolved alert outcomes and counterfactuals."""
    import json as _json

    from src.db.models import Alert, CounterfactualOutcome, Outcome, Setup

    rows: list[tuple[dict, list, int]] = []
    with get_session() as session:
        real = (
            session.query(Setup, Outcome)
            .join(Alert, Alert.setup_id == Setup.id)
            .join(Outcome, Outcome.alert_id == Alert.id)
            .filter(Outcome.resolved_at.isnot(None))
            .filter(Outcome.pnl_pct.isnot(None))
            .all()
        )
        for setup, outcome in real:
            rows.append((setup.score_breakdown, setup.raw_signals, 1 if (outcome.pnl_pct or 0) > 0 else 0))

        cfs = (
            session.query(Setup, CounterfactualOutcome)
            .join(CounterfactualOutcome, CounterfactualOutcome.setup_id == Setup.id)
            .filter(CounterfactualOutcome.resolved_at.isnot(None))
            .filter(CounterfactualOutcome.pnl_pct.isnot(None))
            .all()
        )
        for setup, cf in cfs:
            rows.append((setup.score_breakdown, setup.raw_signals, 1 if (cf.pnl_pct or 0) > 0 else 0))

    X, y = [], []
    for breakdown_json, signals_json, label in rows:
        try:
            breakdown = _json.loads(breakdown_json or "{}")
            signals = _json.loads(signals_json or "[]")
        except (json.JSONDecodeError, TypeError):
            continue
        if not all(k in breakdown for k in FEATURE_KEYS):
            continue
        X.append(features_from(breakdown, len(signals)))
        y.append(label)
    return np.array(X, dtype=float), np.array(y, dtype=float)


def _fit_logreg(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features and fit L2-regularized logistic regression.
    Returns (theta_with_bias, mean, std)."""
    from scipy.optimize import minimize

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    n, d = Xs.shape
    Xb = np.hstack([np.ones((n, 1)), Xs])  # bias column

    def nll(theta):
        z = Xb @ theta
        # mean logistic loss + L2 on non-bias weights
        loss = np.mean(np.logaddexp(0.0, z) - y * z)
        return loss + L2 * np.sum(theta[1:] ** 2) / n

    theta0 = np.zeros(d + 1)
    res = minimize(nll, theta0, method="L-BFGS-B")
    return res.x, mean, std


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + np.exp(-z))
    e = np.exp(z)
    return e / (1.0 + e)


def train(min_samples: int = MIN_SAMPLES) -> dict:
    """Train and persist the model. Returns a status dict (always)."""
    X, y = _load_samples()
    n = len(y)
    status = {"samples": n, "active": False, "train_acc": None}
    if n < min_samples or len(np.unique(y)) < 2:
        logger.info("ML model: %d samples (need >= %d, both classes). Not training.",
                    n, min_samples)
        _save({**status, "reason": "insufficient_data"})
        return status

    theta, mean, std = _fit_logreg(X, y)
    Xs = (X - mean) / std
    z = np.hstack([np.ones((len(Xs), 1)), Xs]) @ theta
    probs = np.array([_sigmoid(v) for v in z])
    acc = float(((probs > 0.5).astype(float) == y).mean())
    active = acc >= MIN_TRAIN_ACC

    model = {
        "active": active,
        "samples": n,
        "train_acc": round(acc, 3),
        "feature_keys": FEATURE_KEYS,
        "theta": theta.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    _save(model)
    logger.info("ML model trained: n=%d acc=%.3f active=%s", n, acc, active)
    return model


def _save(model: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_FILE.write_text(json.dumps(model, indent=2))


def load_model() -> dict | None:
    if not MODEL_FILE.exists():
        return None
    try:
        return json.loads(MODEL_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def predict_win_prob(feature_vec: list[float]) -> float | None:
    """Win probability for a feature vector, or None if no active model."""
    model = load_model()
    if not model or not model.get("active"):
        return None
    try:
        theta = np.array(model["theta"], dtype=float)
        mean = np.array(model["mean"], dtype=float)
        std = np.array(model["std"], dtype=float)
        x = (np.array(feature_vec, dtype=float) - mean) / std
        z = float(theta[0] + x @ theta[1:])
        return float(_sigmoid(z))
    except Exception:
        logger.exception("ML predict failed.")
        return None


def score_modifier(breakdown: dict, signal_count: int) -> float:
    """Bounded multiplicative modifier in [MIN_MOD, MAX_MOD] from win prob.
    Returns 1.0 (no-op) when the model is inactive/untrained."""
    prob = predict_win_prob(features_from(breakdown, signal_count))
    if prob is None:
        return 1.0
    # Map prob in [0,1] around 0.5 to [MIN_MOD, MAX_MOD] linearly.
    span = (MAX_MOD - MIN_MOD)
    return round(MIN_MOD + span * max(0.0, min(1.0, prob)), 3)


def status() -> dict:
    model = load_model()
    if not model:
        return {"active": False, "samples": 0, "trained": False}
    return {
        "active": bool(model.get("active")),
        "samples": model.get("samples", 0),
        "train_acc": model.get("train_acc"),
        "trained": True,
    }
