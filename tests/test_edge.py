import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --- Phase A: BSM pricing + backtest aggregation ---

def test_bs_price_basic_and_monotonic():
    from src.scanner.greeks import bs_price
    atm = bs_price(100, 100, 9, 0.30, "call")
    assert atm > 0
    itm = bs_price(110, 100, 9, 0.30, "call")
    otm = bs_price(90, 100, 9, 0.30, "call")
    assert itm > atm > otm  # call value rises with underlying
    # zero IV -> intrinsic
    assert bs_price(110, 100, 9, 0.0, "call") == 10.0
    assert bs_price(90, 100, 9, 0.0, "put") == 10.0


def test_backtest_aggregate():
    from src.backtest.engine import BacktestEngine
    eng = BacktestEngine({})
    trades = [
        {"ticker": "X", "direction": "bullish", "signals": ["volume_spike"], "score": 75, "exit_reason": "target", "pnl_pct": 30.0},
        {"ticker": "X", "direction": "bullish", "signals": ["volume_spike"], "score": 72, "exit_reason": "stop", "pnl_pct": -30.0},
        {"ticker": "Y", "direction": "bearish", "signals": ["breakdown_below_support"], "score": 40, "exit_reason": "time_exit", "pnl_pct": 5.0},
    ]
    res = eng._aggregate(trades)
    assert res.samples == 3
    assert res.wins == 2
    assert abs(res.win_rate - round(2 / 3, 3)) < 1e-6
    assert "volume_spike" in res.by_signal


# --- Phase B: critic verdict parsing + fail-open ---

def _fake_scored():
    return SimpleNamespace(
        ticker="SPY", direction="bullish", setup_type="breakout", score=75,
        raw_signals=["breakout_above_resistance"], technicals={"rsi": 60},
    )


def _fake_card():
    return SimpleNamespace(
        contract="SPY 530C 7/3", entry_low=2.0, entry_high=2.2,
        target=2.7, stop=1.5, rationale="momentum", confidence=4,
    )


class _FakeOrch:
    def __init__(self, resp, raise_it=False):
        self.resp = resp
        self.raise_it = raise_it

    def call_model_json(self, *a, **k):
        if self.raise_it:
            raise RuntimeError("boom")
        return self.resp


def test_critic_verdicts():
    from src.analysis.critic import RiskCritic
    cfg = {"critic": {"enabled": True}}
    veto = RiskCritic(_FakeOrch({"verdict": "VETO", "key_risk": "trend conflict"}), cfg)
    assert veto.critique(_fake_scored(), _fake_card()).verdict == "VETO"
    caution = RiskCritic(_FakeOrch({"verdict": "CAUTION"}), cfg)
    assert caution.critique(_fake_scored(), _fake_card()).verdict == "CAUTION"
    # invalid verdict -> PROCEED
    weird = RiskCritic(_FakeOrch({"verdict": "lol"}), cfg)
    assert weird.critique(_fake_scored(), _fake_card()).verdict == "PROCEED"
    # None / exception -> fail open (PROCEED)
    assert RiskCritic(_FakeOrch(None), cfg).critique(_fake_scored(), _fake_card()).verdict == "PROCEED"
    assert RiskCritic(_FakeOrch(None, raise_it=True), cfg).critique(_fake_scored(), _fake_card()).verdict == "PROCEED"


# --- Phase C: volatility-scaled exits ---

def _synth_cfg():
    return {"synthesis": {
        "target_pct": 0.30, "stop_pct": 0.30, "reference_iv": 0.30,
        "target_pct_min": 0.20, "target_pct_max": 0.45,
        "stop_pct_min": 0.20, "stop_pct_max": 0.40,
    }}


def test_scaled_exits_bounds():
    from src.analysis.synthesizer import Synthesizer
    s = Synthesizer(None, _synth_cfg())
    high_iv = SimpleNamespace(implied_volatility=0.60)  # ratio 2.0 -> clamp 1.8
    t, st = s._scaled_exit_pcts(high_iv, {})
    assert t == 0.45 and st == 0.40  # hit max bounds
    low_iv = SimpleNamespace(implied_volatility=0.15)   # ratio 0.5 -> clamp 0.6
    t2, st2 = s._scaled_exit_pcts(low_iv, {})
    assert t2 == 0.20 and st2 == 0.20  # hit min bounds
    # no IV, no atr -> base
    none = SimpleNamespace(implied_volatility=0)
    assert s._scaled_exit_pcts(none, {}) == (0.30, 0.30)


def test_derive_prices_uses_scaled():
    from src.analysis.synthesizer import Synthesizer
    s = Synthesizer(None, _synth_cfg())
    cand = SimpleNamespace(bid=2.0, ask=2.2, mid=2.1, implied_volatility=0.30)
    prices = s._derive_prices(cand, {})
    assert prices["entry_low"] == 2.0 and prices["entry_high"] == 2.2
    assert prices["target"] == round(2.1 * 1.30, 2)
    assert prices["stop"] == round(2.1 * 0.70, 2)


# --- Phase D: ML model ---

def test_ml_features_length():
    from src.learning.ml_model import FEATURE_KEYS, features_from
    vec = features_from({k: 50 for k in FEATURE_KEYS}, 3)
    assert len(vec) == len(FEATURE_KEYS) + 1


def test_ml_train_predict_and_modifier(tmp_path, monkeypatch):
    from src.learning import ml_model
    monkeypatch.setattr(ml_model, "MODEL_FILE", tmp_path / "m.json")
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0.3, 0.05, (80, 5)), rng.normal(0.7, 0.05, (80, 5))])
    y = np.array([0] * 80 + [1] * 80, dtype=float)
    monkeypatch.setattr(ml_model, "_load_samples", lambda: (X, y))

    res = ml_model.train(min_samples=50)
    assert res["active"] is True
    assert res["train_acc"] >= 0.55

    p = ml_model.predict_win_prob([0.7, 0.7, 0.7, 0.7, 0.7])
    assert p is not None and p > 0.5
    mod = ml_model.score_modifier(
        {"technical_setup": 70, "source_reliability": 70, "risk_reward": 70, "news_catalyst": 70}, 3
    )
    assert ml_model.MIN_MOD <= mod <= ml_model.MAX_MOD


def test_ml_modifier_noop_when_untrained(tmp_path, monkeypatch):
    from src.learning import ml_model
    monkeypatch.setattr(ml_model, "MODEL_FILE", tmp_path / "absent.json")
    mod = ml_model.score_modifier(
        {"technical_setup": 50, "source_reliability": 50, "risk_reward": 50, "news_catalyst": 50}, 2
    )
    assert mod == 1.0


if __name__ == "__main__":
    test_bs_price_basic_and_monotonic()
    test_backtest_aggregate()
    test_critic_verdicts()
    test_scaled_exits_bounds()
    test_derive_prices_uses_scaled()
    test_ml_features_length()
    print("edge tests (non-fixture) passed")
