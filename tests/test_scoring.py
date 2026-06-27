import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.scoring_engine import ScoringEngine


def make_engine(overrides=None):
    config = {
        "scoring": {
            "min_score_to_alert": 65,
            "min_score_to_alert_shadow": 55,
            "weights": {
                "technical_setup": 0.40,
                "source_reliability": 0.15,
                "risk_reward": 0.30,
                "news_catalyst": 0.15,
            },
        }
    }
    if overrides:
        config.update(overrides)
    return ScoringEngine(config)


def test_score_with_strong_signals():
    engine = make_engine()
    result = engine.score_setup(
        ticker="SPY",
        direction="bullish",
        signals=["breakout_above_resistance", "volume_spike", "ema_crossover"],
        technicals={"rsi": 55, "ema20": 530, "ema50": 525, "vwap": 528},
    )
    assert result.score > 0
    assert result.ticker == "SPY"
    assert result.direction == "bullish"
    assert "technical_setup" in result.score_breakdown
    assert result.setup_type == "breakout"


def test_score_with_no_signals():
    engine = make_engine()
    result = engine.score_setup(
        ticker="AAPL",
        direction="bearish",
        signals=[],
        technicals={"rsi": 50},
    )
    assert result.score >= 0
    assert result.score <= 100


def test_score_capped_at_100():
    engine = make_engine()
    result = engine.score_setup(
        ticker="QQQ",
        direction="bullish",
        signals=[
            "breakout_above_resistance", "volume_spike", "ema_crossover",
            "vwap_reclaim", "squeeze_breakout", "rsi_oversold",
        ],
        technicals={"rsi": 25, "ema20": 400, "ema50": 395, "vwap": 398},
    )
    assert result.score <= 100


def test_passes_threshold_shadow():
    # Runtime floors come from adaptive_thresholds DEFAULT_STATE (62 shadow,
    # 72 live) unless a learned_state.json overrides them.
    engine = make_engine()
    assert engine.passes_threshold(62, "shadow") is True
    assert engine.passes_threshold(61, "shadow") is False


def test_passes_threshold_live():
    engine = make_engine()
    assert engine.passes_threshold(72, "live") is True
    assert engine.passes_threshold(71, "live") is False


def test_classify_setup_types():
    engine = make_engine()

    r1 = engine.score_setup("X", "bullish", ["breakout_above_resistance"], {"rsi": 50})
    assert r1.setup_type == "breakout"

    r2 = engine.score_setup("X", "bearish", ["breakdown_below_support"], {"rsi": 50})
    assert r2.setup_type == "breakdown"

    r3 = engine.score_setup("X", "bullish", ["ema_crossover"], {"rsi": 50})
    assert r3.setup_type == "momentum"

    r4 = engine.score_setup("X", "bullish", ["rsi_oversold"], {"rsi": 25})
    assert r4.setup_type == "reversal"


def test_score_breakdown_keys():
    engine = make_engine()
    result = engine.score_setup(
        "SPY", "bullish", ["volume_spike"], {"rsi": 50}
    )
    expected_keys = {
        "technical_setup", "source_reliability", "risk_reward", "news_catalyst"
    }
    assert set(result.score_breakdown.keys()) == expected_keys


if __name__ == "__main__":
    test_score_with_strong_signals()
    test_score_with_no_signals()
    test_score_capped_at_100()
    test_passes_threshold_shadow()
    test_passes_threshold_live()
    test_classify_setup_types()
    test_score_breakdown_keys()
    print("All scoring tests passed!")
