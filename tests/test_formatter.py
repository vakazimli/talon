import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.synthesizer import TradeCard
from src.delivery.formatter import (
    format_budget,
    format_sources,
    format_status,
    format_trade_card,
)

MAX_LENGTH = 500


def _make_card(**kwargs) -> TradeCard:
    defaults = {
        "ticker": "SPY",
        "direction": "bullish",
        "contract": "SPY 530C 4/3",
        "entry_low": 2.10,
        "entry_high": 2.30,
        "target": 3.40,
        "stop": 1.60,
        "sell_by": "sell by 4/2 EOD",
        "confidence": 4,
        "rationale": "Breakout above 528 resistance on 2x volume, VIX declining",
        "score": 78.0,
        "setup_type": "breakout",
    }
    defaults.update(kwargs)
    return TradeCard(**defaults)


def test_trade_card_basic_format():
    card = _make_card()
    msg = format_trade_card(card, mode="shadow")
    assert "SPY 530C 4/3" in msg
    assert "Bullish" in msg
    assert "$2.10" in msg
    assert "$2.30" in msg
    assert "$3.40" in msg
    assert "$1.60" in msg
    assert "SHADOW" in msg
    assert "\u2605" in msg  # filled star


def test_trade_card_under_500_chars():
    card = _make_card()
    msg = format_trade_card(card, mode="shadow")
    assert len(msg) <= MAX_LENGTH, f"Message is {len(msg)} chars, max is {MAX_LENGTH}"


def test_trade_card_bearish():
    card = _make_card(direction="bearish", contract="SPY 520P 4/3")
    msg = format_trade_card(card, mode="paper")
    assert "Bearish" in msg
    assert "PAPER" in msg
    assert "\U0001f534" in msg  # red circle


def test_trade_card_long_rationale_truncated():
    long_rationale = "A" * 400
    card = _make_card(rationale=long_rationale)
    msg = format_trade_card(card, mode="shadow")
    assert len(msg) <= MAX_LENGTH


def test_confidence_stars():
    for conf in range(1, 6):
        card = _make_card(confidence=conf)
        msg = format_trade_card(card)
        filled = msg.count("\u2605")
        empty = msg.count("\u2606")
        assert filled == conf
        assert empty == 5 - conf


def test_format_status():
    msg = format_status(
        mode="shadow",
        alerts_today=3,
        budget_status={"spent_usd": 0.85, "budget_usd": 2.00, "pct_used": 42.5, "remaining_usd": 1.15},
        scanning_paused=False,
    )
    assert "SHADOW" in msg
    assert "3" in msg
    assert "ACTIVE" in msg


def test_format_budget():
    msg = format_budget({
        "spent_usd": 1.23,
        "budget_usd": 2.00,
        "remaining_usd": 0.77,
        "pct_used": 61.5,
    })
    assert "$2.00" in msg
    assert "61.5%" in msg


def test_format_sources_empty():
    msg = format_sources([])
    assert "No source data" in msg


def test_format_sources_with_data():
    data = [
        {"name": "volume_spike", "score": 0.72, "total": 15},
        {"name": "ema_crossover", "score": 0.45, "total": 8},
    ]
    msg = format_sources(data)
    assert "volume_spike" in msg
    assert "ema_crossover" in msg


if __name__ == "__main__":
    test_trade_card_basic_format()
    test_trade_card_under_500_chars()
    test_trade_card_bearish()
    test_trade_card_long_rationale_truncated()
    test_confidence_stars()
    test_format_status()
    test_format_budget()
    test_format_sources_empty()
    test_format_sources_with_data()
    print("All formatter tests passed!")
