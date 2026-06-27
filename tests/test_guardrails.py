import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.synthesizer import Synthesizer
from src.scanner.option_quotes import OptionQuote, _is_sane_quote


class _Cand:
    def __init__(self, strike, option_type="call", bid=2.0, ask=2.2, mid=2.1):
        self.strike = strike
        self.option_type = option_type
        self.bid = bid
        self.ask = ask
        self.mid = mid


def _synth():
    return Synthesizer(orchestrator=None, config={"synthesis": {"target_pct": 0.30, "stop_pct": 0.30}})


def test_derive_prices_from_real_quote():
    s = _synth()
    prices = s._derive_prices(_Cand(530, bid=2.0, ask=2.2, mid=2.1))
    assert prices["entry_low"] == 2.0
    assert prices["entry_high"] == 2.2
    assert prices["target"] == round(2.1 * 1.30, 2)
    assert prices["stop"] == round(2.1 * 0.70, 2)


def test_derive_prices_rejects_dead_quote():
    s = _synth()
    assert s._derive_prices(_Cand(530, bid=0, ask=0, mid=0)) is None


def test_match_candidate_by_strike_and_type():
    s = _synth()
    cands = [_Cand(530, "call"), _Cand(525, "put")]
    assert s._match_candidate({"contract": "SPY 530C 7/3"}, cands) is cands[0]
    assert s._match_candidate({"contract": "SPY 525P 7/3"}, cands) is cands[1]
    # No strike within 3% -> no match (never fabricate).
    assert s._match_candidate({"contract": "SPY 999C 7/3"}, cands) is None


def test_is_sane_quote():
    assert _is_sane_quote(OptionQuote(bid=1.0, ask=1.2, last=1.1, mid=1.1))
    assert not _is_sane_quote(None)
    assert not _is_sane_quote(OptionQuote(bid=0, ask=0, last=0, mid=0))
    # crossed market
    assert not _is_sane_quote(OptionQuote(bid=1.5, ask=1.0, last=0, mid=1.25))
    # NaN
    assert not _is_sane_quote(OptionQuote(bid=float("nan"), ask=1.0, last=1.0, mid=1.0))


def test_candidate_to_contract():
    from src.main import Talon

    class C:
        strike = 530.0
        option_type = "call"
        expiration = "2026-07-03"

    assert Talon._candidate_to_contract("SPY", C()) == "SPY 530C 7/3"

    class P:
        strike = 282.5
        option_type = "put"
        expiration = "2026-05-08"

    assert Talon._candidate_to_contract("IWM", P()) == "IWM 282.5P 5/8"


def test_circuit_breaker_consecutive_losses(tmp_path):
    db = str(tmp_path / "breaker.db")
    from src.db.database import init_database, get_session
    from src.db import models

    init_database(db)
    with get_session() as session:
        for _ in range(5):
            a = models.Alert(
                ticker="X", direction="bullish", contract="X 1C 1/1",
                sent_at=datetime.utcnow().isoformat(), mode="shadow",
            )
            session.add(a)
            session.flush()
            session.add(models.Outcome(
                alert_id=a.id, entry_filled=True, pnl_pct=-10.0,
                resolved_at=datetime.utcnow().isoformat(), exit_reason="stop",
            ))
        session.commit()

    from src.risk import breakers
    open_positions, daily_pnl, streak, weekly_pnl = breakers.evaluate({})
    assert streak >= 4
    assert daily_pnl <= -40.0  # 5 x -10%


if __name__ == "__main__":
    test_derive_prices_from_real_quote()
    test_derive_prices_rejects_dead_quote()
    test_match_candidate_by_strike_and_type()
    test_is_sane_quote()
    test_candidate_to_contract()
    print("guardrail tests passed (excluding DB fixture test)")
