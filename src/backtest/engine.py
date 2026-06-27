"""BSM-modeled backtest harness.

Validates Talon's QUANT layer (technical signals -> scoring -> thresholds ->
modeled option move) over historical daily bars, so the score floors and the
swing thesis can be calibrated without waiting weeks for live data.

Honest scope / caveats:
  * No LLM setup-judgment is replayed (too slow/costly to backtest); we use
    the rule-based signals + scoring + a direction heuristic. Live trading
    additionally applies the LLM "tradeable?" gate and the critic, so live
    selectivity is HIGHER than what the backtest assumes.
  * The free Alpaca tier has no historical options chain, so each setup's
    option is SYNTHETIC: an ATM contract priced with Black-Scholes using an
    IV estimate (HV20 or an override). P&L is therefore modeled, not actual
    fills. Intraday target/stop touches are approximated on daily closes.

Despite the caveats, this is the fastest way to sanity-check the edge and
pick a score floor. Treat results as directional, not gospel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.analysis.iv_filter import hv20_from_returns
from src.analysis.scoring_engine import ScoringEngine
from src.scanner.greeks import bs_price
from src.scanner.market_scanner import MarketScanner

logger = logging.getLogger(__name__)

_BEARISH_SIGNALS = {"breakdown_below_support", "rsi_overbought"}


@dataclass
class BacktestResult:
    samples: int
    wins: int
    win_rate: float
    avg_pnl_pct: float
    by_signal: dict
    by_score_bucket: dict
    suggested_floor: float | None
    trades: list = field(default_factory=list)


class BacktestEngine:
    def __init__(self, config: dict):
        self.config = config
        self.scoring = ScoringEngine(config)
        self.scanner = MarketScanner(config)
        synth = config.get("synthesis", {})
        self.target_pct = float(synth.get("target_pct", 0.30))
        self.stop_pct = float(synth.get("stop_pct", 0.30))

    @staticmethod
    def _infer_direction(signals: list[str], technicals: dict) -> str:
        if any(s in _BEARISH_SIGNALS for s in signals):
            return "bearish"
        if technicals.get("rsi", 50) > 70:
            return "bearish"
        return "bullish"

    def _load_history(self, ticker: str, lookback_days: int):
        provider = self.scanner._alpaca
        df = None
        if provider.enabled:
            df = provider.get_daily_bars(ticker, lookback_days=lookback_days + 90)
        if df is None or df.empty:
            try:
                import yfinance as yf
                period = "2y" if lookback_days > 365 else "1y"
                df = yf.Ticker(ticker).history(period=period, interval="1d")
            except Exception:
                logger.exception("Backtest: history fetch failed for %s", ticker)
                return None
        if df is None or df.empty:
            return None
        # Normalize to lower-case OHLCV columns.
        rename = {}
        for c in df.columns:
            lc = c.lower()
            if lc in ("open", "high", "low", "close", "volume"):
                rename[c] = lc
        df = df.rename(columns=rename)
        needed = {"close", "high", "low", "volume"}
        if not needed.issubset(set(df.columns)):
            return None
        return df

    def run(
        self,
        tickers: list[str],
        lookback_days: int = 365,
        hold_days: int = 3,
        entry_dte: int = 9,
        warmup: int = 60,
        iv_override: float | None = None,
    ) -> BacktestResult:
        trades: list[dict] = []
        for ticker in tickers:
            df = self._load_history(ticker, lookback_days)
            if df is None or len(df) < warmup + hold_days + 1:
                logger.info("Backtest: insufficient history for %s", ticker)
                continue
            close = df["close"]
            high = df["high"]
            low = df["low"]
            volume = df["volume"]
            n = len(df)
            for i in range(warmup, n - hold_days - 1):
                w_close = close.iloc[: i + 1]
                w_high = high.iloc[: i + 1]
                w_low = low.iloc[: i + 1]
                w_vol = volume.iloc[: i + 1]
                try:
                    technicals = self.scanner._compute_technicals(w_close, w_high, w_low, w_vol)
                    signals = self.scanner._detect_signals(w_close, w_high, w_low, w_vol, technicals)
                except Exception:
                    continue
                if not signals:
                    continue
                direction = self._infer_direction(signals, technicals)
                scored = self.scoring.score_setup(
                    ticker=ticker,
                    direction=direction,
                    signals=signals,
                    technicals=technicals,
                    option_candidates=[],
                    news_signals=[],
                )
                trade = self._simulate_trade(
                    ticker, direction, signals, scored.score,
                    close, i, hold_days, entry_dte, iv_override,
                )
                if trade is not None:
                    trades.append(trade)
        return self._aggregate(trades)

    def _simulate_trade(self, ticker, direction, signals, score, close, i,
                        hold_days, entry_dte, iv_override):
        s0 = float(close.iloc[i])
        if s0 <= 0:
            return None
        iv = iv_override if iv_override else hv20_from_returns(close.iloc[: i + 1])
        iv = max(float(iv or 0), 0.15)
        opt_type = "call" if direction == "bullish" else "put"
        strike = round(s0)
        entry = bs_price(s0, strike, entry_dte, iv, opt_type)
        if entry <= 0.05:
            return None
        target = entry * (1 + self.target_pct)
        stop = entry * (1 - self.stop_pct)

        exit_reason = "time_exit"
        exit_price = entry
        for j in range(1, hold_days + 1):
            sj = float(close.iloc[i + j])
            opt = bs_price(sj, strike, entry_dte - j, iv, opt_type)
            if opt >= target:
                exit_reason, exit_price = "target", target
                break
            if opt <= stop:
                exit_reason, exit_price = "stop", stop
                break
            exit_price = opt
        pnl_pct = (exit_price / entry - 1) * 100 if entry > 0 else 0
        return {
            "ticker": ticker,
            "direction": direction,
            "signals": signals,
            "score": round(score, 1),
            "exit_reason": exit_reason,
            "pnl_pct": round(pnl_pct, 1),
        }

    def _aggregate(self, trades: list[dict]) -> BacktestResult:
        n = len(trades)
        if n == 0:
            return BacktestResult(0, 0, 0.0, 0.0, {}, {}, None, [])
        wins = sum(1 for t in trades if t["pnl_pct"] > 0)
        avg_pnl = sum(t["pnl_pct"] for t in trades) / n

        by_signal: dict[str, dict] = {}
        for t in trades:
            for sig in t["signals"]:
                b = by_signal.setdefault(sig, {"n": 0, "wins": 0, "pnl": 0.0})
                b["n"] += 1
                b["wins"] += 1 if t["pnl_pct"] > 0 else 0
                b["pnl"] += t["pnl_pct"]
        for sig, b in by_signal.items():
            b["win_rate"] = round(b["wins"] / b["n"], 3)
            b["avg_pnl_pct"] = round(b["pnl"] / b["n"], 1)

        buckets = [(0, 50), (50, 60), (60, 70), (70, 80), (80, 101)]
        by_bucket: dict[str, dict] = {}
        for lo, hi in buckets:
            sub = [t for t in trades if lo <= t["score"] < hi]
            if not sub:
                continue
            w = sum(1 for t in sub if t["pnl_pct"] > 0)
            by_bucket[f"{lo}-{hi}"] = {
                "n": len(sub),
                "win_rate": round(w / len(sub), 3),
                "avg_pnl_pct": round(sum(t["pnl_pct"] for t in sub) / len(sub), 1),
            }

        # Suggested floor: lowest bucket lower-bound with win_rate >= 0.5 and
        # a reasonable sample size.
        suggested = None
        for lo, hi in buckets:
            key = f"{lo}-{hi}"
            stats = by_bucket.get(key)
            if stats and stats["n"] >= 20 and stats["win_rate"] >= 0.50:
                suggested = float(lo)
                break

        return BacktestResult(
            samples=n,
            wins=wins,
            win_rate=round(wins / n, 3),
            avg_pnl_pct=round(avg_pnl, 1),
            by_signal=by_signal,
            by_score_bucket=by_bucket,
            suggested_floor=suggested,
            trades=trades,
        )


def format_report(res: BacktestResult) -> str:
    if res.samples == 0:
        return "Backtest: no trades generated (insufficient history or no signals)."
    lines = [
        "=== Talon Backtest (BSM-modeled, quant layer only) ===",
        f"Trades: {res.samples} | Win rate: {res.win_rate:.0%} | Avg P&L: {res.avg_pnl_pct:+.1f}%",
        "",
        "By score bucket:",
    ]
    for bucket, s in res.by_score_bucket.items():
        lines.append(f"  {bucket:>7}: n={s['n']:<4} win={s['win_rate']:.0%} avgP&L={s['avg_pnl_pct']:+.1f}%")
    lines.append("")
    lines.append("By signal:")
    for sig, s in sorted(res.by_signal.items(), key=lambda kv: kv[1]["win_rate"], reverse=True):
        lines.append(f"  {sig:<26}: n={s['n']:<4} win={s['win_rate']:.0%} avgP&L={s['avg_pnl_pct']:+.1f}%")
    lines.append("")
    if res.suggested_floor is not None:
        lines.append(f"Suggested score floor: >= {res.suggested_floor:.0f}")
    else:
        lines.append("Suggested score floor: inconclusive (no bucket with n>=20 and win>=50%).")
    lines.append("Caveat: modeled option P&L, no LLM gate/critic, no real chain. Directional only.")
    return "\n".join(lines)
