"""Run a BSM-modeled backtest of Talon's quant layer.

Usage:
    python -m scripts.backtest [--days 365] [--hold 3] [--iv 0.0] [TICKER ...]

With no tickers, uses the watchlist from config/settings.yaml. Prints a
report and writes data/backtest_report.json.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.backtest.engine import BacktestEngine, format_report
from src.main import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Talon backtest")
    parser.add_argument("tickers", nargs="*", help="Tickers (default: watchlist)")
    parser.add_argument("--days", type=int, default=365, help="Lookback days")
    parser.add_argument("--hold", type=int, default=3, help="Max hold (trading days)")
    parser.add_argument("--dte", type=int, default=9, help="Synthetic contract DTE at entry")
    parser.add_argument("--iv", type=float, default=0.0, help="IV override (0 = estimate from HV20)")
    args = parser.parse_args()

    config = load_config()
    tickers = args.tickers
    if not tickers:
        wl = config.get("watchlist", {})
        tickers = list(wl.get("tier1", [])) + list(wl.get("tier2", []))

    engine = BacktestEngine(config)
    res = engine.run(
        tickers,
        lookback_days=args.days,
        hold_days=args.hold,
        entry_dte=args.dte,
        iv_override=args.iv if args.iv > 0 else None,
    )
    print(format_report(res))

    out = Path(__file__).resolve().parent.parent / "data" / "backtest_report.json"
    payload = {
        "samples": res.samples,
        "win_rate": res.win_rate,
        "avg_pnl_pct": res.avg_pnl_pct,
        "by_score_bucket": res.by_score_bucket,
        "by_signal": res.by_signal,
        "suggested_floor": res.suggested_floor,
        "params": {"days": args.days, "hold": args.hold, "dte": args.dte,
                   "iv_override": args.iv, "tickers": tickers},
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
