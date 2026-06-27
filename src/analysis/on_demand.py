"""On-demand single-ticker analysis.

Runs the same pipeline as a scheduled scan (scan -> detect -> options ->
IV filter -> score -> optional vision -> deterministic-priced synthesis)
for one user-supplied ticker, on demand. Used by the `/analyze` Telegram
command and the OpenClaw skill entry point.

It applies the SAME guardrails as the live pipeline (deterministic prices,
contract validation) but does NOT send alerts or write to the DB — it just
returns a structured result so a human can decide.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class OnDemandAnalyzer:
    def __init__(self, config, market_scanner, options_scanner, news_scanner,
                 setup_detector, scoring_engine, synthesizer, vision_analyst,
                 chart_dataframes_fn=None):
        self.config = config
        self.market_scanner = market_scanner
        self.options_scanner = options_scanner
        self.news_scanner = news_scanner
        self.setup_detector = setup_detector
        self.scoring_engine = scoring_engine
        self.synthesizer = synthesizer
        self.vision_analyst = vision_analyst
        self._chart_dataframes_fn = chart_dataframes_fn
        self.min_confidence = config.get("alerts", {}).get("min_confidence", 4)

    def analyze(self, ticker: str, deep: bool = False) -> dict:
        """Return a structured analysis dict for `ticker`.

        Keys: ticker, ok, stage (where it ended), and when a setup exists:
        direction, setup_type, score, gate_pass, card (dict) or note.
        """
        ticker = (ticker or "").upper().strip()
        if not ticker or not ticker.isalpha():
            return {"ticker": ticker, "ok": False, "note": "Invalid ticker."}

        try:
            results = self.market_scanner.scan_tickers([ticker])
        except Exception:
            logger.exception("On-demand scan failed for %s", ticker)
            return {"ticker": ticker, "ok": False, "note": "Market data unavailable."}
        if not results:
            return {"ticker": ticker, "ok": True, "stage": "scan",
                    "note": "No market data / insufficient history."}
        result = results[0]
        if not result.signals:
            return {"ticker": ticker, "ok": True, "stage": "scan",
                    "note": "No technical signals right now.",
                    "price": result.price}

        news_items = []
        if self.news_scanner.enabled:
            try:
                news_items = self.news_scanner.fetch_news(ticker)
            except Exception:
                news_items = []

        detected = self.setup_detector.detect_setups([result], {ticker: news_items})
        if not detected:
            return {"ticker": ticker, "ok": True, "stage": "detect",
                    "note": "Signals present but no tradeable setup.",
                    "signals": result.signals, "price": result.price}
        setup_data = detected[0]
        direction = setup_data["direction"]

        try:
            candidates = self.options_scanner.scan_options(ticker, direction)
        except Exception:
            candidates = []
        if not candidates:
            return {"ticker": ticker, "ok": True, "stage": "options",
                    "direction": direction,
                    "note": "Setup found but no liquid option candidates in the DTE window."}

        from src.scanner.news_scanner import extract_signals as extract_news_signals
        news_signals = extract_news_signals(news_items) if news_items else []
        scored = self.scoring_engine.score_setup(
            ticker=ticker,
            direction=direction,
            signals=result.signals,
            technicals=result.technicals,
            option_candidates=candidates,
            news_signals=news_signals,
        )

        vision_note = None
        if deep and self.vision_analyst.enabled and self._chart_dataframes_fn is not None:
            try:
                intraday, daily = self._chart_dataframes_fn(ticker)
                if intraday is not None and not intraday.empty:
                    v = self.vision_analyst.analyze(
                        ticker, direction, scored.setup_type, intraday,
                        scored.technicals, daily,
                    )
                    if v is not None:
                        vision_note = f"{v.verdict} (chart_score={v.chart_score}, {v.pattern})"
                        scored.score_breakdown["chart_analyst"] = v.chart_score
            except Exception:
                logger.exception("On-demand vision failed for %s", ticker)

        card = self.synthesizer.synthesize(scored, news_items=news_items)
        if card is None:
            return {"ticker": ticker, "ok": True, "stage": "synthesize",
                    "direction": direction, "score": scored.score,
                    "note": "Could not produce a validated, real-priced contract.",
                    "vision": vision_note}

        gate_pass = (
            self.scoring_engine.passes_threshold(scored.score, "shadow")
            and card.confidence >= self.min_confidence
        )
        return {
            "ticker": ticker,
            "ok": True,
            "stage": "card",
            "direction": card.direction,
            "setup_type": card.setup_type,
            "score": round(scored.score, 1),
            "gate_pass": gate_pass,
            "min_confidence": self.min_confidence,
            "vision": vision_note,
            "card": {
                "contract": card.contract,
                "entry_low": card.entry_low,
                "entry_high": card.entry_high,
                "target": card.target,
                "stop": card.stop,
                "sell_by": card.sell_by,
                "confidence": card.confidence,
                "rationale": card.rationale,
            },
        }


def format_analysis(res: dict) -> str:
    """Human-readable Telegram text for an analyze() result."""
    t = res.get("ticker", "?")
    if not res.get("ok"):
        return f"{t}: {res.get('note', 'analysis failed')}"
    if res.get("stage") != "card":
        base = f"{t}: {res.get('note', 'no setup')}"
        if res.get("score") is not None:
            base += f" (score {res['score']})"
        if res.get("vision"):
            base += f"\nChart: {res['vision']}"
        return base
    c = res["card"]
    gate = "PASSES alert gates" if res.get("gate_pass") else "below alert gates (informational)"
    lines = [
        f"{t} {res['direction'].upper()} — {res['setup_type']} | score {res['score']} | {gate}",
        f"{c['contract']}",
        f"Entry: ${c['entry_low']:.2f}-${c['entry_high']:.2f}",
        f"Target: ${c['target']:.2f} | Stop: ${c['stop']:.2f}",
        f"Hold: {c['sell_by']}",
        f"Confidence: {c['confidence']}/5 (min {res.get('min_confidence', 4)})",
        f"Why: {c['rationale']}",
    ]
    if res.get("vision"):
        lines.append(f"Chart: {res['vision']}")
    lines.append("Prices derived from live option quotes. Not financial advice.")
    return "\n".join(lines)
