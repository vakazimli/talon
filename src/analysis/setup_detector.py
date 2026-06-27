import logging

from src.orchestrator import TalonOrchestrator
from src.scanner.market_scanner import ScanResult

logger = logging.getLogger(__name__)

SETUP_DETECTION_PROMPT = """Analyze this market data for a short-term options trade setup.

Ticker: {ticker}
Price: ${price}
Daily Change: {change}%
Volume vs Average: {vol_ratio}x
Signals Detected: {signals}
Technicals:
  RSI: {rsi}
  EMA20: {ema20} | EMA50: {ema50}
  VWAP: {vwap}
  Bollinger: {bb_lower} - {bb_upper}
  Volume Z-Score: {vol_zscore}

Recent news (last 5 days):
{news}

Is this a tradeable short-term setup? Factor news catalysts into your direction
and confidence. Respond with JSON:
{{"tradeable": true/false, "direction": "bullish"/"bearish", "setup_type": "breakout"/"momentum"/"reversal"/"gap_fill", "confidence": 1-10, "reasoning": "one sentence"}}"""


class SetupDetector:
    """Uses cheap LLM model to evaluate whether scan results are tradeable setups."""

    def __init__(self, orchestrator: TalonOrchestrator):
        self.orchestrator = orchestrator
        # High bar: only treat a setup as tradeable when the detector is
        # clearly confident (1-10 scale).
        self.min_confidence = 7

    def detect_setups(
        self,
        scan_results: list[ScanResult],
        news_by_ticker: dict[str, list] | None = None,
    ) -> list[dict]:
        """Run setup detection on scan results. Returns list of detected setups."""
        detected = []
        news_by_ticker = news_by_ticker or {}
        for result in scan_results:
            if not result.signals:
                continue

            setup = self._evaluate(result, news_by_ticker.get(result.ticker, []))
            if setup and setup.get("confidence", 0) >= self.min_confidence:
                setup["scan_result"] = result
                detected.append(setup)

        detected.sort(key=lambda s: s.get("confidence", 0), reverse=True)
        return detected

    def _evaluate(self, result: ScanResult, news_items: list) -> dict | None:
        from src.scanner.news_scanner import format_for_prompt
        news_text = format_for_prompt(news_items) if news_items else "  (no recent news)"
        prompt = SETUP_DETECTION_PROMPT.format(
            ticker=result.ticker,
            price=result.price,
            change=result.daily_change_pct,
            vol_ratio=result.volume_vs_avg,
            signals=", ".join(result.signals),
            rsi=result.technicals.get("rsi", "N/A"),
            ema20=result.technicals.get("ema20", "N/A"),
            ema50=result.technicals.get("ema50", "N/A"),
            vwap=result.technicals.get("vwap", "N/A"),
            bb_lower=result.technicals.get("bb_lower", "N/A"),
            bb_upper=result.technicals.get("bb_upper", "N/A"),
            vol_zscore=result.technicals.get("volume_zscore", "N/A"),
            news=news_text,
        )

        response = self.orchestrator.call_model_json("scanner", prompt)
        if response is None:
            return None

        if not response.get("tradeable", False):
            logger.debug("Setup rejected for %s: %s", result.ticker, response.get("reasoning", ""))
            return None

        return {
            "ticker": result.ticker,
            "direction": response.get("direction", "bullish"),
            "setup_type": response.get("setup_type", "mixed"),
            "confidence": response.get("confidence", 0),
            "reasoning": response.get("reasoning", ""),
        }
