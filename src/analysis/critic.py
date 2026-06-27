"""Bear-case critic agent.

A second, adversarial LLM pass that argues the strongest case AGAINST a
synthesized trade before it is alerted. Inspired by the multi-agent
"bull vs bear researcher + risk manager" pattern: a one-sided thesis is the
most common false positive, so an explicit devil's-advocate step raises
conviction quality and supports the "only very confident" goal.

Runs only on setups that already passed scoring + synthesis, so cost is
bounded. Returns a verdict the caller uses to VETO or downgrade the alert.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

CRITIC_SYSTEM_PROMPT = (
    "You are a disciplined options risk manager reviewing a proposed SHORT-TERM "
    "trade (1-3 day hold). Your job is to argue the STRONGEST honest case AGAINST "
    "taking it: for a bullish/call idea make the bear case, for a bearish/put idea "
    "make the bull case. Consider trend conflict, overextension, weak or ambiguous "
    "catalysts, poor risk/reward, and crowded/obvious setups. Be decisive but "
    "fair: only VETO setups that are genuinely fragile or one-sided. Respond ONLY "
    "with valid JSON, no markdown."
)

CRITIC_PROMPT = """Proposed trade:
Ticker: {ticker}
Direction: {direction}
Setup: {setup_type} | Score: {score}/100
Contract: {contract}
Entry: ${entry_low}-${entry_high} | Target: ${target} | Stop: ${stop}
Rationale given: {rationale}

Signals: {signals}
Technicals: RSI={rsi} EMA20={ema20} EMA50={ema50} VWAP={vwap} BB={bb_lower}-{bb_upper}

Recent news:
{news}

Argue the case against this trade, then decide. Schema (JSON only):
{{
  "counter_case": "1-2 sentences: the strongest argument against",
  "key_risk": "the single biggest risk in a few words",
  "verdict": "PROCEED" | "CAUTION" | "VETO"
}}

Use VETO only when the setup is fragile or the thesis is clearly one-sided.
Use CAUTION when there is a real but not disqualifying concern."""


@dataclass
class CriticVerdict:
    verdict: str  # PROCEED | CAUTION | VETO
    counter_case: str = ""
    key_risk: str = ""


class RiskCritic:
    def __init__(self, orchestrator, config: dict | None = None):
        self.orchestrator = orchestrator
        cfg = (config or {}).get("critic", {}) if isinstance(config, dict) else {}
        self.enabled = bool(cfg.get("enabled", True))

    def critique(self, scored, card, news_items: list | None = None) -> CriticVerdict:
        """Return a CriticVerdict. Fails OPEN (PROCEED) on any error so the
        critic can never silently block the whole pipeline."""
        news_text = "  (no recent news)"
        if news_items:
            try:
                from src.scanner.news_scanner import format_for_prompt
                news_text = format_for_prompt(news_items)
            except Exception:
                pass

        t = scored.technicals or {}
        prompt = CRITIC_PROMPT.format(
            ticker=scored.ticker,
            direction=scored.direction,
            setup_type=scored.setup_type,
            score=scored.score,
            contract=card.contract,
            entry_low=card.entry_low,
            entry_high=card.entry_high,
            target=card.target,
            stop=card.stop,
            rationale=card.rationale,
            signals=", ".join(scored.raw_signals),
            rsi=t.get("rsi", "N/A"),
            ema20=t.get("ema20", "N/A"),
            ema50=t.get("ema50", "N/A"),
            vwap=t.get("vwap", "N/A"),
            bb_lower=t.get("bb_lower", "N/A"),
            bb_upper=t.get("bb_upper", "N/A"),
            news=news_text,
        )
        try:
            data = self.orchestrator.call_model_json(
                "critic", prompt, system_prompt=CRITIC_SYSTEM_PROMPT
            )
        except Exception:
            logger.exception("Critic call failed for %s; proceeding.", scored.ticker)
            return CriticVerdict("PROCEED")
        if not data:
            return CriticVerdict("PROCEED")
        verdict = str(data.get("verdict", "PROCEED")).upper()
        if verdict not in ("PROCEED", "CAUTION", "VETO"):
            verdict = "PROCEED"
        return CriticVerdict(
            verdict=verdict,
            counter_case=str(data.get("counter_case", "")),
            key_risk=str(data.get("key_risk", "")),
        )
