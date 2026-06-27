import json
import logging
from dataclasses import dataclass

from src.analysis.scoring_engine import ScoredSetup
from src.orchestrator import TalonOrchestrator

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """You are a concise options trading analyst. Given a scored setup, produce a structured trade card.

Respond ONLY with valid JSON matching this exact schema:
{
  "contract": "TICKER STRIKE_TYPE EXPIRY",
  "entry_low": float,
  "entry_high": float,
  "target": float,
  "stop": float,
  "sell_by": "description of when to exit",
  "confidence": int (1-5),
  "rationale": "one to two sentences max"
}

Rules:
- You MUST pick a contract from the "Top Option Candidates" list provided. Do NOT invent contracts.
- Contract format: "SPY 530C 4/3" (ticker, strike+type, MM/DD expiry)
- Entry is a range based on the bid-ask of the candidate you chose
- SWING FOCUS: this is a 1-3 TRADING-DAY hold on a weekly/biweekly contract (5-14 DTE). Do NOT pick 0-1DTE; enough time value must remain that 1-3 days of theta won't gut the trade.
- Target: 20-40% gain over the 1-3 day hold. Be realistic — this is a short swing, not a multi-week runner.
- Stop: 25-35% loss. Size it so a normal pullback doesn't stop you out, but a thesis break does.
- Sell_by: specify a 1-3 day exit, e.g. "exit within 1-3 trading days" or "sell by <MM/DD> EOD". Never "sell by EOD today".
- Confidence 1-5 where 5 is highest conviction. Be honest and strict — only 4-5 for clean, high-probability setups; 2-3 if signals are mixed.
- Rationale: what makes this trade work, max 2 sentences
- Be concise. No fluff."""

SYNTHESIS_PROMPT = """Analyze this setup and produce a trade card.

Ticker: {ticker}
Direction: {direction}
Setup Type: {setup_type}
Score: {score}/100
Timeframe: {timeframe}

Score Breakdown:
{score_breakdown}

Signals: {signals}

Technicals:
  RSI: {rsi} | EMA20: {ema20} | EMA50: {ema50}
  VWAP: {vwap} | BB: {bb_lower}-{bb_upper}
  Volume Z-Score: {vol_zscore}

Recent News:
{news}

Top Option Candidates:
{candidates}

Produce the trade card JSON."""


@dataclass
class TradeCard:
    ticker: str
    direction: str
    contract: str
    entry_low: float
    entry_high: float
    target: float
    stop: float
    sell_by: str
    confidence: int
    rationale: str
    score: float
    setup_type: str


class Synthesizer:
    """Final LLM synthesis — expensive model, only for high-scoring setups."""

    MAX_PER_CYCLE = 5

    def __init__(self, orchestrator: TalonOrchestrator, config: dict | None = None):
        self.orchestrator = orchestrator
        self._calls_this_cycle = 0
        synth_cfg = (config or {}).get("synthesis", {}) if isinstance(config, dict) else {}
        # Code owns the numbers, not the LLM: target/stop are derived from the
        # real option mid by these fractions. Defaults sit inside the
        # prompt's stated 20-40% / 25-35% bands.
        self.target_pct = float(synth_cfg.get("target_pct", 0.30))
        self.stop_pct = float(synth_cfg.get("stop_pct", 0.30))
        # Volatility-aware scaling: a jumpier option (higher IV) gets wider
        # target/stop than a calm one, bounded so it never goes silly.
        self.reference_iv = float(synth_cfg.get("reference_iv", 0.30))
        self.target_pct_min = float(synth_cfg.get("target_pct_min", 0.20))
        self.target_pct_max = float(synth_cfg.get("target_pct_max", 0.45))
        self.stop_pct_min = float(synth_cfg.get("stop_pct_min", 0.20))
        self.stop_pct_max = float(synth_cfg.get("stop_pct_max", 0.40))

    def reset_cycle(self):
        self._calls_this_cycle = 0

    def synthesize(
        self,
        setup: ScoredSetup,
        news_items: list | None = None,
    ) -> TradeCard | None:
        if self._calls_this_cycle >= self.MAX_PER_CYCLE:
            logger.warning("Synthesizer call limit reached (%d). Skipping %s.",
                           self.MAX_PER_CYCLE, setup.ticker)
            return None

        candidates_text = self._format_candidates(setup.option_candidates)
        breakdown_text = "\n".join(
            f"  {k}: {v}" for k, v in setup.score_breakdown.items()
        )
        news_text = "  (no recent news)"
        if news_items:
            try:
                from src.scanner.news_scanner import format_for_prompt
                news_text = format_for_prompt(news_items)
            except Exception:
                pass

        prompt = SYNTHESIS_PROMPT.format(
            ticker=setup.ticker,
            direction=setup.direction,
            setup_type=setup.setup_type,
            score=setup.score,
            timeframe=setup.timeframe,
            score_breakdown=breakdown_text,
            signals=", ".join(setup.raw_signals),
            rsi=setup.technicals.get("rsi", "N/A"),
            ema20=setup.technicals.get("ema20", "N/A"),
            ema50=setup.technicals.get("ema50", "N/A"),
            vwap=setup.technicals.get("vwap", "N/A"),
            bb_lower=setup.technicals.get("bb_lower", "N/A"),
            bb_upper=setup.technicals.get("bb_upper", "N/A"),
            vol_zscore=setup.technicals.get("volume_zscore", "N/A"),
            candidates=candidates_text,
            news=news_text,
        )

        result = self.orchestrator.call_model(
            "synthesizer",
            prompt,
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
        )

        if result is None:
            return None

        card_data = self._parse_response(result.content)
        if card_data is None:
            return None

        self._calls_this_cycle += 1

        if not self._validate_contract(card_data, setup.option_candidates):
            logger.warning("Synthesizer output failed validation for %s. Rejecting.", setup.ticker)
            return None

        # Anti-hallucination: the LLM only PICKS the contract and writes the
        # rationale. All dollar values are derived in code from the matched
        # candidate's real bid/ask, never trusted from the model output.
        candidate = self._match_candidate(card_data, setup.option_candidates)
        if candidate is None:
            logger.warning(
                "Synthesizer: no matching option candidate for %s contract %r. Rejecting.",
                setup.ticker, card_data.get("contract"),
            )
            return None

        prices = self._derive_prices(candidate, setup.technicals)
        if prices is None:
            logger.warning(
                "Synthesizer: candidate for %s has no usable quote (bid/ask/mid). Rejecting.",
                setup.ticker,
            )
            return None

        return TradeCard(
            ticker=setup.ticker,
            direction=setup.direction,
            contract=card_data.get("contract", f"{setup.ticker} ???"),
            entry_low=prices["entry_low"],
            entry_high=prices["entry_high"],
            target=prices["target"],
            stop=prices["stop"],
            sell_by=card_data.get("sell_by", ""),
            confidence=min(5, max(1, card_data.get("confidence", 3))),
            rationale=card_data.get("rationale", ""),
            score=setup.score,
            setup_type=setup.setup_type,
        )

    def _match_candidate(self, card_data: dict, candidates: list):
        """Return the OptionCandidate that the LLM's chosen contract refers
        to, matched by strike (within 3%) and option type. None if no match."""
        if not candidates:
            return None
        contract = card_data.get("contract", "")
        parts = contract.split()
        if len(parts) < 2:
            return None
        strike_str = parts[1]
        otype = "call" if strike_str[-1:].upper() == "C" else (
            "put" if strike_str[-1:].upper() == "P" else None
        )
        try:
            strike_num = float("".join(ch for ch in strike_str if ch.isdigit() or ch == "."))
        except ValueError:
            return None
        if strike_num <= 0:
            return None

        best = None
        best_err = None
        for c in candidates:
            cs = getattr(c, "strike", 0) or 0
            if cs <= 0:
                continue
            if otype and getattr(c, "option_type", None) and getattr(c, "option_type") != otype:
                continue
            err = abs(strike_num - cs) / cs
            if err > 0.03:
                continue
            if best_err is None or err < best_err:
                best_err = err
                best = c
        return best

    def _derive_prices(self, candidate, technicals: dict | None = None) -> dict | None:
        """Compute entry range + target/stop from the candidate's REAL quote.

        entry is the live bid-ask (what you'd actually pay); target/stop are
        fractions of the mid, SCALED by the option's IV (relative to a
        reference) within configured bounds so volatile names get wider bands.
        Returns None if the quote is unusable.
        """
        bid = float(getattr(candidate, "bid", 0) or 0)
        ask = float(getattr(candidate, "ask", 0) or 0)
        mid = float(getattr(candidate, "mid", 0) or 0)
        if mid <= 0:
            mid = (bid + ask) / 2 if (bid + ask) > 0 else 0.0
        if mid <= 0:
            return None

        entry_low = round(bid, 2) if bid > 0 else round(mid * 0.97, 2)
        entry_high = round(ask, 2) if ask > 0 else round(mid * 1.03, 2)
        if entry_low <= 0 or entry_high <= 0 or entry_high < entry_low:
            return None

        target_pct, stop_pct = self._scaled_exit_pcts(candidate, technicals)
        target = round(mid * (1 + target_pct), 2)
        stop = round(mid * (1 - stop_pct), 2)
        return {
            "entry_low": entry_low,
            "entry_high": entry_high,
            "target": target,
            "stop": stop,
        }

    def _scaled_exit_pcts(self, candidate, technicals: dict | None) -> tuple[float, float]:
        """Scale base target/stop fractions by option IV vs the reference,
        clamped to configured bounds. Falls back to base pcts if IV is missing
        (then nudged by underlying ATR% if available)."""
        iv = float(getattr(candidate, "implied_volatility", 0) or 0)
        ratio = None
        if iv > 0 and self.reference_iv > 0:
            ratio = iv / self.reference_iv
        elif technicals:
            atr_pct = float(technicals.get("atr_pct", 0) or 0)
            if atr_pct > 0:
                # ~2% daily ATR is "normal"; scale around that.
                ratio = atr_pct / 0.02
        if ratio is None:
            return self.target_pct, self.stop_pct
        ratio = max(0.6, min(1.8, ratio))
        target = max(self.target_pct_min, min(self.target_pct_max, self.target_pct * ratio))
        stop = max(self.stop_pct_min, min(self.stop_pct_max, self.stop_pct * ratio))
        return round(target, 4), round(stop, 4)

    def _format_candidates(self, candidates: list) -> str:
        if not candidates:
            return "  No options data available"
        lines = []
        for c in candidates[:3]:
            symbol = getattr(c, "contract_symbol", "?")
            strike = getattr(c, "strike", 0)
            exp = getattr(c, "expiration", "?")
            otype = getattr(c, "option_type", "?")
            bid = getattr(c, "bid", 0)
            ask = getattr(c, "ask", 0)
            vol = getattr(c, "volume", 0)
            oi = getattr(c, "open_interest", 0)
            delta = getattr(c, "delta", 0)
            iv = getattr(c, "implied_volatility", 0)
            lines.append(
                f"  {symbol}: ${strike} {otype} exp={exp} "
                f"bid=${bid} ask=${ask} vol={vol} OI={oi} "
                f"delta={delta:.2f} IV={iv:.1%}"
            )
        return "\n".join(lines)

    def _validate_contract(self, card_data: dict, candidates: list) -> bool:
        """Hard validation: expiry must be within max_dte, strike must match a candidate."""
        from datetime import date, timedelta
        contract = card_data.get("contract", "")
        parts = contract.split()
        if len(parts) < 3:
            logger.warning("Invalid contract format: %s", contract)
            return False

        date_part = parts[-1]
        try:
            slash_parts = date_part.split("/")
            if len(slash_parts) == 2:
                month, day = int(slash_parts[0]), int(slash_parts[1])
                year = date.today().year
                exp_date = date(year, month, day)
                if exp_date < date.today():
                    exp_date = date(year + 1, month, day)
                dte = (exp_date - date.today()).days
                if dte > 15:
                    logger.warning("Contract %s has DTE=%d, exceeds limit. Rejected.", contract, dte)
                    return False
        except (ValueError, IndexError):
            logger.warning("Could not parse expiry from contract: %s", contract)
            return False

        strike_str = parts[1] if len(parts) >= 2 else ""
        try:
            strike_num = float("".join(c for c in strike_str if c.isdigit() or c == "."))
        except ValueError:
            return True

        if candidates:
            strikes = [getattr(c, "strike", 0) for c in candidates if getattr(c, "strike", 0) > 0]
            # Relative tolerance: 3% of the candidate strike. Works for both
            # $50 and $500 underlyings without false negatives/positives.
            if strikes and all(abs(strike_num - s) / s > 0.03 for s in strikes):
                logger.warning(
                    "Strike %.2f not within 3%% of any candidate strikes %s. Rejected.",
                    strike_num, strikes[:3],
                )
                return False

        return True

    def _parse_response(self, text: str) -> dict | None:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

        try:
            data = json.loads(text)
            required = ["contract", "entry_low", "entry_high", "target", "stop", "confidence", "rationale"]
            for key in required:
                if key not in data:
                    logger.error("Synthesizer response missing key: %s", key)
                    return None
            return data
        except json.JSONDecodeError:
            logger.error("Failed to parse synthesizer JSON: %s", text[:200])
            return None
