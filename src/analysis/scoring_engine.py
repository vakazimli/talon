import json
import logging
from dataclasses import dataclass, field
from datetime import datetime

from src.db.database import get_session
from src.db.models import SourceScore

logger = logging.getLogger(__name__)


@dataclass
class ScoredSetup:
    ticker: str
    direction: str
    setup_type: str
    timeframe: str
    score: float
    score_breakdown: dict
    raw_signals: list[str]
    sources_used: list[str]
    technicals: dict
    option_candidates: list = field(default_factory=list)


class ScoringEngine:
    """Scores setups from 0-100 using weighted components with source reliability feedback."""

    def __init__(self, config: dict):
        weights = config.get("scoring", {}).get("weights", {})
        self.config_weights = {
            "technical_setup": weights.get("technical_setup", 0.40),
            "source_reliability": weights.get("source_reliability", 0.15),
            "risk_reward": weights.get("risk_reward", 0.30),
            "news_catalyst": weights.get("news_catalyst", 0.15),
        }
        # `weights` resolves dynamically — adaptive tuner may override.
        self.weights = self.config_weights
        self.min_score = config.get("scoring", {}).get("min_score_to_alert", 65)
        self.min_score_shadow = config.get("scoring", {}).get("min_score_to_alert_shadow", 55)

    def _active_weights(self) -> dict:
        """Read tuned weights if active, else config defaults."""
        try:
            from src.learning.weight_tuner import get_active_weights
            return get_active_weights(self.config_weights)
        except Exception:
            return self.config_weights

    def score_setup(
        self,
        ticker: str,
        direction: str,
        signals: list[str],
        technicals: dict,
        option_candidates: list | None = None,
        news_signals: list[str] | None = None,
    ) -> ScoredSetup:
        source_scores = self._load_source_scores()
        disabled = {s for s in self.get_disabled_sources()}
        active_signals = [s for s in signals if s not in disabled]
        active_news = [s for s in (news_signals or []) if s not in disabled]
        if disabled & set(signals):
            logger.debug("Excluding disabled signals: %s", disabled & set(signals))

        tech_score = self._score_technical(active_signals, technicals, direction)
        reliability = self._score_source_reliability(
            active_signals + active_news, source_scores
        )
        rr_score = self._score_risk_reward(technicals, option_candidates or [])
        news_score = self._score_news(active_news)

        breakdown = {
            "technical_setup": round(tech_score, 1),
            "source_reliability": round(reliability, 1),
            "risk_reward": round(rr_score, 1),
            "news_catalyst": round(news_score, 1),
        }

        active_weights = self._active_weights()
        total = sum(breakdown[k] * active_weights.get(k, 0) for k in breakdown)

        setup_type = self._classify_setup(active_signals)
        timeframe = self._determine_timeframe(option_candidates)

        # Swing timeframe bonus: reward the 1-3 day-hold sweet spot
        # (weekly/biweekly contracts with enough time value), and give NO
        # bonus to 0-1DTE, which is theta/gamma-hostile for a multi-day hold.
        if timeframe == "biweekly":
            total += 10
        elif timeframe == "weekly":
            total += 6

        # Options liquidity bonus from computed Greeks
        if option_candidates:
            liquidity_bonus = self._score_options_liquidity(option_candidates)
            total += liquidity_bonus

        # Hour-of-day modifier (no-op when stats sample size is too low).
        try:
            from src.learning.hour_of_day import hour_score_modifier
            modifier = hour_score_modifier()
            if modifier != 1.0:
                total *= modifier
                breakdown["hour_modifier"] = round(modifier, 3)
        except Exception:
            pass

        # ML win-probability modifier (no-op until a model is trained+active).
        try:
            from src.learning.ml_model import score_modifier as ml_score_modifier
            ml_mod = ml_score_modifier(breakdown, len(active_signals))
            if ml_mod != 1.0:
                total *= ml_mod
                breakdown["ml_modifier"] = round(ml_mod, 3)
        except Exception:
            pass

        total = min(100.0, max(0.0, total))

        sources_used = ["technical_patterns"]
        if active_news:
            sources_used.append("finnhub_news")

        return ScoredSetup(
            ticker=ticker,
            direction=direction,
            setup_type=setup_type,
            timeframe=timeframe,
            score=round(total, 1),
            score_breakdown=breakdown,
            raw_signals=active_signals,
            sources_used=sources_used,
            technicals=technicals,
            option_candidates=option_candidates or [],
        )

    def _score_technical(
        self, signals: list[str], technicals: dict, direction: str
    ) -> float:
        score = 0.0
        signal_values = {
            "breakout_above_resistance": 30,
            "breakdown_below_support": 30,
            "volume_spike": 20,
            "ema_crossover": 15,
            "vwap_reclaim": 15,
            "squeeze_breakout": 25,
            "gap_fill": 10,
            "rsi_oversold": 10,
            "rsi_overbought": 10,
        }

        for sig in signals:
            score += signal_values.get(sig, 5)

        rsi = technicals.get("rsi", 50)
        if direction == "bullish" and rsi < 40:
            score += 10  # oversold bounce setup
        elif direction == "bearish" and rsi > 60:
            score += 10  # overbought breakdown

        return min(100.0, score)

    def _score_news(self, news_signals: list[str]) -> float:
        """0-100 catalyst score based on number/type of news events.

        Returns a neutral 50 baseline when no news signals are present so
        the news_catalyst component doesn't bias scores in either
        direction for tickers without coverage. Positive signals like
        upgrades / earnings beats add extra points.
        """
        if not news_signals:
            return 50.0
        per_signal = 12.0
        score = 50.0 + per_signal * len(news_signals)
        # Mild bonus for clearly directional events
        positive = {"news_upgrade"}
        negative = {"news_downgrade"}
        if any(s in positive for s in news_signals):
            score += 5
        if any(s in negative for s in news_signals):
            score -= 5
        return max(0.0, min(100.0, score))

    def _score_source_reliability(
        self, signals: list[str], source_scores: dict[str, float]
    ) -> float:
        if not signals:
            return 50.0

        scores = []
        for sig in signals:
            reliability = source_scores.get(sig, source_scores.get("technical_patterns", 0.50))
            scores.append(reliability)

        avg = sum(scores) / len(scores)
        return avg * 100  # Convert 0-1 range to 0-100

    def _score_risk_reward(self, technicals: dict, candidates: list) -> float:
        if not candidates:
            return 50.0

        score = 50.0
        for c in candidates:
            mid = getattr(c, "mid", 0) or 0
            if mid > 0:
                spread_pct = getattr(c, "spread_pct", 100)
                if spread_pct < 3:
                    score += 15
                elif spread_pct < 7:
                    score += 8

                oi = getattr(c, "open_interest", 0) or 0
                if oi > 2000:
                    score += 10
                elif oi > 1000:
                    score += 5

        return min(100.0, score)

    def _load_source_scores(self) -> dict[str, float]:
        """Returns reliability per source. Disabled sources contribute 0
        so they're effectively excluded from setup scoring."""
        try:
            with get_session() as session:
                rows = session.query(SourceScore).all()
                scores = {}
                for row in rows:
                    key = row.source_subtype or row.source_name
                    if row.enabled is False:
                        scores[key] = 0.0
                    else:
                        scores[key] = row.reliability_score
                return scores
        except Exception:
            return {}

    def get_disabled_sources(self) -> list[str]:
        try:
            with get_session() as session:
                rows = session.query(SourceScore).filter(SourceScore.enabled.is_(False)).all()
                return [r.source_subtype or r.source_name for r in rows]
        except Exception:
            return []

    def _classify_setup(self, signals: list[str]) -> str:
        if "breakout_above_resistance" in signals or "squeeze_breakout" in signals:
            return "breakout"
        if "breakdown_below_support" in signals:
            return "breakdown"
        if "ema_crossover" in signals or "vwap_reclaim" in signals:
            return "momentum"
        if "rsi_oversold" in signals or "rsi_overbought" in signals:
            return "reversal"
        if "gap_fill" in signals:
            return "gap_fill"
        return "mixed"

    def _determine_timeframe(self, candidates: list) -> str:
        if not candidates:
            return "weekly"
        min_dte = 999
        for c in candidates:
            exp = getattr(c, "expiration", "")
            if exp:
                try:
                    from datetime import date
                    exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                    dte = (exp_date - date.today()).days
                    min_dte = min(min_dte, dte)
                except ValueError:
                    pass
        # Buckets aligned to the swing strategy: 0-1 DTE is "ultra_short"
        # (outside our preferred window), 2-7 DTE "weekly", 8-15 DTE
        # "biweekly" (the 1-3 day-hold sweet spot).
        if min_dte <= 1:
            return "ultra_short"
        elif min_dte <= 7:
            return "weekly"
        return "biweekly"

    def _score_options_liquidity(self, candidates: list) -> float:
        """Bonus points for options with good delta, tight spreads, high liquidity."""
        if not candidates:
            return 0.0
        bonus = 0.0
        best = candidates[0] if candidates else None
        if best:
            delta = abs(getattr(best, "delta", 0) or 0)
            if 0.30 <= delta <= 0.40:
                bonus += 3  # perfect sweet spot
            elif 0.25 <= delta <= 0.50:
                bonus += 1.5

            spread = getattr(best, "spread_pct", 100) or 100
            if spread < 3:
                bonus += 2
            elif spread < 6:
                bonus += 1

        return min(5.0, bonus)

    def passes_threshold(self, score: float, mode: str) -> bool:
        from src.learning.adaptive_thresholds import get_threshold
        config_default = self.min_score_shadow if mode == "shadow" else self.min_score
        threshold = get_threshold(mode, config_default)
        return score >= threshold
