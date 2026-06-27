"""Daily and weekly performance reviews with LLM-generated insights."""

import json
import logging
from datetime import date, datetime, timedelta

from src.db.database import get_session
from src.db.models import (
    Alert,
    CounterfactualOutcome,
    Outcome,
    PerformanceReview,
    SourceScore,
)
from src.orchestrator import TalonOrchestrator

logger = logging.getLogger(__name__)


class PostmortemRunner:
    def __init__(self, orchestrator: TalonOrchestrator, config: dict):
        self.orchestrator = orchestrator
        self.config = config

    async def run_daily(self) -> dict | None:
        today = date.today().isoformat()
        return await self._run_review("daily", today, today)

    async def run_weekly(self) -> dict | None:
        today = date.today()
        week_start = (today - timedelta(days=7)).isoformat()
        week_end = today.isoformat()
        return await self._run_review("weekly", week_start, week_end)

    async def _run_review(
        self, period_type: str, start: str, end: str
    ) -> dict | None:
        with get_session() as session:
            alerts = (
                session.query(Alert)
                .filter(Alert.sent_at >= start, Alert.sent_at <= end + "T23:59:59")
                .all()
            )
            outcomes = (
                session.query(Outcome)
                .join(Alert)
                .filter(Alert.sent_at >= start, Alert.sent_at <= end + "T23:59:59")
                .filter(Outcome.resolved_at.isnot(None))
                .all()
            )
            sources = session.query(SourceScore).all()
            cfs = (
                session.query(CounterfactualOutcome)
                .filter(CounterfactualOutcome.created_at >= start)
                .filter(CounterfactualOutcome.created_at <= end + "T23:59:59")
                .filter(CounterfactualOutcome.resolved_at.isnot(None))
                .filter(CounterfactualOutcome.pnl_pct.isnot(None))
                .all()
            )

        # Counterfactual (filtered-but-tracked) stats: did we leave winners on
        # the table? High win rate here means the gates may be too strict.
        cf_resolved = len(cfs)
        cf_wins = sum(1 for c in cfs if (c.pnl_pct or 0) > 0)
        cf_win_rate = round(cf_wins / cf_resolved, 3) if cf_resolved else None

        if not alerts:
            logger.info("No alerts in period %s to %s. Skipping review.", start, end)
            # Still surface counterfactual info even with no alerts.
            return {
                "period": f"{start} to {end}",
                "total_alerts": 0,
                "resolved": 0,
                "wins": 0,
                "losses": 0,
                "avg_pnl_pct": 0,
                "counterfactual": {
                    "resolved": cf_resolved,
                    "wins": cf_wins,
                    "win_rate": cf_win_rate,
                },
            }

        wins = [o for o in outcomes if (o.pnl_pct or 0) > 0]
        losses = [o for o in outcomes if (o.pnl_pct or 0) < 0]
        total_pnl = sum(o.pnl_pct or 0 for o in outcomes)
        avg_pnl = total_pnl / len(outcomes) if outcomes else 0

        best = max(outcomes, key=lambda o: o.pnl_pct or 0) if outcomes else None
        worst = min(outcomes, key=lambda o: o.pnl_pct or 0) if outcomes else None

        top_sources = sorted(sources, key=lambda s: s.reliability_score or 0, reverse=True)[:3]
        bottom_sources = sorted(sources, key=lambda s: s.reliability_score or 0)[:3]

        summary_data = {
            "period": f"{start} to {end}",
            "total_alerts": len(alerts),
            "resolved": len(outcomes),
            "wins": len(wins),
            "losses": len(losses),
            "total_pnl_pct": round(total_pnl, 2),
            "avg_pnl_pct": round(avg_pnl, 2),
            "top_sources": [
                {"name": s.source_subtype or s.source_name, "score": s.reliability_score}
                for s in top_sources
            ],
            "bottom_sources": [
                {"name": s.source_subtype or s.source_name, "score": s.reliability_score}
                for s in bottom_sources
            ],
            "counterfactual": {
                "resolved": cf_resolved,
                "wins": cf_wins,
                "win_rate": cf_win_rate,
            },
        }

        lessons = await self._generate_insights(summary_data)

        best_json = None
        if best:
            alert = next((a for a in alerts if a.id == best.alert_id), None)
            if alert:
                best_json = json.dumps({
                    "ticker": alert.ticker, "pnl_pct": best.pnl_pct,
                    "contract": alert.contract, "exit_reason": best.exit_reason,
                })

        worst_json = None
        if worst:
            alert = next((a for a in alerts if a.id == worst.alert_id), None)
            if alert:
                worst_json = json.dumps({
                    "ticker": alert.ticker, "pnl_pct": worst.pnl_pct,
                    "contract": alert.contract, "exit_reason": worst.exit_reason,
                })

        review = PerformanceReview(
            period_type=period_type,
            period_start=start,
            period_end=end,
            total_alerts=len(alerts),
            winning_alerts=len(wins),
            losing_alerts=len(losses),
            total_pnl_pct=total_pnl,
            avg_pnl_pct=avg_pnl,
            best_trade=best_json,
            worst_trade=worst_json,
            top_sources=json.dumps(summary_data["top_sources"]),
            bottom_sources=json.dumps(summary_data["bottom_sources"]),
            lessons=lessons,
            source_score_updates=json.dumps({}),
            created_at=datetime.utcnow().isoformat(),
        )

        with get_session() as session:
            session.add(review)
            session.commit()

        logger.info("%s review saved: %d alerts, %d wins, %d losses, avg P&L=%.1f%%",
                     period_type, len(alerts), len(wins), len(losses), avg_pnl)

        # Close the self-improvement loop. Three adapters run weekly:
        #   1. Threshold adjuster — raise/lower alert score floor based
        #      on rolling 7-day win rate.
        #   2. Weight tuner — random-search scoring weights against
        #      stored breakdowns + outcomes (kicks in once we have 100+
        #      resolved trades; no-op until then).
        #   3. Hour-of-day stats — record per-(scan-hour) win rate so
        #      the scoring engine can favour productive scan times.
        if period_type == "weekly":
            try:
                from src.learning.adaptive_thresholds import adjust_thresholds
                new_state = adjust_thresholds(reason="weekly_postmortem")
                summary_data["adaptive_thresholds"] = {
                    "shadow": new_state.get("min_score_to_alert_shadow"),
                    "live": new_state.get("min_score_to_alert"),
                    "win_rate_7d": new_state.get("win_rate_7d"),
                    "samples_7d": new_state.get("samples_7d"),
                }
            except Exception:
                logger.exception("Adaptive threshold adjustment failed.")

            try:
                from src.learning.weight_tuner import tune_weights
                tuned = tune_weights(reason="weekly_postmortem")
                tw = tuned.get("tuned_weights", {})
                summary_data["weight_tuner"] = {
                    "samples": tw.get("samples"),
                    "correlation": tw.get("correlation"),
                    "active": tw.get("active"),
                    "weights": tw.get("weights"),
                }
            except Exception:
                logger.exception("Weight tuner run failed.")

            try:
                from src.learning.hour_of_day import update_hour_stats
                hs_state = update_hour_stats(reason="weekly_postmortem")
                summary_data["hour_stats"] = hs_state.get("hour_stats", {})
            except Exception:
                logger.exception("Hour-of-day stats update failed.")

            try:
                from src.learning.adaptive_thresholds import calibrate_from_counterfactuals
                cal = calibrate_from_counterfactuals(reason="weekly_postmortem")
                summary_data["counterfactual_calibration"] = cal
            except Exception:
                logger.exception("Counterfactual calibration failed.")

            try:
                from src.learning.ml_model import train as train_ml
                ml = train_ml()
                summary_data["ml_model"] = {
                    "samples": ml.get("samples"),
                    "active": ml.get("active"),
                    "train_acc": ml.get("train_acc"),
                }
            except Exception:
                logger.exception("ML model training failed.")

        return summary_data

    async def _generate_insights(self, summary: dict) -> str:
        prompt = f"""Analyze this trading performance summary and provide 2-3 actionable insights.
Focus on what's working, what's not, and specific adjustments to consider.

Performance Data:
{json.dumps(summary, indent=2)}

Respond with 2-3 bullet points. Be specific and concise. No fluff."""

        result = self.orchestrator.call_model("reviewer", prompt)
        if result:
            return result.content
        return "Insufficient data for automated insights."
