import logging
import os
from datetime import datetime

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from src.db.database import get_session
from src.db.models import Alert, Feedback, Outcome, SourceScore
from src.delivery.formatter import format_budget, format_sources, format_status

logger = logging.getLogger(__name__)


class TalonTelegramBot:
    """Telegram bot for sending alerts and handling user commands."""

    def __init__(self, app_state: dict):
        """app_state is a shared dict providing access to system components."""
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        self.app_state = app_state
        self.application: Application | None = None

    def build_application(self) -> Application:
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not set")

        self.application = (
            Application.builder()
            .token(self.token)
            .build()
        )

        handlers = [
            CommandHandler("status", self._cmd_status),
            CommandHandler("health", self._cmd_health),
            CommandHandler("feedback", self._cmd_feedback),
            CommandHandler("note", self._cmd_note),
            CommandHandler("review", self._cmd_review),
            CommandHandler("weekly", self._cmd_weekly),
            CommandHandler("sources", self._cmd_sources),
            CommandHandler("pause", self._cmd_pause),
            CommandHandler("resume", self._cmd_resume),
            CommandHandler("risk", self._cmd_risk),
            CommandHandler("mode", self._cmd_mode),
            CommandHandler("budget", self._cmd_budget),
            CommandHandler("last", self._cmd_last),
            CommandHandler("took", self._cmd_took),
            CommandHandler("positions", self._cmd_positions),
            CommandHandler("exit", self._cmd_exit),
            CommandHandler("watchlist", self._cmd_watchlist),
            CommandHandler("analyze", self._cmd_analyze),
        ]
        for h in handlers:
            self.application.add_handler(h)

        return self.application

    async def send_alert(self, message: str, alert_id: int | None = None) -> int | None:
        """Send a message to the configured chat. Returns the Telegram message ID."""
        if not self.application or not self.chat_id:
            logger.warning("Cannot send alert: bot not initialized or no chat_id.")
            return None

        try:
            msg = await self.application.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=None,
            )
            return msg.message_id
        except Exception:
            logger.exception("Failed to send Telegram message")
            return None

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        mode = self.app_state.get("mode", "shadow")
        alerts_today = self._count_alerts_today()
        router = self.app_state.get("model_router")
        budget = router.budget_status() if router else {"spent_usd": 0, "budget_usd": 2, "remaining_usd": 2, "pct_used": 0}
        paused = self.app_state.get("paused", False)
        await update.message.reply_text(format_status(mode, alerts_today, budget, paused))

    async def _cmd_health(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comprehensive health snapshot — confirms the process is alive
        and reports the most recent activity timestamps from each subsystem."""
        from sqlalchemy import func
        from src.db.models import CostLog, Outcome, Scan
        mode = self.app_state.get("mode", "shadow")
        paused = self.app_state.get("paused", False)
        router = self.app_state.get("model_router")
        budget = router.budget_status() if router else {}
        with get_session() as session:
            last_scan = session.query(func.max(Scan.completed_at)).scalar() or "never"
            last_alert = session.query(func.max(Alert.sent_at)).scalar() or "never"
            last_outcome = session.query(func.max(Outcome.resolved_at)).scalar() or "never"
            last_llm_call = session.query(func.max(CostLog.logged_at)).scalar() or "never"
        try:
            from src.learning.ml_model import status as ml_status
            ms = ml_status()
            if ms.get("trained"):
                ml_line = (
                    f"ML model: {'ACTIVE' if ms['active'] else 'inactive'} "
                    f"(n={ms['samples']}, acc={ms.get('train_acc')})"
                )
            else:
                ml_line = "ML model: not trained yet"
        except Exception:
            ml_line = "ML model: n/a"

        text = (
            f"Talon Health\n"
            f"Mode: {mode.upper()} | Paused: {paused}\n"
            f"Alerts today: {self._count_alerts_today()}\n"
            f"Budget: ${budget.get('spent_usd', 0):.2f}/${budget.get('budget_usd', 0):.2f} "
            f"({budget.get('pct_used', 0):.0f}%)\n"
            f"Last scan: {last_scan}\n"
            f"Last alert: {last_alert}\n"
            f"Last outcome: {last_outcome}\n"
            f"Last LLM call: {last_llm_call}\n"
            f"{ml_line}"
        )
        await update.message.reply_text(text)

    async def _cmd_feedback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        args = context.args or []
        if len(args) < 2:
            await update.message.reply_text("Usage: /feedback <alert_id> <good|bad|late|skip>")
            return

        try:
            alert_id = int(args[0])
        except ValueError:
            await update.message.reply_text("Invalid alert ID.")
            return

        fb_type = args[1].lower()
        valid_types = {"good", "bad", "too_late", "late", "wrong_direction", "good_idea_bad_timing", "skip"}
        if fb_type not in valid_types:
            await update.message.reply_text(f"Valid feedback types: {', '.join(sorted(valid_types))}")
            return

        with get_session() as session:
            alert = session.query(Alert).filter(Alert.id == alert_id).first()
            if not alert:
                await update.message.reply_text(f"Alert #{alert_id} not found.")
                return
            fb = Feedback(
                alert_id=alert_id,
                feedback_type=fb_type,
                received_at=datetime.utcnow().isoformat(),
            )
            session.add(fb)
            session.commit()

        try:
            from src.learning.source_evaluator import SourceEvaluator
            evaluator = SourceEvaluator()
            evaluator.apply_feedback_multiplier(alert_id, fb_type)
        except Exception:
            pass

        await update.message.reply_text(f"Feedback '{fb_type}' recorded for alert #{alert_id}.")

    async def _cmd_note(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        args = context.args or []
        if len(args) < 2:
            await update.message.reply_text("Usage: /note <alert_id> <text>")
            return

        try:
            alert_id = int(args[0])
        except ValueError:
            await update.message.reply_text("Invalid alert ID.")
            return

        note_text = " ".join(args[1:])
        with get_session() as session:
            fb = Feedback(
                alert_id=alert_id,
                feedback_type="note",
                user_note=note_text,
                received_at=datetime.utcnow().isoformat(),
            )
            session.add(fb)
            session.commit()

        await update.message.reply_text(f"Note added to alert #{alert_id}.")

    async def _cmd_review(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        from zoneinfo import ZoneInfo
        ny_now = datetime.now(ZoneInfo("America/New_York"))
        ny_midnight = ny_now.replace(hour=0, minute=0, second=0, microsecond=0)
        utc_cutoff = ny_midnight.astimezone(ZoneInfo("UTC")).replace(tzinfo=None).isoformat()
        today_label = ny_now.strftime("%Y-%m-%d")
        from src.db.models import CounterfactualOutcome
        with get_session() as session:
            alerts = session.query(Alert).filter(Alert.sent_at >= utc_cutoff).all()
            outcomes = (
                session.query(Outcome)
                .join(Alert)
                .filter(Alert.sent_at >= utc_cutoff)
                .all()
            )
            cfs = (
                session.query(CounterfactualOutcome)
                .filter(CounterfactualOutcome.created_at >= utc_cutoff)
                .filter(CounterfactualOutcome.resolved_at.isnot(None))
                .filter(CounterfactualOutcome.pnl_pct.isnot(None))
                .all()
            )

        wins = sum(1 for o in outcomes if o.pnl_pct and o.pnl_pct > 0)
        losses = sum(1 for o in outcomes if o.pnl_pct and o.pnl_pct < 0)
        total_pnl = sum(o.pnl_pct or 0 for o in outcomes)
        cf_wins = sum(1 for c in cfs if (c.pnl_pct or 0) > 0)

        text = (
            f"Today's Review ({today_label} ET):\n"
            f"Alerts sent: {len(alerts)}\n"
            f"Resolved: {len(outcomes)} (W:{wins} L:{losses})\n"
            f"Total P&L: {total_pnl:+.1f}%"
        )
        if cfs:
            text += (
                f"\nFiltered (not alerted): {len(cfs)} resolved, "
                f"{cf_wins} would-be wins"
            )
        await update.message.reply_text(text)

    async def _cmd_weekly(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        from src.db.models import PerformanceReview
        with get_session() as session:
            review = (
                session.query(PerformanceReview)
                .filter(PerformanceReview.period_type == "weekly")
                .order_by(PerformanceReview.created_at.desc())
                .first()
            )
        if not review:
            await update.message.reply_text("No weekly review available yet.")
            return

        text = (
            f"Weekly Review ({review.period_start} to {review.period_end}):\n"
            f"Alerts: {review.total_alerts} (W:{review.winning_alerts} L:{review.losing_alerts})\n"
            f"Avg P&L: {review.avg_pnl_pct:+.1f}%\n"
        )
        if review.lessons:
            text += f"Insights: {review.lessons[:200]}"
        await update.message.reply_text(text)

    async def _cmd_sources(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        with get_session() as session:
            rows = session.query(SourceScore).order_by(SourceScore.reliability_score.desc()).all()
        source_data = [
            {
                "name": r.source_subtype or r.source_name,
                "score": r.reliability_score,
                "total": r.total_signals,
                "enabled": r.enabled if r.enabled is not None else True,
                "disabled_reason": r.disabled_reason,
            }
            for r in rows
        ]
        await update.message.reply_text(format_sources(source_data))

    async def _cmd_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        import asyncio
        args = context.args or []
        if not args:
            await update.message.reply_text("Usage: /analyze <TICKER> [deep]")
            return
        ticker = args[0].upper().strip()
        deep = len(args) > 1 and args[1].lower() in ("deep", "full", "vision")
        analyzer = self.app_state.get("analyzer")
        if analyzer is None:
            await update.message.reply_text("Analyzer not available.")
            return
        await update.message.reply_text(f"Analyzing {ticker}{' (deep)' if deep else ''}...")
        try:
            from src.analysis.on_demand import format_analysis
            res = await asyncio.to_thread(analyzer.analyze, ticker, deep)
            await update.message.reply_text(format_analysis(res))
        except Exception as e:
            await update.message.reply_text(f"Analysis failed: {type(e).__name__}: {e}")

    async def _cmd_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        from src.risk.breakers import manual_reset, status_text
        config = self.app_state.get("config", {})
        args = context.args or []
        if args and args[0].lower() == "reset":
            manual_reset()
            self.app_state.pop("_breaker_notified", None)
            await update.message.reply_text("Circuit breaker reset. " + status_text(config))
            return
        await update.message.reply_text(status_text(config))

    async def _cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.app_state["paused"] = True
        await update.message.reply_text("Scanning paused.")

    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.app_state["paused"] = False
        await update.message.reply_text("Scanning resumed.")

    async def _cmd_mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        args = context.args or []
        if not args:
            await update.message.reply_text(f"Current mode: {self.app_state.get('mode', 'shadow')}")
            return

        new_mode = args[0].lower()
        if new_mode not in ("shadow", "paper", "live"):
            await update.message.reply_text("Valid modes: shadow, paper, live")
            return

        if new_mode == "live":
            pending = self.app_state.get("pending_live_confirm")
            now = datetime.utcnow()
            if pending and (now - pending).total_seconds() <= 60:
                self.app_state["mode"] = "live"
                self.app_state.pop("pending_live_confirm", None)
                await update.message.reply_text("Mode set to: LIVE. Real-money alerts enabled.")
                return
            self.app_state["pending_live_confirm"] = now
            await update.message.reply_text(
                "WARNING: Live mode places real trades. "
                "Send /mode live again within 60s to confirm."
            )
            return

        self.app_state["mode"] = new_mode
        self.app_state.pop("pending_live_confirm", None)
        await update.message.reply_text(f"Mode set to: {new_mode.upper()}")

    async def _cmd_budget(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        router = self.app_state.get("model_router")
        if not router:
            await update.message.reply_text("Budget tracking not initialized.")
            return
        await update.message.reply_text(format_budget(router.budget_status()))

    async def _cmd_last(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        with get_session() as session:
            alert = session.query(Alert).order_by(Alert.id.desc()).first()
        if not alert:
            await update.message.reply_text("No alerts sent yet.")
            return

        text = (
            f"Last Alert (#{alert.id}):\n"
            f"{alert.contract} — {alert.direction}\n"
            f"Entry: ${alert.entry_price_low or 0:.2f}–${alert.entry_price_high or 0:.2f}\n"
            f"Target: ${alert.target_price or 0:.2f} | Stop: ${alert.stop_price or 0:.2f}\n"
            f"Sent: {alert.sent_at}\n"
            f"Mode: {alert.mode.upper()}"
        )
        await update.message.reply_text(text)

    async def _cmd_watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        config = self.app_state.get("config", {})
        watchlist = config.get("watchlist", {})
        t1 = watchlist.get("tier1", [])
        t2 = watchlist.get("tier2", [])
        text = f"Tier 1 (every cycle): {', '.join(t1)}\nTier 2 (alternating): {', '.join(t2)}"
        await update.message.reply_text(text)

    async def _cmd_took(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        from src.learning.position_monitor import add_position
        args = context.args or []
        if not args:
            await update.message.reply_text("Usage: /took <alert_id>")
            return
        try:
            alert_id = int(args[0])
        except ValueError:
            await update.message.reply_text("Invalid alert ID.")
            return
        pos = add_position(alert_id)
        if pos:
            await update.message.reply_text(
                f"Position tracked: {pos['contract']} [#{alert_id}]\n"
                f"Entry: ${pos['entry_price']:.2f}\n"
                f"Target: ${pos['target_price']:.2f} | Stop: ${pos['stop_price']:.2f}\n"
                f"Monitoring every 15 min. I'll notify you on updates."
            )
        else:
            await update.message.reply_text(f"Alert #{alert_id} not found or already tracked.")

    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        from src.learning.position_monitor import get_active_positions
        positions = get_active_positions()
        if not positions:
            await update.message.reply_text("No active positions.")
            return
        lines = ["Active positions:"]
        for p in positions:
            entry = p["entry_price"]
            last = p.get("last_price", entry)
            pnl = ((last / entry) - 1) * 100 if entry > 0 else 0
            lines.append(
                f"  [#{p['alert_id']}] {p['contract']}\n"
                f"    Entry: ${entry:.2f} | Last: ${last:.2f} ({pnl:+.1f}%)\n"
                f"    Target: ${p['target_price']:.2f} | Stop: ${p['stop_price']:.2f}"
            )
        await update.message.reply_text("\n".join(lines))

    async def _cmd_exit(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        from src.learning.position_monitor import close_position
        args = context.args or []
        if not args:
            await update.message.reply_text("Usage: /exit <alert_id> [exit_price]")
            return
        try:
            alert_id = int(args[0])
        except ValueError:
            await update.message.reply_text("Invalid alert ID.")
            return
        exit_price = None
        if len(args) > 1:
            try:
                exit_price = float(args[1])
            except ValueError:
                pass
        result = close_position(alert_id, exit_price)
        if result:
            await update.message.reply_text(
                f"Position closed: {result['contract']} [#{alert_id}]\n"
                f"Entry: ${result['entry_price']:.2f} | Exit: ${result['exit_price']:.2f}\n"
                f"P&L: {result.get('pnl_pct', 0):+.1f}%"
            )
        else:
            await update.message.reply_text(f"No active position found for alert #{alert_id}.")

    def _count_alerts_today(self) -> int:
        from zoneinfo import ZoneInfo
        ny_now = datetime.now(ZoneInfo("America/New_York"))
        ny_midnight = ny_now.replace(hour=0, minute=0, second=0, microsecond=0)
        utc_cutoff = ny_midnight.astimezone(ZoneInfo("UTC")).replace(tzinfo=None).isoformat()
        with get_session() as session:
            return (
                session.query(Alert)
                .filter(Alert.sent_at >= utc_cutoff)
                .count()
            )
