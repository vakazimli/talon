"""Talon — Entry point and scheduler.

Ties together all modules: scanner, analysis, delivery, cost control, and learning.
Runs the scan cycle at configured times using APScheduler.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

from src.analysis.scoring_engine import ScoringEngine
from src.analysis.setup_detector import SetupDetector
from src.analysis.synthesizer import Synthesizer
from src.cost.model_router import ModelRouter
from src.cost.rate_limiter import RateLimiter
from src.cost.token_tracker import TokenTracker
from src.data.cache import ScanCache
from src.db.database import get_session, init_database
from src.db.models import Alert, Scan, Setup
from src.delivery.formatter import format_trade_card
from src.delivery.telegram_bot import TalonTelegramBot
from src.obs.trace import stage
from src.orchestrator import TalonOrchestrator
from src.scanner.market_scanner import MarketScanner
from src.scanner.news_scanner import NewsScanner, extract_signals as extract_news_signals
from src.scanner.options_scanner import OptionsScanner

logger = logging.getLogger("talon")


def load_config() -> dict:
    config_dir = Path(__file__).resolve().parent.parent / "config"
    config = {}
    for name in ("settings", "model_routing", "sources"):
        path = config_dir / f"{name}.yaml"
        if path.exists():
            with open(path) as f:
                config.update(yaml.safe_load(f) or {})
    return config


class Talon:
    """Main application class. Owns all components and runs the scan cycle."""

    def __init__(self):
        load_dotenv()
        self.config = load_config()
        self.mode = os.environ.get("TALON_MODE", self.config.get("system", {}).get("mode", "shadow"))

        log_level = os.environ.get("TALON_LOG_LEVEL", "INFO")
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        from logging.handlers import RotatingFileHandler
        rotating = RotatingFileHandler(
            log_dir / "talon.log", maxBytes=10_000_000, backupCount=3
        )
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            handlers=[
                logging.StreamHandler(),
                rotating,
            ],
        )
        # Quiet the libraries that spam INFO every 10s on Telegram polling.
        for noisy in ("httpx", "httpcore", "apscheduler.scheduler",
                      "apscheduler.executors.default", "telegram.ext"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        self.db_session_factory = init_database()
        try:
            from src.learning.source_evaluator import seed_source_scores
            seed_source_scores()
        except Exception:
            logger.exception("Source-score seeding failed (non-fatal).")
        self.token_tracker = TokenTracker()
        self.rate_limiter = RateLimiter()

        model_config = self.config.get("models", {})
        cost_controls = self.config.get("cost_controls", {})
        self.model_router = ModelRouter(model_config, cost_controls, self.token_tracker)

        self.orchestrator = TalonOrchestrator(
            model_config, cost_controls, self.token_tracker, self.rate_limiter,
            router=self.model_router,
        )

        self.market_scanner = MarketScanner(self.config)
        self.options_scanner = OptionsScanner(self.config)
        self.news_scanner = NewsScanner(self.config)
        self.setup_detector = SetupDetector(self.orchestrator)
        self.scoring_engine = ScoringEngine(self.config)
        self.synthesizer = Synthesizer(self.orchestrator, self.config)
        from src.analysis.critic import RiskCritic
        self.critic = RiskCritic(self.orchestrator, self.config)
        from src.analysis.vision_analyst import VisionAnalyst
        self.vision_analyst = VisionAnalyst(self.orchestrator, self.config)
        self.cache = ScanCache(ttl_seconds=300)

        # On-demand analyzer (/analyze and the skill). Uses its own Synthesizer
        # so its call counter never interferes with scheduled scan cycles.
        from src.analysis.on_demand import OnDemandAnalyzer
        self.on_demand = OnDemandAnalyzer(
            self.config,
            self.market_scanner,
            self.options_scanner,
            self.news_scanner,
            self.setup_detector,
            self.scoring_engine,
            Synthesizer(self.orchestrator, self.config),
            self.vision_analyst,
            chart_dataframes_fn=self._chart_dataframes,
        )

        self.app_state = {
            "mode": self.mode,
            "paused": False,
            "config": self.config,
            "model_router": self.model_router,
            "analyzer": self.on_demand,
        }

        self.bot = TalonTelegramBot(self.app_state)
        self._scan_cycle_count = 0

        self.alerts_config = self.config.get("alerts", {})
        self._last_alert_time: dict[str, datetime] = {}
        self._hydrate_cooldowns()

    def _hydrate_cooldowns(self) -> None:
        """Populate per-ticker cooldown timestamps from the alerts table.

        Without this, restarting the process clears the in-memory cooldown
        and the next scan can re-alert a ticker we just alerted minutes ago.
        """
        try:
            from sqlalchemy import func
            with get_session() as session:
                rows = (
                    session.query(Alert.ticker, func.max(Alert.sent_at))
                    .group_by(Alert.ticker)
                    .all()
                )
            for ticker, sent_at in rows:
                if not ticker or not sent_at:
                    continue
                try:
                    self._last_alert_time[ticker] = datetime.fromisoformat(sent_at)
                except (ValueError, TypeError):
                    continue
            if self._last_alert_time:
                logger.info(
                    "Hydrated cooldown state for %d tickers from DB.",
                    len(self._last_alert_time),
                )
        except Exception:
            logger.exception("Failed to hydrate cooldowns from DB; continuing empty.")

    def _safe_job(self, name: str, coro_factory):
        """Wrap a scheduled coroutine so an exception reports to Telegram
        instead of silently dropping the cycle."""
        async def runner(*args, **kwargs):
            try:
                await coro_factory(*args, **kwargs)
            except Exception as e:
                logger.exception("Scheduled job %s failed", name)
                try:
                    await self.bot.send_alert(
                        f"[Talon error] {name}: {type(e).__name__}: {e}"
                    )
                except Exception:
                    pass
        return runner

    async def run(self):
        logger.info("Talon starting in %s mode", self.mode.upper())

        # coalesce + a generous misfire grace window so a scan that was due
        # while the Mac was asleep still runs on wake instead of being
        # silently dropped (the old behavior that meant a sleeping laptop
        # produced zero scans all day). max_instances=1 prevents overlap.
        scheduler = AsyncIOScheduler(
            timezone="America/New_York",
            job_defaults={
                "coalesce": True,
                "misfire_grace_time": 3600,
                "max_instances": 1,
            },
        )

        scanning = self.config.get("scanning", {})

        pre_market = scanning.get("pre_market_scan", "09:00")
        h, m = pre_market.split(":")
        scheduler.add_job(
            self._safe_job("pre_market_scan", self._run_scan_cycle),
            CronTrigger(hour=int(h), minute=int(m), day_of_week="mon-fri"),
            args=["pre_market"], id="pre_market", replace_existing=True,
        )

        for scan_time in scanning.get("intraday_scans", []):
            h, m = scan_time.split(":")
            scheduler.add_job(
                self._safe_job(f"intraday_{scan_time}", self._run_scan_cycle),
                CronTrigger(hour=int(h), minute=int(m), day_of_week="mon-fri"),
                args=["scheduled"], id=f"intraday_{scan_time}", replace_existing=True,
            )

        post_market = scanning.get("post_market_review", "16:15")
        h, m = post_market.split(":")
        scheduler.add_job(
            self._safe_job("post_market_review", self._run_post_market_review),
            CronTrigger(hour=int(h), minute=int(m), day_of_week="mon-fri"),
            id="post_market", replace_existing=True,
        )

        scheduler.add_job(
            self._safe_job("weekly_postmortem", self._run_weekly_postmortem),
            CronTrigger(day_of_week="sun", hour=18, minute=0),
            id="weekly_postmortem", replace_existing=True,
        )

        scheduler.add_job(
            self._safe_job("position_monitor", self._run_position_check),
            CronTrigger(minute="*/15", hour="9-16", day_of_week="mon-fri"),
            id="position_monitor", replace_existing=True,
        )

        # Resolve open outcomes intraday (every 15 min) so target/stop hits
        # and MFE/MAE are captured live, not from a single stale 16:15
        # snapshot. Critical for short-hold trades whose option can hit
        # target midday and fade by the close.
        scheduler.add_job(
            self._safe_job("outcome_poll", self._run_outcome_poll),
            CronTrigger(minute="2-59/15", hour="9-16", day_of_week="mon-fri"),
            id="outcome_poll", replace_existing=True,
        )

        scheduler.add_job(
            self._safe_job("wal_checkpoint", self._run_wal_checkpoint),
            CronTrigger(hour=4, minute=0),
            id="wal_checkpoint", replace_existing=True,
        )

        scheduler.add_job(
            self._safe_job("daily_heartbeat", self._run_daily_heartbeat),
            CronTrigger(hour=8, minute=45, day_of_week="mon-fri"),
            id="daily_heartbeat", replace_existing=True,
        )

        scheduler.start()
        logger.info("Scheduler started with %d jobs", len(scheduler.get_jobs()))

        if self.bot.token:
            app = self.bot.build_application()
            await app.initialize()
            await app.start()
            await app.updater.start_polling(drop_pending_updates=True)
            logger.info("Telegram bot started")
            # Boot heartbeat — confirms the process is alive after a restart.
            try:
                await self.bot.send_alert(
                    f"[Talon] Online. Mode={self.mode.upper()} | "
                    f"Watchlist tier1={len(self.config.get('watchlist', {}).get('tier1', []))} "
                    f"tier2={len(self.config.get('watchlist', {}).get('tier2', []))}"
                )
            except Exception:
                logger.exception("Boot heartbeat send failed.")
        else:
            logger.warning("No TELEGRAM_BOT_TOKEN set. Bot commands disabled.")

        # Catch-up scan: if we (re)started during market hours, run one scan
        # now instead of idling until the next cron slot. This is what makes a
        # restart after a crash/sleep immediately useful.
        if self._is_market_hours():
            logger.info("Started during market hours — running a catch-up scan.")
            asyncio.create_task(
                self._safe_job("boot_catch_up", self._run_scan_cycle)("boot")
            )

        try:
            while True:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutting down...")
            scheduler.shutdown()
            if self.bot.application:
                await self.bot.application.updater.stop()
                await self.bot.application.stop()
                await self.bot.application.shutdown()

    @staticmethod
    def _is_market_hours() -> bool:
        """True on weekdays between 09:30 and 16:00 America/New_York."""
        from datetime import time as dtime
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("America/New_York"))
        if now.weekday() >= 5:  # Sat/Sun
            return False
        return dtime(9, 30) <= now.time() <= dtime(16, 0)

    async def _run_scan_cycle(self, scan_type: str = "scheduled"):
        if self.app_state.get("paused"):
            logger.info("Scanning paused. Skipping %s scan.", scan_type)
            return

        if self.model_router.budget_pct_used >= self.model_router.hard_stop_pct:
            logger.warning("Budget >%.0f%% used. Skipping scan.", self.model_router.hard_stop_pct * 100)
            return

        self._scan_cycle_count += 1
        self.cache.clear()
        self.synthesizer.reset_cycle()
        spend_at_start = self.token_tracker.get_daily_spend()

        scan = Scan(
            scan_type=scan_type,
            started_at=datetime.utcnow().isoformat(),
        )

        mode = self.app_state.get("mode", "shadow")
        watchlist = self.config.get("watchlist", {})
        tickers = list(watchlist.get("tier1", []))
        if self._scan_cycle_count % 2 == 0:
            tickers.extend(watchlist.get("tier2", []))

        logger.info("Scan cycle #%d: %s — %d tickers", self._scan_cycle_count, scan_type, len(tickers))

        # 1. Market scan (blocking yfinance/network — offload to thread)
        with stage("market_scan", tickers=len(tickers)) as sp:
            scan_results = await asyncio.to_thread(self.market_scanner.scan_tickers, tickers)
            sp["with_signals"] = sum(1 for r in scan_results if r.signals)
        interesting = [r for r in scan_results if r.signals]
        logger.info("Market scan: %d/%d tickers have signals", len(interesting), len(scan_results))

        # 2. Filter cooldowns
        interesting = self._apply_cooldowns(interesting)

        # 3. Fetch news for interesting tickers (offloaded; cached per cycle).
        news_by_ticker: dict[str, list] = {}
        if self.news_scanner.enabled:
            for r in interesting:
                items = await asyncio.to_thread(self.news_scanner.fetch_news, r.ticker)
                news_by_ticker[r.ticker] = items

        # 4. Setup detection via LLM (blocking httpx — offload to thread).
        # Detector receives news context so it can factor catalysts into
        # the tradeable / not-tradeable decision.
        with stage("setup_detection", candidates=len(interesting)) as sp:
            detected = await asyncio.to_thread(
                self.setup_detector.detect_setups, interesting, news_by_ticker
            )
            sp["detected"] = len(detected)
        logger.info("Setup detection: %d setups found", len(detected))

        # 5. Earnings-window gate: skip alerts within 2 trading days of earnings.
        if self.news_scanner.enabled:
            filtered = []
            for sd in detected:
                if await asyncio.to_thread(
                    self.news_scanner.has_earnings_within, sd["ticker"], 2
                ):
                    logger.info(
                        "Skipping %s: earnings within 2 days (IV-crush risk).",
                        sd["ticker"],
                    )
                    continue
                filtered.append(sd)
            detected = filtered

        # 6. Options scan for detected setups
        for setup_data in detected:
            ticker = setup_data["ticker"]
            direction = setup_data["direction"]
            cache_key = f"options:{ticker}:{direction}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                setup_data["option_candidates"] = cached
            else:
                candidates = await asyncio.to_thread(
                    self.options_scanner.scan_options, ticker, direction
                )
                self.cache.set(cache_key, candidates)
                setup_data["option_candidates"] = candidates

        # 6b. IV richness filter — drop setups where option premium is
        #     extremely elevated vs underlying HV20 (theta will eat us).
        from src.analysis.iv_filter import evaluate_iv, hv20_from_returns
        survivors = []
        for setup_data in detected:
            scan_result = setup_data.get("scan_result")
            candidates = setup_data.get("option_candidates") or []
            if not candidates or not scan_result:
                survivors.append(setup_data)
                continue
            best_iv = float(getattr(candidates[0], "implied_volatility", 0) or 0)
            try:
                import yfinance as yf
                hist = await asyncio.to_thread(
                    lambda t=setup_data["ticker"]: yf.Ticker(t).history(period="2mo", interval="1d")
                )
                hv = hv20_from_returns(hist["Close"]) if hist is not None and not hist.empty else 0
            except Exception:
                hv = 0
            check = evaluate_iv(best_iv, hv)
            if check.rich:
                logger.info(
                    "Skipping %s: IV/HV ratio %.2f elevated (IV=%.2f HV20=%.2f).",
                    setup_data["ticker"], check.ratio, check.iv, check.hv20,
                )
                continue
            if check.elevated:
                setup_data["iv_penalty"] = True
                setup_data["iv_check"] = check
                logger.info(
                    "%s: IV elevated (ratio=%.2f), penalizing score.",
                    setup_data["ticker"], check.ratio,
                )
            survivors.append(setup_data)
        detected = survivors

        # 7. Score setups (now including news catalyst signals)
        scored_setups = []
        for setup_data in detected:
            scan_result = setup_data.get("scan_result")
            news_items = news_by_ticker.get(setup_data["ticker"], [])
            news_signals = extract_news_signals(news_items) if news_items else []
            setup_data["news_items"] = news_items
            setup_data["news_signals"] = news_signals
            scored = self.scoring_engine.score_setup(
                ticker=setup_data["ticker"],
                direction=setup_data["direction"],
                signals=scan_result.signals if scan_result else [],
                technicals=scan_result.technicals if scan_result else {},
                option_candidates=setup_data.get("option_candidates", []),
                news_signals=news_signals,
            )
            if setup_data.get("iv_penalty"):
                from src.analysis.iv_filter import PENALTY_POINTS
                scored.score = max(0.0, scored.score - PENALTY_POINTS)
                scored.score_breakdown["iv_penalty"] = -PENALTY_POINTS
            scored_setups.append(scored)

        # 6. Log all setups to DB (short session, no awaits)
        setup_id_map = {}
        all_setup_ids = {}
        # Reasons a non-alerted setup was filtered (for counterfactual tagging).
        filter_reasons: dict[str, str] = {}
        with get_session() as session:
            session.add(scan)
            session.flush()
            # Capture the PK as a plain int while the instance is still bound
            # to this session. After the session closes, `scan` is detached and
            # any `scan.id` access raises DetachedInstanceError.
            scan_id = scan.id

            for scored in scored_setups:
                promoted = self.scoring_engine.passes_threshold(scored.score, mode)
                db_setup = Setup(
                    scan_id=scan_id,
                    ticker=scored.ticker,
                    direction=scored.direction,
                    setup_type=scored.setup_type,
                    timeframe=scored.timeframe,
                    detected_at=datetime.utcnow().isoformat(),
                    score=scored.score,
                    score_breakdown=json.dumps(scored.score_breakdown),
                    raw_signals=json.dumps(scored.raw_signals),
                    sources_used=json.dumps(scored.sources_used),
                    promoted_to_alert=promoted,
                )
                session.add(db_setup)
                session.flush()
                key = scored.ticker + scored.direction
                all_setup_ids[key] = db_setup.id
                if promoted:
                    setup_id_map[key] = db_setup.id
                else:
                    filter_reasons[key] = "below_floor"

            scan.tickers_scanned = json.dumps(tickers)
            scan.setups_found = len(scored_setups)
            session.commit()

        # 6c. Risk circuit breakers — if tripped, hold alerts for this cycle
        #     (setups are still scored + logged above for learning).
        breaker = None
        try:
            from src.risk.breakers import check_and_update
            breaker = check_and_update(self.config)
        except Exception:
            logger.exception("Circuit-breaker check failed; proceeding without it.")
        if breaker is not None and breaker.halted:
            logger.warning("Alerts held by circuit breaker: %s", breaker.reason)
            await self._notify_breaker_once(breaker.reason)
            for s in scored_setups:
                filter_reasons[s.ticker + s.direction] = "breaker"
            self._record_counterfactuals(scored_setups, set(), all_setup_ids, filter_reasons)
            with get_session() as session:
                db_scan = session.query(Scan).filter(Scan.id == scan_id).first()
                if db_scan:
                    db_scan.alerts_sent = 0
                    db_scan.completed_at = datetime.utcnow().isoformat()
                session.commit()
            logger.info("Scan cycle #%d complete: %d setups, alerts HELD (breaker).",
                        self._scan_cycle_count, len(scored_setups))
            return
        else:
            self.app_state.pop("_breaker_notified", None)

        # 7. Synthesize and send alerts (NO open DB session during awaits)
        alerts_sent = 0
        max_alerts = self.alerts_config.get("max_alerts_per_day", 5)
        alerts_today = self._count_alerts_today()
        alerted_keys: set[str] = set()

        for scored in scored_setups:
            key = scored.ticker + scored.direction
            setup_db_id = setup_id_map.get(key)
            if not setup_db_id:
                continue

            if alerts_today + alerts_sent >= max_alerts:
                logger.info("Max alerts per day reached (%d). Stopping.", max_alerts)
                break

            if not scored.option_candidates:
                logger.info("Skipping %s: no real options data.", scored.ticker)
                filter_reasons[key] = "no_candidates"
                continue

            # Optional: vision-based chart analyst (opt-in via config).
            # Runs only on setups that already passed scoring, so cost is bounded.
            vision = None
            if self.vision_analyst.enabled:
                df_for_chart, df_daily = await asyncio.to_thread(
                    self._chart_dataframes, scored.ticker
                )
                if df_for_chart is not None and not df_for_chart.empty:
                    vision = await asyncio.to_thread(
                        self.vision_analyst.analyze,
                        scored.ticker, scored.direction, scored.setup_type,
                        df_for_chart, scored.technicals,
                        df_daily,
                    )
                if vision is not None:
                    scored.score_breakdown["chart_analyst"] = vision.chart_score
                    chart_cfg = self.config.get("chart_analyst", {})
                    if chart_cfg.get("pass_verdict_blocks", True) and vision.verdict == "PASS":
                        logger.info(
                            "Vision analyst: %s verdict=PASS (%s); skipping.",
                            scored.ticker, vision.pattern,
                        )
                        filter_reasons[key] = "vision_pass"
                        continue
                    if vision.chart_score < chart_cfg.get("min_chart_score", 50):
                        logger.info(
                            "Vision analyst: %s chart_score=%d below threshold; skipping.",
                            scored.ticker, vision.chart_score,
                        )
                        filter_reasons[key] = "low_chart_score"
                        continue

            card = self.synthesizer.synthesize(
                scored,
                news_items=news_by_ticker.get(scored.ticker, []),
            )
            if card is None:
                filter_reasons[key] = "no_synth"
                continue

            # Bear-case critic: adversarial review before alerting.
            if self.critic.enabled:
                verdict = await asyncio.to_thread(
                    self.critic.critique, scored, card,
                    news_by_ticker.get(scored.ticker, []),
                )
                if verdict.verdict == "VETO":
                    logger.info("Critic VETO %s: %s", card.ticker, verdict.key_risk)
                    filter_reasons[key] = "critic_veto"
                    continue
                if verdict.verdict == "CAUTION":
                    card.confidence = max(1, card.confidence - 1)
                    logger.info("Critic CAUTION %s: %s (confidence -> %d)",
                                card.ticker, verdict.key_risk, card.confidence)

            # High-conviction gate: only deliver cards the synthesizer rates
            # at or above the configured confidence floor (1-5 scale).
            min_conf = self.alerts_config.get("min_confidence", 4)
            if card.confidence < min_conf:
                logger.info(
                    "Skipping %s: confidence %d < min_confidence %d.",
                    card.ticker, card.confidence, min_conf,
                )
                filter_reasons[key] = "low_confidence"
                continue

            # Write alert to DB in a short session
            with get_session() as session:
                alert = Alert(
                    setup_id=setup_db_id,
                    ticker=card.ticker,
                    direction=card.direction,
                    contract=card.contract,
                    entry_price_low=card.entry_low,
                    entry_price_high=card.entry_high,
                    target_price=card.target,
                    stop_price=card.stop,
                    confidence=card.confidence,
                    rationale=card.rationale,
                    sent_at=datetime.utcnow().isoformat(),
                    mode=mode,
                )
                session.add(alert)
                session.commit()
                alert_id = alert.id

            # Send Telegram (no DB session open)
            message = format_trade_card(card, mode, alert_id=alert_id)
            telegram_msg_id = await self.bot.send_alert(message)

            if telegram_msg_id is None:
                logger.warning("Telegram send failed for %s.", card.ticker)
                continue

            # Update telegram_message_id in a short session
            with get_session() as session:
                db_alert = session.query(Alert).filter(Alert.id == alert_id).first()
                if db_alert:
                    db_alert.telegram_message_id = telegram_msg_id
                session.commit()

            # Record entry through OutcomeTracker (single source of truth).
            try:
                from src.learning.outcome_tracker import OutcomeTracker
                with get_session() as session:
                    fresh_alert = session.query(Alert).filter(Alert.id == alert_id).first()
                    if fresh_alert is not None:
                        await OutcomeTracker(self.config).record_entry(fresh_alert)
            except Exception:
                logger.exception("Failed to record outcome entry for alert #%d", alert_id)

            alerts_sent += 1
            alerted_keys.add(key)
            self._last_alert_time[card.ticker] = datetime.utcnow()
            logger.info("Alert #%d sent: %s %s (score=%.1f, conf=%d)",
                        alert_id, card.contract, card.direction, scored.score, card.confidence)

        # 7b. Counterfactual tracking — shadow-track near-miss setups that
        #     were NOT alerted so we can learn whether the gates are calibrated.
        self._record_counterfactuals(scored_setups, alerted_keys, all_setup_ids, filter_reasons)

        # Update scan record
        with get_session() as session:
            db_scan = session.query(Scan).filter(Scan.id == scan_id).first()
            if db_scan:
                db_scan.alerts_sent = alerts_sent
                db_scan.completed_at = datetime.utcnow().isoformat()
                # Per-cycle cost = spend delta over this cycle (not the
                # running daily total, which is what was logged before).
                db_scan.cost_usd = round(
                    max(0.0, self.token_tracker.get_daily_spend() - spend_at_start), 6
                )
            session.commit()

        logger.info("Scan cycle #%d complete: %d setups, %d alerts sent",
                     self._scan_cycle_count, len(scored_setups), alerts_sent)

    def _record_counterfactuals(self, scored_setups, alerted_keys, all_setup_ids,
                                filter_reasons) -> None:
        """Record shadow outcomes for near-miss setups that had real option
        candidates but were not alerted, so the gates can be calibrated later.

        Bounded to setups scoring within a band of the shadow floor to avoid
        tracking obviously-weak setups.
        """
        try:
            from src.db.models import CounterfactualOutcome
            from src.learning.adaptive_thresholds import get_threshold
            shadow_floor = get_threshold("shadow", self.scoring_engine.min_score_shadow)
            band = 10.0
            recorded = 0
            with get_session() as session:
                for scored in scored_setups:
                    key = scored.ticker + scored.direction
                    if key in alerted_keys:
                        continue
                    candidates = scored.option_candidates or []
                    if not candidates:
                        continue
                    if scored.score < shadow_floor - band:
                        continue
                    setup_id = all_setup_ids.get(key)
                    if setup_id is None:
                        continue
                    if (
                        session.query(CounterfactualOutcome)
                        .filter(CounterfactualOutcome.setup_id == setup_id)
                        .first()
                    ):
                        continue
                    cand = candidates[0]
                    mid = float(getattr(cand, "mid", 0) or 0)
                    if mid <= 0:
                        bid = float(getattr(cand, "bid", 0) or 0)
                        ask = float(getattr(cand, "ask", 0) or 0)
                        mid = (bid + ask) / 2 if (bid + ask) > 0 else 0.0
                    if mid <= 0:
                        continue
                    contract = self._candidate_to_contract(scored.ticker, cand)
                    if contract is None:
                        continue
                    # Same IV-scaled exits the real synthesizer would use, so
                    # the counterfactual mirrors an actual alert's risk frame.
                    tgt_pct, stp_pct = self.synthesizer._scaled_exit_pcts(
                        cand, scored.technicals
                    )
                    session.add(CounterfactualOutcome(
                        setup_id=setup_id,
                        ticker=scored.ticker,
                        direction=scored.direction,
                        contract=contract,
                        filter_reason=filter_reasons.get(key, "not_alerted"),
                        score=scored.score,
                        entry_price=round(mid, 2),
                        target_price=round(mid * (1 + tgt_pct), 2),
                        stop_price=round(mid * (1 - stp_pct), 2),
                        created_at=datetime.utcnow().isoformat(),
                    ))
                    recorded += 1
                session.commit()
            if recorded:
                logger.info("Recorded %d counterfactual (non-alerted) setups.", recorded)
        except Exception:
            logger.exception("Counterfactual recording failed (non-fatal).")

    @staticmethod
    def _candidate_to_contract(ticker: str, cand) -> str | None:
        """Build a human contract string ('SPY 530C 7/3') from a candidate,
        matching the format option_quotes.parse_contract expects."""
        try:
            strike = float(getattr(cand, "strike", 0) or 0)
            if strike <= 0:
                return None
            otype = getattr(cand, "option_type", "call")
            type_char = "C" if otype == "call" else "P"
            strike_str = str(int(strike)) if float(strike).is_integer() else str(strike)
            exp = getattr(cand, "expiration", "") or ""
            from datetime import datetime as _dt
            exp_date = _dt.strptime(exp, "%Y-%m-%d").date()
            return f"{ticker} {strike_str}{type_char} {exp_date.month}/{exp_date.day}"
        except (ValueError, TypeError):
            return None

    def _apply_cooldowns(self, scan_results: list) -> list:
        cooldown_min = self.alerts_config.get("cooldown_minutes", 60)
        now = datetime.utcnow()
        filtered = []
        for r in scan_results:
            last = self._last_alert_time.get(r.ticker)
            if last and (now - last).total_seconds() < cooldown_min * 60:
                logger.debug("Cooldown active for %s. Skipping.", r.ticker)
                continue
            filtered.append(r)
        return filtered

    def _count_alerts_today(self) -> int:
        # Count alerts within the current trading day in America/New_York,
        # not UTC. Avoids edge cases where a 4 PM ET alert spills into the
        # next UTC day for 8 PM ET- onwards readers.
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

    async def _notify_breaker_once(self, reason: str) -> None:
        """Send a single Telegram notice when the breaker first trips today."""
        from zoneinfo import ZoneInfo
        today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
        if self.app_state.get("_breaker_notified") == today:
            return
        self.app_state["_breaker_notified"] = today
        try:
            await self.bot.send_alert(
                f"[Talon risk] Alerts paused: {reason}. "
                f"Auto-resets next trading day; /risk to review or /risk reset to override."
            )
        except Exception:
            logger.exception("Breaker notify failed.")

    async def _run_outcome_poll(self):
        """Intraday outcome resolution. Runs every 15 min during market
        hours so option target/stop crossings and MFE/MAE are recorded
        live instead of from a single end-of-day snapshot."""
        from src.learning.outcome_tracker import OutcomeTracker
        tracker = OutcomeTracker(self.config)
        with stage("outcome_poll"):
            await tracker.check_all_open_outcomes()
            await tracker.check_all_open_counterfactuals()

    async def _run_post_market_review(self):
        logger.info("Running post-market review...")
        from src.learning.outcome_tracker import OutcomeTracker
        tracker = OutcomeTracker(self.config)
        await tracker.check_all_open_outcomes()
        await tracker.check_all_open_counterfactuals()

        # Daily postmortem: summarize the day's resolved trades and push the
        # LLM insights to Telegram so learning is visible day-to-day (the
        # weight/threshold/hour adapters still run only in the weekly run).
        try:
            from src.learning.postmortem import PostmortemRunner
            runner = PostmortemRunner(self.orchestrator, self.config)
            summary = await runner.run_daily()
            if summary:
                text = (
                    f"[Talon daily review] {summary.get('period', '')}\n"
                    f"Alerts: {summary.get('total_alerts', 0)} | "
                    f"Resolved: {summary.get('resolved', 0)} "
                    f"(W:{summary.get('wins', 0)} L:{summary.get('losses', 0)})\n"
                    f"Avg P&L: {summary.get('avg_pnl_pct', 0):+.1f}%"
                )
                await self.bot.send_alert(text)
        except Exception:
            logger.exception("Daily postmortem failed (non-fatal).")

        logger.info("Post-market review complete.")

    async def _run_position_check(self):
        from src.learning.position_monitor import check_positions
        notifications = await asyncio.to_thread(check_positions)
        for msg in notifications:
            await self.bot.send_alert(msg)

    async def _run_weekly_postmortem(self):
        logger.info("Running weekly postmortem...")
        from src.learning.postmortem import PostmortemRunner
        runner = PostmortemRunner(self.orchestrator, self.config)
        await runner.run_weekly()
        logger.info("Weekly postmortem complete.")

    async def _run_wal_checkpoint(self):
        from src.db.database import checkpoint_wal
        await asyncio.to_thread(checkpoint_wal)

    def _chart_dataframes(self, ticker: str):
        """Return (intraday_df, daily_df) suitable for vision chart rendering.

        Intraday: ~3 days of 15-min bars, daily: ~3 months of daily bars.
        Either may be None on failure; vision_analyst tolerates one being
        absent (it dups it for both panes).
        """
        from src.data.alpaca_provider import get_alpaca_provider
        provider = get_alpaca_provider()
        intraday = None
        daily = None
        if provider.enabled:
            intraday = provider.get_intraday_bars(ticker, minutes=15, lookback_hours=24)
            daily = provider.get_daily_bars(ticker, lookback_days=90)
        if intraday is None or intraday.empty:
            try:
                import yfinance as yf
                intraday = yf.Ticker(ticker).history(period="5d", interval="15m")
            except Exception:
                logger.debug("yfinance intraday fetch failed for %s", ticker)
                intraday = None
        if daily is None or (hasattr(daily, "empty") and daily.empty):
            try:
                import yfinance as yf
                daily = yf.Ticker(ticker).history(period="3mo", interval="1d")
            except Exception:
                logger.debug("yfinance daily fetch failed for %s", ticker)
                daily = None
        return intraday, daily

    async def _run_daily_heartbeat(self):
        """Pre-open status ping so a silent failure is noticed by the user."""
        budget = self.model_router.budget_status() if self.model_router else {}
        alerts_today = self._count_alerts_today()
        text = (
            f"[Talon heartbeat] {self.mode.upper()} | "
            f"alerts today: {alerts_today} | "
            f"spend: ${budget.get('spent_usd', 0):.2f}/${budget.get('budget_usd', 0):.2f} "
            f"({budget.get('pct_used', 0):.0f}%)"
        )
        await self.bot.send_alert(text)


def _enforce_single_instance():
    """Replace any existing Talon process and write our PID.

    Verifies the recorded PID is actually a Talon process before signalling.
    Sends SIGTERM with a 3s grace period; only escalates to SIGKILL if the
    process refuses to exit. This prevents accidentally killing an unrelated
    process that has reused the PID after a reboot.
    """
    import signal
    import subprocess
    import time

    pid_file = Path(__file__).resolve().parent.parent / "data" / "talon.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)

    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text().strip())
        except ValueError:
            old_pid = None
        if old_pid and old_pid != os.getpid():
            _terminate_if_talon(old_pid, signal_first=signal.SIGTERM,
                                grace_seconds=3, subprocess=subprocess, time=time)

    pid_file.write_text(str(os.getpid()))


def _terminate_if_talon(pid: int, *, signal_first, grace_seconds: int,
                        subprocess, time) -> None:
    """Verify pid belongs to a Talon process, then SIGTERM/SIGKILL."""
    import signal as _sig
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return

    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True, text=True, timeout=2,
        )
        cmdline = (result.stdout or "").strip()
    except Exception:
        cmdline = ""

    if "src.main" not in cmdline and "talon" not in cmdline.lower():
        logger.warning(
            "Stale PID %d does not look like a Talon process (cmd=%r). "
            "Refusing to signal it.", pid, cmdline,
        )
        return

    try:
        os.kill(pid, signal_first)
    except (ProcessLookupError, PermissionError):
        return

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            logger.info("Old Talon process %d exited cleanly.", pid)
            return
        time.sleep(0.2)

    try:
        os.kill(pid, _sig.SIGKILL)
        logger.info("Force-killed old Talon process %d (SIGKILL).", pid)
        time.sleep(0.5)
    except (ProcessLookupError, PermissionError):
        return


def main():
    _enforce_single_instance()
    talon = Talon()
    asyncio.run(talon.run())


if __name__ == "__main__":
    main()
