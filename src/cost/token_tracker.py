import logging
from datetime import datetime, date

from src.db.database import get_session
from src.db.models import CostLog

logger = logging.getLogger(__name__)


class TokenTracker:
    """Tracks token usage and costs per model call. Logs every call to the DB."""

    def __init__(self):
        self._daily_cache: dict[str, float] = {}

    def log_usage(
        self,
        model: str,
        task: str,
        tokens_input: int,
        tokens_output: int,
        cost_usd: float,
    ) -> None:
        today = date.today().isoformat()
        now = datetime.utcnow().isoformat()

        entry = CostLog(
            date=today,
            model=model,
            task=task,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost_usd,
            logged_at=now,
        )

        with get_session() as session:
            session.add(entry)
            session.commit()

        self._daily_cache[today] = self._daily_cache.get(today, 0.0) + cost_usd
        logger.debug(
            "Logged %s tokens (%s in, %s out) for %s/%s — $%.4f",
            tokens_input + tokens_output,
            tokens_input,
            tokens_output,
            task,
            model,
            cost_usd,
        )

    def get_daily_spend(self, day: str | None = None) -> float:
        day = day or date.today().isoformat()
        if day in self._daily_cache:
            return self._daily_cache[day]

        with get_session() as session:
            from sqlalchemy import func
            result = session.query(func.sum(CostLog.cost_usd)).filter(
                CostLog.date == day
            ).scalar()
            total = result or 0.0
            self._daily_cache[day] = total
            return total

    def get_daily_breakdown(self, day: str | None = None) -> dict:
        day = day or date.today().isoformat()
        with get_session() as session:
            from sqlalchemy import func
            rows = (
                session.query(
                    CostLog.task,
                    func.sum(CostLog.cost_usd).label("total_cost"),
                    func.sum(CostLog.tokens_input).label("total_input"),
                    func.sum(CostLog.tokens_output).label("total_output"),
                )
                .filter(CostLog.date == day)
                .group_by(CostLog.task)
                .all()
            )
            return {
                row.task: {
                    "cost_usd": row.total_cost,
                    "tokens_input": row.total_input,
                    "tokens_output": row.total_output,
                }
                for row in rows
            }

    def estimate_cost(
        self, model_config: dict, tokens_input: int = 500, tokens_output: int = 500
    ) -> float:
        input_cost = (tokens_input / 1000) * model_config.get("cost_per_1k_input", 0)
        output_cost = (tokens_output / 1000) * model_config.get("cost_per_1k_output", 0)
        return input_cost + output_cost
