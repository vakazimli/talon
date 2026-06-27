import logging
import os

from .token_tracker import TokenTracker

logger = logging.getLogger(__name__)


class ModelRouter:
    """Budget-aware model routing. Downgrades expensive models when budget is tight."""

    def __init__(self, model_config: dict, cost_controls: dict, token_tracker: TokenTracker):
        self.models = model_config
        # The DAILY_TOKEN_BUDGET_USD env var (documented in .env.example) takes
        # precedence over the YAML default so the two can't silently diverge.
        env_budget = os.environ.get("DAILY_TOKEN_BUDGET_USD")
        if env_budget:
            try:
                self.daily_budget = float(env_budget)
            except ValueError:
                self.daily_budget = cost_controls.get("daily_budget_usd", 2.0)
        else:
            self.daily_budget = cost_controls.get("daily_budget_usd", 2.0)
        self.warn_pct = cost_controls.get("warn_at_pct", 75) / 100.0
        self.hard_stop_pct = cost_controls.get("hard_stop_at_pct", 95) / 100.0
        self.tracker = token_tracker

    @property
    def spent_today(self) -> float:
        return self.tracker.get_daily_spend()

    @property
    def budget_pct_used(self) -> float:
        if self.daily_budget <= 0:
            return 1.0
        return self.spent_today / self.daily_budget

    def can_afford(self, model_config: dict, estimated_tokens: int = 1000) -> bool:
        estimated_cost = self.tracker.estimate_cost(
            model_config, estimated_tokens, estimated_tokens
        )
        return (self.spent_today + estimated_cost) < (self.daily_budget * self.hard_stop_pct)

    def get_model_for_task(self, task_type: str) -> dict:
        """Return model config for a task, downgrading if budget is tight."""
        config = self.models.get(task_type)
        if config is None:
            raise ValueError(f"Unknown task type: {task_type}")

        pct = self.budget_pct_used

        if pct >= self.hard_stop_pct:
            logger.warning("Budget exhausted (%.0f%%). Blocking all model calls.", pct * 100)
            return None

        if pct >= self.warn_pct and task_type == "synthesizer":
            logger.warning(
                "Budget at %.0f%%. Downgrading synthesizer to analyst model.", pct * 100
            )
            return self.models.get("analyst", config)

        return config

    def budget_status(self) -> dict:
        spent = self.spent_today
        return {
            "spent_usd": round(spent, 4),
            "budget_usd": self.daily_budget,
            "remaining_usd": round(self.daily_budget - spent, 4),
            "pct_used": round(self.budget_pct_used * 100, 1),
        }
