"""Run a manual postmortem review (daily or weekly)."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import yaml

from src.cost.rate_limiter import RateLimiter
from src.cost.token_tracker import TokenTracker
from src.db.database import init_database
from src.learning.postmortem import PostmortemRunner
from src.orchestrator import TalonOrchestrator


def load_config() -> dict:
    config_dir = Path(__file__).resolve().parent.parent / "config"
    config = {}
    for name in ("settings", "model_routing", "sources"):
        path = config_dir / f"{name}.yaml"
        if path.exists():
            with open(path) as f:
                config.update(yaml.safe_load(f) or {})
    return config


async def main():
    init_database()
    config = load_config()

    tracker = TokenTracker()
    limiter = RateLimiter()
    model_config = config.get("models", {})
    cost_controls = config.get("cost_controls", {})

    orchestrator = TalonOrchestrator(model_config, cost_controls, tracker, limiter)
    runner = PostmortemRunner(orchestrator, config)

    period = sys.argv[1] if len(sys.argv) > 1 else "daily"

    if period == "weekly":
        result = await runner.run_weekly()
    else:
        result = await runner.run_daily()

    if result:
        import json
        print(json.dumps(result, indent=2))
    else:
        print("No data for the requested period.")


if __name__ == "__main__":
    asyncio.run(main())
