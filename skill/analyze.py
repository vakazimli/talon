"""Talon analysis skill entry point.

Usage:
    python -m skill.analyze <TICKER> [quick|deep]

Prints a JSON analysis for a single ticker using Talon's full pipeline
(scan -> detect -> options -> score -> optional vision -> deterministically
priced synthesis), with the same guardrails as the live alerts. Does not
send alerts or write to the database.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.analysis.on_demand import OnDemandAnalyzer, format_analysis
from src.analysis.scoring_engine import ScoringEngine
from src.analysis.setup_detector import SetupDetector
from src.analysis.synthesizer import Synthesizer
from src.analysis.vision_analyst import VisionAnalyst
from src.cost.model_router import ModelRouter
from src.cost.rate_limiter import RateLimiter
from src.cost.token_tracker import TokenTracker
from src.db.database import init_database
from src.main import load_config
from src.orchestrator import TalonOrchestrator
from src.scanner.market_scanner import MarketScanner
from src.scanner.news_scanner import NewsScanner
from src.scanner.options_scanner import OptionsScanner


def build_analyzer() -> OnDemandAnalyzer:
    config = load_config()
    init_database()
    tracker = TokenTracker()
    rate_limiter = RateLimiter()
    router = ModelRouter(config.get("models", {}), config.get("cost_controls", {}), tracker)
    orch = TalonOrchestrator(
        config.get("models", {}), config.get("cost_controls", {}),
        tracker, rate_limiter, router=router,
    )
    return OnDemandAnalyzer(
        config,
        MarketScanner(config),
        OptionsScanner(config),
        NewsScanner(config),
        SetupDetector(orch),
        ScoringEngine(config),
        Synthesizer(orch, config),
        VisionAnalyst(orch, config),
    )


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m skill.analyze <TICKER> [quick|deep]")
        sys.exit(1)
    ticker = sys.argv[1]
    deep = len(sys.argv) > 2 and sys.argv[2].lower() == "deep"
    analyzer = build_analyzer()
    result = analyzer.analyze(ticker, deep=deep)
    print(format_analysis(result))
    print("---JSON---")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
