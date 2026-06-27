from src.analysis.synthesizer import TradeCard

MAX_MESSAGE_LENGTH = 500


def format_trade_card(card: TradeCard, mode: str = "shadow", alert_id: int | None = None) -> str:
    """Format a TradeCard into a concise Telegram message.

    Target: under 500 characters, no walls of text.
    """
    direction_emoji = "\U0001f7e2" if card.direction == "bullish" else "\U0001f534"
    direction_label = "Bullish" if card.direction == "bullish" else "Bearish"
    stars = "\u2605" * card.confidence + "\u2606" * (5 - card.confidence)
    mode_tag = mode.upper()
    id_tag = f"  [#{alert_id}]" if alert_id else ""

    msg = (
        f"{direction_emoji} {card.contract} -- {direction_label}{id_tag}\n"
        f"Entry: ${card.entry_low:.2f}-${card.entry_high:.2f}\n"
        f"Target: ${card.target:.2f}"
    )

    if card.sell_by:
        msg += f" ({card.sell_by})"
    msg += "\n"

    msg += f"Stop: ${card.stop:.2f}\n"
    msg += f"Why: {card.rationale}\n"
    msg += f"Setup: {card.setup_type} | Score: {card.score:.0f}\n"
    msg += f"Confidence: {stars}\n"
    msg += f"Mode: {mode_tag}"

    if alert_id:
        msg += f"\n/took {alert_id} -- I took this trade"
        msg += f"\n/feedback {alert_id} good|bad|late|skip"

    if len(msg) > MAX_MESSAGE_LENGTH:
        # Rebuild with truncated rationale and dropped sell_by suffix.
        # Reserve 3 chars for the "..." we'll append.
        scaffold = (
            f"{direction_emoji} {card.contract} -- {direction_label}{id_tag}\n"
            f"Entry: ${card.entry_low:.2f}-${card.entry_high:.2f}\n"
            f"Target: ${card.target:.2f}\n"
            f"Stop: ${card.stop:.2f}\n"
            f"Why: \n"
            f"Setup: {card.setup_type} | Score: {card.score:.0f}\n"
            f"Confidence: {stars}\n"
            f"Mode: {mode_tag}"
        )
        suffix = ""
        if alert_id:
            suffix = (
                f"\n/took {alert_id} -- I took this trade"
                f"\n/feedback {alert_id} good|bad|late|skip"
            )
        rationale_budget = MAX_MESSAGE_LENGTH - len(scaffold) - len(suffix) - 3
        if rationale_budget > 20:
            truncated_rationale = card.rationale[:rationale_budget] + "..."
            msg = (
                f"{direction_emoji} {card.contract} -- {direction_label}{id_tag}\n"
                f"Entry: ${card.entry_low:.2f}-${card.entry_high:.2f}\n"
                f"Target: ${card.target:.2f}\n"
                f"Stop: ${card.stop:.2f}\n"
                f"Why: {truncated_rationale}\n"
                f"Setup: {card.setup_type} | Score: {card.score:.0f}\n"
                f"Confidence: {stars}\n"
                f"Mode: {mode_tag}"
            )
            if alert_id:
                msg += suffix

    return msg


def format_status(
    mode: str,
    alerts_today: int,
    budget_status: dict,
    scanning_paused: bool,
) -> str:
    status = "PAUSED" if scanning_paused else "ACTIVE"
    return (
        f"Talon Status: {status}\n"
        f"Mode: {mode.upper()}\n"
        f"Alerts today: {alerts_today}\n"
        f"Budget: ${budget_status['spent_usd']:.2f} / ${budget_status['budget_usd']:.2f} "
        f"({budget_status['pct_used']:.0f}%)"
    )


def format_budget(budget_status: dict) -> str:
    return (
        f"Daily Budget: ${budget_status['budget_usd']:.2f}\n"
        f"Spent: ${budget_status['spent_usd']:.4f}\n"
        f"Remaining: ${budget_status['remaining_usd']:.4f}\n"
        f"Used: {budget_status['pct_used']:.1f}%"
    )


def format_sources(source_scores: list[dict]) -> str:
    if not source_scores:
        return "No source data yet."
    lines = ["Source Reliability Rankings:"]
    for s in sorted(source_scores, key=lambda x: x["score"] or 0, reverse=True):
        score = s.get("score") or 0
        bar = "\u2588" * int(score * 10) + "\u2591" * (10 - int(score * 10))
        status = "" if s.get("enabled", True) else "  [DISABLED]"
        lines.append(
            f"  {s['name']}: {bar} {score:.2f} ({s.get('total') or 0} signals){status}"
        )
        if not s.get("enabled", True) and s.get("disabled_reason"):
            lines.append(f"    -> {s['disabled_reason']}")
    return "\n".join(lines)
