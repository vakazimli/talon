"""Active position monitoring. Tracks trades user confirmed with /took."""

import json
import logging
from datetime import datetime
from pathlib import Path

from src.db.database import get_session
from src.db.models import Alert, Outcome
from src.scanner.option_quotes import get_option_quote

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
POSITIONS_FILE = DATA_DIR / "active_positions.json"


def _load_positions() -> list[dict]:
    if POSITIONS_FILE.exists():
        with open(POSITIONS_FILE) as f:
            return json.load(f)
    return []


def _save_positions(positions: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2)


def add_position(alert_id: int) -> dict | None:
    """Mark that the user took a trade from alert #id."""
    with get_session() as session:
        alert = session.query(Alert).filter(Alert.id == alert_id).first()
        if not alert:
            return None

        entry_price = ((alert.entry_price_low or 0) + (alert.entry_price_high or 0)) / 2

        position = {
            "alert_id": alert_id,
            "ticker": alert.ticker,
            "contract": alert.contract,
            "direction": alert.direction,
            "entry_price": round(entry_price, 2),
            "target_price": alert.target_price,
            "stop_price": alert.stop_price,
            "took_at": datetime.utcnow().isoformat(),
            "status": "active",
            "last_price": entry_price,
            "last_check": datetime.utcnow().isoformat(),
            "last_notified": None,
            "max_price": entry_price,
            "min_price": entry_price,
        }

        positions = _load_positions()
        if any(p["alert_id"] == alert_id for p in positions):
            return None  # already tracked
        positions.append(position)
        _save_positions(positions)

        logger.info("Position opened: alert #%d %s at $%.2f", alert_id, alert.contract, entry_price)
        return position


def close_position(alert_id: int, exit_price: float | None = None) -> dict | None:
    """Close a tracked position."""
    positions = _load_positions()
    closed = None
    for p in positions:
        if p["alert_id"] == alert_id and p["status"] == "active":
            p["status"] = "closed"
            p["closed_at"] = datetime.utcnow().isoformat()
            if exit_price:
                p["exit_price"] = exit_price
            else:
                p["exit_price"] = p.get("last_price", p["entry_price"])

            entry = p["entry_price"]
            ext = p["exit_price"]
            if entry > 0:
                p["pnl_pct"] = round(((ext / entry) - 1) * 100, 1)
                p["pnl_dollars"] = round(ext - entry, 2)
            closed = dict(p)
            break
    _save_positions(positions)

    if closed:
        logger.info("Position closed: alert #%d, P&L: %.1f%%", alert_id, closed.get("pnl_pct", 0))
    return closed


def get_active_positions() -> list[dict]:
    return [p for p in _load_positions() if p["status"] == "active"]


def check_positions() -> list[dict]:
    """Check all active positions and return notifications to send."""
    positions = _load_positions()
    notifications = []
    changed = False

    for pos in positions:
        if pos["status"] != "active":
            continue

        contract = pos.get("contract", "")
        quote = get_option_quote(contract)
        if quote is None:
            logger.debug("Could not fetch option quote for %s", contract)
            continue
        current_price = quote.mid if quote.mid > 0 else quote.last
        if current_price <= 0:
            continue

        entry = pos["entry_price"]
        target = pos["target_price"] or 0
        stop = pos["stop_price"] or 0

        if current_price > pos.get("max_price", 0):
            pos["max_price"] = current_price
        if current_price < pos.get("min_price", 999999):
            pos["min_price"] = current_price

        pos["last_price"] = round(current_price, 2)
        pos["last_check"] = datetime.utcnow().isoformat()

        if entry <= 0:
            continue

        pnl_pct = ((current_price / entry) - 1) * 100
        changed = True
        notification = None

        # Both directions are long-the-option, so target > entry and stop < entry
        # regardless of bullish (calls) vs bearish (puts).
        if target > 0 and current_price >= target:
            notification = _format_target_hit(pos, current_price, pnl_pct)
            pos["status"] = "target_hit"
        elif stop > 0 and current_price <= stop:
            notification = _format_stop_hit(pos, current_price, pnl_pct)
            pos["status"] = "stopped"
        elif target > 0 and current_price >= target * 0.90:
            notification = _format_approaching_target(pos, current_price, pnl_pct, target)
        elif stop > 0 and current_price <= stop * 1.10:
            notification = _format_approaching_stop(pos, current_price, pnl_pct, stop)

        if notification:
            last_notified = pos.get("last_notified")
            if not last_notified or _minutes_since(last_notified) >= 15:
                notifications.append(notification)
                pos["last_notified"] = datetime.utcnow().isoformat()

    if changed:
        _save_positions(positions)

    return notifications


def _format_target_hit(pos: dict, price: float, pnl: float) -> str:
    return (
        f"{pos['contract']} TARGET HIT [#{pos['alert_id']}]\n"
        f"Entry: ${pos['entry_price']:.2f} | Exit: ${price:.2f}\n"
        f"P&L: {pnl:+.1f}%\n"
        f"Take profit now."
    )


def _format_stop_hit(pos: dict, price: float, pnl: float) -> str:
    return (
        f"{pos['contract']} STOP HIT [#{pos['alert_id']}]\n"
        f"Entry: ${pos['entry_price']:.2f} | Now: ${price:.2f}\n"
        f"P&L: {pnl:+.1f}%\n"
        f"Exit position."
    )


def _format_approaching_target(pos: dict, price: float, pnl: float, target: float) -> str:
    dist = abs(target - price) / target * 100
    return (
        f"{pos['contract']} update [#{pos['alert_id']}]\n"
        f"Entry: ${pos['entry_price']:.2f} | Now: ${price:.2f} ({pnl:+.1f}%)\n"
        f"Target ${target:.2f} is {dist:.0f}% away. Consider partial profit."
    )


def _format_approaching_stop(pos: dict, price: float, pnl: float, stop: float) -> str:
    dist = abs(price - stop) / stop * 100
    return (
        f"{pos['contract']} warning [#{pos['alert_id']}]\n"
        f"Entry: ${pos['entry_price']:.2f} | Now: ${price:.2f} ({pnl:+.1f}%)\n"
        f"Stop ${stop:.2f} is {dist:.0f}% away. Watch closely."
    )


def _minutes_since(iso_str: str) -> float:
    try:
        then = datetime.fromisoformat(iso_str)
        return (datetime.utcnow() - then).total_seconds() / 60
    except (ValueError, TypeError):
        return 999
