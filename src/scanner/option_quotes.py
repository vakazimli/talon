"""Option contract quote helper.

Looks up the live price of a specific option contract by its
human-readable symbol (e.g. "IWM 282C 5/8"). Tradier first if a
key is configured, yfinance options-chain fallback otherwise.

Used by:
  - learning/outcome_tracker.py (resolves alert outcomes)
  - learning/position_monitor.py (tracks user-confirmed positions)
"""

import logging
import math
import os
from dataclasses import dataclass
from datetime import date

import httpx

logger = logging.getLogger(__name__)

TRADIER_PROD_BASE = "https://api.tradier.com/v1"
TRADIER_SANDBOX_BASE = "https://sandbox.tradier.com/v1"


@dataclass
class OptionQuote:
    bid: float
    ask: float
    last: float
    mid: float


@dataclass
class ParsedContract:
    ticker: str
    strike: float
    option_type: str  # "call" | "put"
    expiration: date


def parse_contract(contract: str) -> ParsedContract | None:
    """Parse a contract string like 'IWM 282C 5/8' into components.

    Returns None if the string is malformed.
    """
    if not contract:
        return None

    parts = contract.split()
    if len(parts) < 3:
        return None

    ticker = parts[0]
    strike_part = parts[1]
    expiry_part = parts[-1]

    if not strike_part:
        return None
    type_char = strike_part[-1].upper()
    if type_char not in ("C", "P"):
        return None
    strike_str = strike_part[:-1]
    try:
        strike = float(strike_str)
    except ValueError:
        return None
    option_type = "call" if type_char == "C" else "put"

    try:
        m, d = expiry_part.split("/")
        month = int(m)
        day = int(d)
    except (ValueError, IndexError):
        return None

    year = date.today().year
    try:
        exp = date(year, month, day)
    except ValueError:
        return None
    if exp < date.today():
        try:
            exp = date(year + 1, month, day)
        except ValueError:
            return None

    return ParsedContract(
        ticker=ticker,
        strike=strike,
        option_type=option_type,
        expiration=exp,
    )


def to_occ_symbol(parsed: ParsedContract) -> str:
    """Convert parsed contract to OCC option symbol (e.g. IWM260508C00282000)."""
    type_char = "C" if parsed.option_type == "call" else "P"
    strike_thousands = int(round(parsed.strike * 1000))
    return (
        f"{parsed.ticker}"
        f"{parsed.expiration.strftime('%y%m%d')}"
        f"{type_char}"
        f"{strike_thousands:08d}"
    )


def _is_sane_quote(q: OptionQuote | None) -> bool:
    """Reject quotes that would corrupt outcome P&L: NaN, all-zero, or a
    crossed market (bid > ask). At least one of mid/last must be positive."""
    if q is None:
        return False
    for v in (q.bid, q.ask, q.last, q.mid):
        if v is None or math.isnan(v) or v < 0:
            return False
    if q.bid > 0 and q.ask > 0 and q.ask < q.bid:
        return False  # crossed market = bad data
    if q.mid <= 0 and q.last <= 0:
        return False
    return True


def get_option_quote(contract: str) -> OptionQuote | None:
    """Fetch the current quote for an option contract.

    Provider preference: Alpaca > Tradier > yfinance.
    Returns None if all paths fail or only return unusable (NaN/zero/crossed)
    quotes — we never resolve an outcome on bad data.
    """
    parsed = parse_contract(contract)
    if parsed is None:
        logger.warning("Could not parse contract symbol: %r", contract)
        return None

    occ = to_occ_symbol(parsed)
    quote = _alpaca_quote(occ)
    if _is_sane_quote(quote):
        return quote

    api_key = os.environ.get("TRADIER_API_KEY", "")
    if api_key:
        quote = _tradier_quote(parsed, api_key)
        if _is_sane_quote(quote):
            return quote

    quote = _yfinance_quote(parsed)
    return quote if _is_sane_quote(quote) else None


def _alpaca_quote(occ: str) -> OptionQuote | None:
    try:
        from src.data.alpaca_provider import get_alpaca_provider
    except Exception:
        return None
    provider = get_alpaca_provider()
    if not provider.enabled:
        return None
    raw = provider.get_option_quote(occ)
    if not raw:
        return None
    bid = float(raw.get("bid") or 0)
    ask = float(raw.get("ask") or 0)
    if bid + ask == 0:
        return None
    mid = (bid + ask) / 2
    return OptionQuote(bid=bid, ask=ask, last=0.0, mid=round(mid, 4))


def _tradier_quote(parsed: ParsedContract, api_key: str) -> OptionQuote | None:
    occ = to_occ_symbol(parsed)
    base = TRADIER_PROD_BASE
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(
                f"{base}/markets/quotes",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                },
                params={"symbols": occ, "greeks": "false"},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        logger.exception("Tradier quote fetch failed for %s", occ)
        return None

    quote = data.get("quotes", {}).get("quote", {})
    if isinstance(quote, list):
        quote = quote[0] if quote else {}
    if not quote:
        logger.debug("Tradier returned empty quote for %s", occ)
        return None

    bid = _safe_float(quote.get("bid"))
    ask = _safe_float(quote.get("ask"))
    last = _safe_float(quote.get("last"))
    if bid + ask + last == 0:
        return None

    mid = (bid + ask) / 2 if (bid + ask) > 0 else last
    return OptionQuote(bid=bid, ask=ask, last=last, mid=round(mid, 4))


def _yfinance_quote(parsed: ParsedContract) -> OptionQuote | None:
    try:
        import yfinance as yf

        tk = yf.Ticker(parsed.ticker)
        try:
            available = tk.options or ()
        except Exception:
            logger.exception("yfinance: failed to list options for %s", parsed.ticker)
            return None

        exp_str = parsed.expiration.strftime("%Y-%m-%d")
        if exp_str not in available:
            logger.debug(
                "yfinance: expiration %s not in chain for %s", exp_str, parsed.ticker
            )
            return None

        chain = tk.option_chain(exp_str)
        df = chain.calls if parsed.option_type == "call" else chain.puts
        if df is None or df.empty:
            return None

        target = parsed.strike
        match = df[df["strike"].between(target - 0.001, target + 0.001)]
        if match.empty:
            logger.debug(
                "yfinance: no row at strike %.3f for %s %s chain",
                target, parsed.ticker, parsed.option_type,
            )
            return None
        row = match.iloc[0]

        bid = _safe_float(row.get("bid"))
        ask = _safe_float(row.get("ask"))
        last = _safe_float(row.get("lastPrice"))
        if bid + ask + last == 0:
            return None

        mid = (bid + ask) / 2 if (bid + ask) > 0 else last
        return OptionQuote(bid=bid, ask=ask, last=last, mid=round(mid, 4))
    except Exception:
        logger.exception("yfinance options-chain lookup failed for %s", parsed)
        return None


def _safe_float(value) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f):
        return 0.0
    return f
