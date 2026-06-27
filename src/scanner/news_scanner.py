"""News and earnings-calendar scanner backed by Finnhub.

Free tier limits: 60 calls/min. We hit the API at most once per ticker per
scan cycle and cache responses for the cycle, so even a 10-ticker watchlist
stays well under the limit.

Public API:
  - NewsScanner(config).fetch_news(ticker) -> list[NewsItem]
  - NewsScanner(config).has_earnings_within(ticker, days) -> bool
  - extract_signals(news_items) -> list[str]   (event-type signal labels)
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Iterable

import httpx

logger = logging.getLogger(__name__)

FINNHUB_BASE = "https://finnhub.io/api/v1"

# Heuristics for translating headlines into event signals
_EVENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "news_earnings": ("earnings", "beat", "miss", "guidance", "raised guidance",
                      "lowered guidance", "Q1", "Q2", "Q3", "Q4", "fiscal"),
    "news_upgrade": ("upgrade", "upgraded", "raised price target",
                     "outperform", "buy rating"),
    "news_downgrade": ("downgrade", "downgraded", "cut price target",
                       "underperform", "sell rating"),
    "news_mna": ("acquire", "acquisition", "merger", "buyout", "takeover",
                 "stake"),
    "news_regulatory": ("FDA", "approval", "clearance", "investigation",
                        "lawsuit", "SEC", "antitrust"),
    "news_product": ("launches", "unveils", "announces", "partnership"),
    "news_macro": ("Fed", "FOMC", "CPI", "inflation", "jobs report",
                   "unemployment", "tariff"),
}


@dataclass
class NewsItem:
    headline: str
    summary: str
    source: str
    timestamp: int  # unix seconds
    sentiment: float | None = None  # -1..1 if provided


@dataclass
class _CacheEntry:
    fetched_at: float
    payload: list[NewsItem] = field(default_factory=list)


class NewsScanner:
    """Pulls company news for a ticker and exposes simple event signals.

    No-ops gracefully when FINNHUB_API_KEY isn't configured so the rest of
    the pipeline keeps working (just without news context).
    """

    def __init__(self, config: dict | None = None, ttl_seconds: int = 600):
        self.api_key = os.environ.get("FINNHUB_API_KEY", "")
        self.ttl = ttl_seconds
        self._cache: dict[str, _CacheEntry] = {}
        self._earnings_cache: dict[str, _CacheEntry] = {}
        self._quote_cache: dict[str, _CacheEntry] = {}
        self._enabled = bool(self.api_key)
        if not self._enabled:
            logger.info("NewsScanner disabled (FINNHUB_API_KEY not set).")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def fetch_news(self, ticker: str, max_items: int = 6) -> list[NewsItem]:
        if not self._enabled:
            return []
        cached = self._cache.get(ticker)
        now = time.monotonic()
        if cached and (now - cached.fetched_at) < self.ttl:
            return cached.payload[:max_items]

        today = date.today()
        from_date = today - timedelta(days=5)
        try:
            with httpx.Client(timeout=8) as client:
                resp = client.get(
                    f"{FINNHUB_BASE}/company-news",
                    params={
                        "symbol": ticker,
                        "from": from_date.isoformat(),
                        "to": today.isoformat(),
                        "token": self.api_key,
                    },
                )
                resp.raise_for_status()
                articles = resp.json() or []
        except Exception:
            logger.exception("Finnhub news fetch failed for %s", ticker)
            self._cache[ticker] = _CacheEntry(fetched_at=now, payload=[])
            return []

        items = [
            NewsItem(
                headline=(a.get("headline") or "")[:240],
                summary=(a.get("summary") or "")[:300],
                source=a.get("source") or "",
                timestamp=int(a.get("datetime") or 0),
            )
            for a in articles
            if a.get("headline")
        ]
        items.sort(key=lambda i: i.timestamp, reverse=True)
        self._cache[ticker] = _CacheEntry(fetched_at=now, payload=items)
        return items[:max_items]

    def get_quote(self, ticker: str) -> dict | None:
        """Real-time US-stock quote. Free-tier endpoint, no ID required.

        Returns {"current": float, "high": float, "low": float, "open": float,
        "prev_close": float, "ts": int} or None on failure.
        Cached for the per-cycle TTL to stay polite on rate limits.
        """
        if not self._enabled:
            return None
        cache_key = f"quote:{ticker}"
        cached = self._quote_cache.get(cache_key)
        now = time.monotonic()
        if cached and (now - cached.fetched_at) < 30:
            return cached.payload  # type: ignore[return-value]
        try:
            with httpx.Client(timeout=6) as client:
                resp = client.get(
                    f"{FINNHUB_BASE}/quote",
                    params={"symbol": ticker, "token": self.api_key},
                )
                resp.raise_for_status()
                data = resp.json() or {}
        except Exception:
            logger.exception("Finnhub quote fetch failed for %s", ticker)
            return None
        if not data or float(data.get("c") or 0) <= 0:
            return None
        payload = {
            "current": float(data.get("c") or 0),
            "high": float(data.get("h") or 0),
            "low": float(data.get("l") or 0),
            "open": float(data.get("o") or 0),
            "prev_close": float(data.get("pc") or 0),
            "ts": int(data.get("t") or 0),
        }
        self._quote_cache[cache_key] = _CacheEntry(fetched_at=now, payload=payload)
        return payload

    def has_earnings_within(self, ticker: str, days: int = 2) -> bool:
        """Check if earnings are scheduled in the next N business days.

        Uses the Finnhub /calendar/earnings endpoint. Returns False on
        any failure (we'd rather alert and let the user override than
        silently block on flaky network calls).
        """
        if not self._enabled:
            return False
        cache_key = f"{ticker}:{days}"
        cached = self._earnings_cache.get(cache_key)
        now = time.monotonic()
        if cached and (now - cached.fetched_at) < 6 * 3600:
            return bool(cached.payload)

        today = date.today()
        end = today + timedelta(days=days)
        try:
            with httpx.Client(timeout=8) as client:
                resp = client.get(
                    f"{FINNHUB_BASE}/calendar/earnings",
                    params={
                        "from": today.isoformat(),
                        "to": end.isoformat(),
                        "symbol": ticker,
                        "token": self.api_key,
                    },
                )
                resp.raise_for_status()
                data = resp.json() or {}
        except Exception:
            logger.exception("Finnhub earnings calendar fetch failed for %s", ticker)
            self._earnings_cache[cache_key] = _CacheEntry(fetched_at=now, payload=[])
            return False

        rows = data.get("earningsCalendar") or []
        has = bool(rows)
        self._earnings_cache[cache_key] = _CacheEntry(
            fetched_at=now,
            payload=rows if has else [],
        )
        if has:
            logger.info("Earnings within %d days for %s: %s", days, ticker, rows[0].get("date"))
        return has


def extract_signals(items: Iterable[NewsItem]) -> list[str]:
    """Translate news headlines/summaries into event-style signal tags.

    These tags piggyback on the existing source-reliability machinery —
    they will get tracked and auto-disabled like technical signals do.
    """
    signals: set[str] = set()
    for item in items:
        text = f"{item.headline} {item.summary}".lower()
        for label, keywords in _EVENT_KEYWORDS.items():
            if any(kw.lower() in text for kw in keywords):
                signals.add(label)
    return sorted(signals)


def format_for_prompt(items: Iterable[NewsItem], max_items: int = 4) -> str:
    """Compact human-readable summary for inclusion in LLM prompts."""
    items = list(items)[:max_items]
    if not items:
        return "  (no recent news)"
    lines = []
    for it in items:
        ts = ""
        if it.timestamp:
            try:
                ts = datetime.utcfromtimestamp(it.timestamp).strftime("%m/%d %H:%MZ")
            except Exception:
                ts = ""
        lines.append(f"  - [{ts}] {it.headline[:140]}")
    return "\n".join(lines)
