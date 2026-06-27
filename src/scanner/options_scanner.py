import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import httpx

from src.data.alpaca_provider import get_alpaca_provider

logger = logging.getLogger(__name__)

TRADIER_SANDBOX_BASE = "https://sandbox.tradier.com/v1"
TRADIER_PROD_BASE = "https://api.tradier.com/v1"


@dataclass
class OptionCandidate:
    ticker: str
    contract_symbol: str
    expiration: str
    strike: float
    option_type: str  # "call" or "put"
    bid: float
    ask: float
    mid: float
    last: float
    volume: int
    open_interest: int
    delta: float
    gamma: float
    theta: float
    implied_volatility: float
    spread_pct: float


class OptionsScanner:
    """Pulls options chains from Tradier and filters by configured criteria."""

    def __init__(self, config: dict):
        self.api_key = os.environ.get("TRADIER_API_KEY", "")
        self.base_url = TRADIER_SANDBOX_BASE if not self.api_key else TRADIER_PROD_BASE
        if not self.api_key:
            self.base_url = TRADIER_SANDBOX_BASE

        opts = config.get("options", {})
        self.min_oi = opts.get("min_open_interest", 500)
        self.min_volume = opts.get("min_volume", 100)
        self.max_dte = opts.get("max_dte", 14)
        self.min_dte = opts.get("min_dte", 0)
        self.max_spread_pct = opts.get("max_spread_pct", 10)
        self.preferred_delta = opts.get("preferred_delta_range", [0.25, 0.50])
        self.max_price = opts.get("max_contract_price", 5.00)
        self._alpaca = get_alpaca_provider()

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

    def scan_options(
        self, ticker: str, direction: str = "bullish"
    ) -> list[OptionCandidate]:
        """Get filtered option candidates for a ticker/direction.

        Provider preference: Alpaca (real-time) > Tradier (paid) > yfinance.
        """
        # Try Alpaca first (free real-time with paper account, no SSN).
        if self._alpaca.enabled:
            try:
                cands = self._alpaca_scan(ticker, direction)
                if cands:
                    return cands
            except Exception:
                logger.exception("Alpaca options scan failed for %s; falling back.", ticker)

        if self.api_key:
            try:
                expirations = self._get_expirations(ticker)
                candidates = []
                for exp in expirations:
                    chain = self._get_chain(ticker, exp, direction)
                    candidates.extend(chain)
                candidates.sort(
                    key=lambda c: abs(c.delta) if c.delta else 0, reverse=True
                )
                if candidates:
                    return candidates[:3]
            except Exception:
                logger.exception("Failed to scan options for %s via Tradier.", ticker)

        result = self._yfinance_fallback(ticker, direction)
        if not result:
            import time
            time.sleep(2)
            result = self._yfinance_fallback(ticker, direction)
        return result

    def _alpaca_scan(self, ticker: str, direction: str) -> list[OptionCandidate]:
        from src.scanner.greeks import compute_greeks
        option_type = "call" if direction == "bullish" else "put"
        today = date.today()

        # Pull a few near-term expirations and merge.
        candidates: list[OptionCandidate] = []
        # Alpaca returns the chain for whatever expiration we ask for; we'll
        # query days within [min_dte, max_dte] and pick the closest few.
        # Dense coverage across the swing window (weekly + biweekly expiries);
        # non-expiry/weekend dates return empty and are skipped cheaply.
        target_dtes = sorted({d for d in (4, 5, 6, 7, 8, 9, 10, 11, 12, 14) if self.min_dte <= d <= self.max_dte})
        for dte in target_dtes:
            exp = today + timedelta(days=dte)
            # Skip weekends — equities don't expire on Sat/Sun
            if exp.weekday() >= 5:
                continue
            snaps = self._alpaca.get_option_chain(ticker, expiration=exp, option_type=option_type)
            if not snaps:
                continue
            # Need underlying price for Greeks fallback if Alpaca didn't include them.
            underlying_price = self._fetch_underlying_price(ticker)
            for s in snaps:
                bid = s.bid
                ask = s.ask
                mid = s.mid if s.mid > 0 else max(s.last, 0)
                if mid <= 0:
                    continue
                spread_pct = ((ask - bid) / mid * 100) if mid > 0 else 100
                delta = s.delta
                gamma = s.gamma
                theta = s.theta
                iv = s.implied_volatility
                if (delta == 0 or iv == 0) and underlying_price > 0:
                    g = compute_greeks(underlying_price, s.strike, dte or 0.25, iv or 0.30, option_type)
                    delta = delta or g["delta"]
                    gamma = gamma or g["gamma"]
                    theta = theta or g["theta"]
                cand = OptionCandidate(
                    ticker=ticker,
                    contract_symbol=s.symbol,
                    expiration=s.expiration,
                    strike=s.strike,
                    option_type=option_type,
                    bid=bid,
                    ask=ask,
                    mid=round(mid, 2),
                    last=s.last,
                    volume=s.volume,
                    open_interest=s.open_interest,
                    delta=delta,
                    gamma=gamma,
                    theta=theta,
                    implied_volatility=iv,
                    spread_pct=round(spread_pct, 2),
                )
                if self._passes_filter_lenient(cand):
                    candidates.append(cand)

        # Rank: tight spread + reasonable delta + price in budget.
        def _score(c: OptionCandidate) -> float:
            d = abs(c.delta) if c.delta else 0
            delta_fit = 1.0 - min(abs(d - 0.35), 0.35)
            spread_pen = max(0, 1 - (c.spread_pct or 0) / 20.0)
            return delta_fit * 50 + spread_pen * 30 + (1 if c.mid > 0.20 else 0) * 5
        candidates.sort(key=_score, reverse=True)
        return candidates[:3]

    def _passes_filter_lenient(self, c: OptionCandidate) -> bool:
        """Same checks as _passes_filter but allows missing volume/OI
        (Alpaca's snapshot doesn't always carry them)."""
        if c.spread_pct > self.max_spread_pct:
            return False
        if c.mid <= 0 or c.mid > self.max_price:
            return False
        delta_abs = abs(c.delta) if c.delta else 0
        if delta_abs == 0:
            return False
        if not (self.preferred_delta[0] <= delta_abs <= self.preferred_delta[1]):
            return False
        return True

    def _fetch_underlying_price(self, ticker: str) -> float:
        """Best-effort spot price for Greek computations."""
        if self._alpaca.enabled:
            df = self._alpaca.get_daily_bars(ticker, lookback_days=3)
            if df is not None and not df.empty:
                col = "close" if "close" in df.columns else "Close"
                return float(df[col].iloc[-1])
        try:
            import yfinance as yf
            tk = yf.Ticker(ticker)
            hist = tk.history(period="2d")
            if hist is not None and not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass
        return 0.0

    def _get_expirations(self, ticker: str) -> list[str]:
        with httpx.Client() as client:
            resp = client.get(
                f"{self.base_url}/markets/options/expirations",
                headers=self._headers(),
                params={"symbol": ticker, "includeAllRoots": "true"},
            )
            resp.raise_for_status()
            data = resp.json()

        expirations = data.get("expirations", {}).get("date", [])
        if isinstance(expirations, str):
            expirations = [expirations]

        today = date.today()
        filtered = []
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if self.min_dte <= dte <= self.max_dte:
                filtered.append(exp_str)
        return filtered

    def _get_chain(
        self, ticker: str, expiration: str, direction: str
    ) -> list[OptionCandidate]:
        option_type = "call" if direction == "bullish" else "put"
        with httpx.Client() as client:
            resp = client.get(
                f"{self.base_url}/markets/options/chains",
                headers=self._headers(),
                params={
                    "symbol": ticker,
                    "expiration": expiration,
                    "greeks": "true",
                    "type": option_type,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        options = data.get("options", {}).get("option", [])
        if isinstance(options, dict):
            options = [options]

        candidates = []
        for opt in options:
            candidate = self._parse_option(ticker, opt)
            if candidate and self._passes_filter(candidate):
                candidates.append(candidate)
        return candidates

    def _parse_option(self, ticker: str, opt: dict) -> OptionCandidate | None:
        bid = opt.get("bid", 0) or 0
        ask = opt.get("ask", 0) or 0
        mid = (bid + ask) / 2 if (bid + ask) > 0 else 0

        greeks = opt.get("greeks", {}) or {}
        spread_pct = ((ask - bid) / mid * 100) if mid > 0 else 100

        return OptionCandidate(
            ticker=ticker,
            contract_symbol=opt.get("symbol", ""),
            expiration=opt.get("expiration_date", ""),
            strike=opt.get("strike", 0),
            option_type=opt.get("option_type", ""),
            bid=bid,
            ask=ask,
            mid=round(mid, 2),
            last=opt.get("last", 0) or 0,
            volume=opt.get("volume", 0) or 0,
            open_interest=opt.get("open_interest", 0) or 0,
            delta=greeks.get("delta", 0) or 0,
            gamma=greeks.get("gamma", 0) or 0,
            theta=greeks.get("theta", 0) or 0,
            implied_volatility=greeks.get("mid_iv", 0) or 0,
            spread_pct=round(spread_pct, 2),
        )

    def _passes_filter(self, c: OptionCandidate) -> bool:
        if c.open_interest < self.min_oi:
            return False
        if c.volume < self.min_volume:
            return False
        if c.spread_pct > self.max_spread_pct:
            return False
        if c.mid > self.max_price:
            return False
        delta_abs = abs(c.delta)
        if not (self.preferred_delta[0] <= delta_abs <= self.preferred_delta[1]):
            return False
        return True

    def _yfinance_fallback(
        self, ticker: str, direction: str
    ) -> list[OptionCandidate]:
        """Fallback using yfinance with locally computed Greeks."""
        try:
            import yfinance as yf
            from .greeks import compute_greeks

            tk = yf.Ticker(ticker)
            expirations = tk.options
            if not expirations:
                return []

            hist = tk.history(period="2d")
            if hist.empty:
                logger.warning("No price history for %s", ticker)
                return []
            current_price = float(hist["Close"].iloc[-1])

            today = date.today()
            valid_exps = []
            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if self.min_dte <= dte <= self.max_dte:
                    valid_exps.append((exp_str, dte))

            candidates = []
            option_type = "call" if direction == "bullish" else "put"

            for exp, dte in valid_exps[:4]:
                chain = tk.option_chain(exp)
                opts_df = chain.calls if direction == "bullish" else chain.puts
                if opts_df.empty:
                    continue

                for _, row in opts_df.iterrows():
                    import math
                    bid = float(row.get("bid", 0) or 0)
                    ask = float(row.get("ask", 0) or 0)
                    if math.isnan(bid): bid = 0
                    if math.isnan(ask): ask = 0
                    mid = (bid + ask) / 2 if (bid + ask) > 0 else 0
                    spread_pct = ((ask - bid) / mid * 100) if mid > 0 else 100
                    raw_iv = row.get("impliedVolatility", 0)
                    iv = float(raw_iv) if raw_iv and not (isinstance(raw_iv, float) and math.isnan(raw_iv)) else 0
                    strike = float(row.get("strike", 0))
                    raw_vol = row.get("volume", 0)
                    volume = int(raw_vol) if raw_vol and not (isinstance(raw_vol, float) and math.isnan(raw_vol)) else 0
                    raw_oi = row.get("openInterest", 0)
                    oi = int(raw_oi) if raw_oi and not (isinstance(raw_oi, float) and math.isnan(raw_oi)) else 0
                    raw_last = row.get("lastPrice", 0)
                    last = float(raw_last) if raw_last and not (isinstance(raw_last, float) and math.isnan(raw_last)) else 0

                    greeks_data = compute_greeks(current_price, strike, dte, iv, option_type)

                    c = OptionCandidate(
                        ticker=ticker,
                        contract_symbol=row.get("contractSymbol", ""),
                        expiration=exp,
                        strike=strike,
                        option_type=option_type,
                        bid=bid,
                        ask=ask,
                        mid=round(mid, 2),
                        last=last,
                        volume=volume,
                        open_interest=oi,
                        delta=greeks_data["delta"],
                        gamma=greeks_data["gamma"],
                        theta=greeks_data["theta"],
                        implied_volatility=iv,
                        spread_pct=round(spread_pct, 2),
                    )
                    if self._passes_filter(c):
                        candidates.append(c)

            candidates.sort(
                key=lambda c: (c.volume * 0.4 + c.open_interest * 0.3 + (1 - c.spread_pct / 10) * 30),
                reverse=True,
            )
            return candidates[:3]
        except Exception:
            logger.exception("yfinance options fallback failed for %s", ticker)
            return []
