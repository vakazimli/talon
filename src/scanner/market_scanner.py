import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import ta

from src.data.alpaca_provider import get_alpaca_provider

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    ticker: str
    price: float
    volume_vs_avg: float
    daily_change_pct: float
    signals: list[str] = field(default_factory=list)
    technicals: dict = field(default_factory=dict)
    priority: str = "low"


class MarketScanner:
    """Pulls market data via yfinance and computes technicals locally using ta."""

    def __init__(self, config: dict):
        self.options_cfg = config.get("options", {})
        self.scanning_cfg = config.get("scanning", {})
        self._alpaca = get_alpaca_provider()
        # Lazily-initialized: NewsScanner reads FINNHUB_API_KEY itself.
        self._news_scanner = None

    def _news(self):
        if self._news_scanner is None:
            from src.scanner.news_scanner import NewsScanner
            self._news_scanner = NewsScanner({})
        return self._news_scanner

    def _fetch_daily_history(self, ticker: str):
        """Daily OHLCV history.

        Provider chain:
          1. Alpaca daily bars (real-time, requires ID-verified account)
          2. yfinance (15-min delayed, no ID required)
          3. Finnhub real-time quote patches the latest bar so technicals
             reflect *current* price even when the rest of the history
             is delayed.
        """
        df = None
        if self._alpaca.enabled:
            df = self._alpaca.get_daily_bars(ticker, lookback_days=90)
            if df is not None and not df.empty and len(df) >= 20:
                return df
        try:
            import yfinance as yf
            tk = yf.Ticker(ticker)
            hist = tk.history(period="3mo", interval="1d")
            if hist is None or hist.empty:
                return None
            df = hist
        except Exception:
            logger.exception("yfinance fallback failed for %s", ticker)
            return None

        # Patch latest bar's close with Finnhub real-time quote when available.
        # yfinance daily bars lag ~15 min; this gives us the freshest close
        # without paying for a real-time stocks plan.
        if df is not None and not df.empty:
            try:
                quote = self._news().get_quote(ticker)
                if quote and quote.get("current", 0) > 0:
                    last_idx = df.index[-1]
                    close_col = "Close" if "Close" in df.columns else "close"
                    high_col = "High" if "High" in df.columns else "high"
                    low_col = "Low" if "Low" in df.columns else "low"
                    # Sanity guard: a real-time quote that disagrees with the
                    # last daily close by more than 20% is almost certainly a
                    # bad tick or wrong symbol — don't let it drive signals.
                    prev_close = float(df.loc[last_idx, close_col])
                    new_close = float(quote["current"])
                    if prev_close > 0 and abs(new_close / prev_close - 1) > 0.20:
                        logger.warning(
                            "%s: rejecting real-time patch %.2f vs daily close %.2f "
                            "(>20%% gap, likely bad data).",
                            ticker, new_close, prev_close,
                        )
                        return df
                    df.loc[last_idx, close_col] = quote["current"]
                    if quote.get("high", 0) > 0:
                        df.loc[last_idx, high_col] = max(
                            float(df.loc[last_idx, high_col]), quote["high"]
                        )
                    if quote.get("low", 0) > 0 and quote["low"] < float(df.loc[last_idx, low_col]):
                        df.loc[last_idx, low_col] = quote["low"]
            except Exception:
                logger.debug("Finnhub quote patch skipped for %s", ticker)

        return df

    def scan_tickers(self, tickers: list[str]) -> list[ScanResult]:
        results = []
        for ticker in tickers:
            try:
                result = self._scan_single(ticker)
                if result:
                    results.append(result)
            except Exception:
                logger.exception("Failed to scan %s", ticker)
        results.sort(key=lambda r: len(r.signals), reverse=True)
        return results

    def _scan_single(self, ticker: str) -> ScanResult | None:
        hist = self._fetch_daily_history(ticker)
        if hist is None or hist.empty or len(hist) < 20:
            logger.warning("Insufficient data for %s", ticker)
            return None

        close = hist["close"] if "close" in hist.columns else hist["Close"]
        volume = hist["volume"] if "volume" in hist.columns else hist["Volume"]
        high = hist["high"] if "high" in hist.columns else hist["High"]
        low = hist["low"] if "low" in hist.columns else hist["Low"]
        current_price = float(close.iloc[-1])

        technicals = self._compute_technicals(close, high, low, volume)
        signals = self._detect_signals(close, high, low, volume, technicals)

        # Real intraday session VWAP (Alpaca intraday bars). Fills the
        # technicals["vwap"] field and emits a proper vwap_reclaim signal
        # when price crosses back above the session VWAP.
        session = self._intraday_session_metrics(ticker)
        if session is not None:
            technicals["vwap"] = round(session["vwap"], 2)
            technicals["price_vs_vwap_pct"] = round(session["price_vs_vwap_pct"], 2)
            if session["reclaim"] and "vwap_reclaim" not in signals:
                signals.append("vwap_reclaim")

        avg_vol_20 = float(volume.iloc[-20:].mean())
        vol_ratio = float(volume.iloc[-1] / avg_vol_20) if avg_vol_20 > 0 else 0.0
        daily_change = float((close.iloc[-1] / close.iloc[-2] - 1) * 100) if len(close) > 1 else 0.0

        priority = "low"
        if len(signals) >= 3:
            priority = "high"
        elif len(signals) >= 2:
            priority = "medium"

        return ScanResult(
            ticker=ticker,
            price=current_price,
            volume_vs_avg=round(vol_ratio, 2),
            daily_change_pct=round(daily_change, 2),
            signals=signals,
            technicals=technicals,
            priority=priority,
        )

    def _intraday_session_metrics(self, ticker: str) -> dict | None:
        """Compute the current trading session's VWAP from intraday bars.

        Returns {"vwap", "last", "price_vs_vwap_pct", "reclaim", "above_vwap"}
        or None when intraday data isn't available (e.g. Alpaca disabled).
        `reclaim` is True when price traded below VWAP earlier in the session
        and is now back above it — a genuine intraday VWAP reclaim.
        """
        if not self._alpaca.enabled:
            return None
        try:
            from zoneinfo import ZoneInfo
            df = self._alpaca.get_intraday_bars(ticker, minutes=15, lookback_hours=8)
            if df is None or df.empty:
                return None
            idx = pd.to_datetime(df.index)
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            et_idx = idx.tz_convert(ZoneInfo("America/New_York"))
            last_date = et_idx[-1].date()
            mask = [d == last_date for d in et_idx.date]
            sess = df[mask]
            if sess is None or sess.empty:
                return None
            close_col = "close" if "close" in sess.columns else "Close"
            high_col = "high" if "high" in sess.columns else "High"
            low_col = "low" if "low" in sess.columns else "Low"
            vol_col = "volume" if "volume" in sess.columns else "Volume"
            typical = (sess[high_col] + sess[low_col] + sess[close_col]) / 3
            cum_vol = sess[vol_col].cumsum()
            if float(cum_vol.iloc[-1]) <= 0:
                return None
            vwap_series = (typical * sess[vol_col]).cumsum() / cum_vol
            vwap_now = float(vwap_series.iloc[-1])
            last = float(sess[close_col].iloc[-1])
            if vwap_now <= 0:
                return None
            below_earlier = bool((sess[close_col].iloc[:-1] < vwap_series.iloc[:-1]).any())
            return {
                "vwap": vwap_now,
                "last": last,
                "price_vs_vwap_pct": (last / vwap_now - 1) * 100,
                "reclaim": last > vwap_now and below_earlier,
                "above_vwap": last > vwap_now,
            }
        except Exception:
            logger.debug("Intraday session VWAP failed for %s", ticker, exc_info=True)
            return None

    def _compute_technicals(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
    ) -> dict:
        ema20 = ta.trend.ema_indicator(close, window=20)
        ema50 = ta.trend.ema_indicator(close, window=50)
        rsi = ta.momentum.rsi(close, window=14)
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)

        # ATR(14) for volatility-aware exit sizing.
        atr14 = 0.0
        atr_pct = 0.0
        try:
            atr_series = ta.volatility.AverageTrueRange(
                high=high, low=low, close=close, window=14
            ).average_true_range()
            last_atr = float(atr_series.iloc[-1])
            last_close = float(close.iloc[-1])
            if not np.isnan(last_atr) and last_close > 0:
                atr14 = round(last_atr, 4)
                atr_pct = round(last_atr / last_close, 4)
        except Exception:
            pass

        avg_vol_20 = volume.rolling(window=20).mean()
        vol_zscore = (
            (volume - avg_vol_20) / volume.rolling(window=20).std()
        )

        return {
            "ema20": round(float(ema20.iloc[-1]), 2),
            "ema50": round(float(ema50.iloc[-1]), 2),
            "rsi": round(float(rsi.iloc[-1]), 2),
            # Real (intraday session) VWAP is filled in by _scan_single when
            # intraday bars are available; 0.0 means "unknown" and is ignored
            # downstream. A cumulative VWAP over months of daily bars is
            # meaningless, so we no longer compute it here.
            "vwap": 0.0,
            "bb_upper": round(float(bb.bollinger_hband().iloc[-1]), 2),
            "bb_lower": round(float(bb.bollinger_lband().iloc[-1]), 2),
            "atr14": atr14,
            "atr_pct": atr_pct,
            "volume_zscore": round(float(vol_zscore.iloc[-1]), 2) if not np.isnan(vol_zscore.iloc[-1]) else 0.0,
            "avg_volume_20d": int(avg_vol_20.iloc[-1]) if not np.isnan(avg_vol_20.iloc[-1]) else 0,
        }

    def _detect_signals(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        technicals: dict,
    ) -> list[str]:
        signals = []
        price = float(close.iloc[-1])
        prev_price = float(close.iloc[-2]) if len(close) > 1 else price

        # Volume spike: z-score > 2
        if technicals.get("volume_zscore", 0) > 2.0:
            signals.append("volume_spike")

        # EMA crossover: price crossed above EMA20 today
        ema20 = technicals.get("ema20", 0)
        if prev_price < ema20 <= price:
            signals.append("ema_crossover")

        # Breakout above resistance (20-day high)
        high_20 = float(high.iloc[-21:-1].max()) if len(high) > 21 else float(high.max())
        if price > high_20:
            signals.append("breakout_above_resistance")

        # Breakdown below support (20-day low)
        low_20 = float(low.iloc[-21:-1].min()) if len(low) > 21 else float(low.min())
        if price < low_20:
            signals.append("breakdown_below_support")

        # NOTE: vwap_reclaim is detected from intraday session VWAP in
        # _scan_single (a daily-bar "VWAP" is not a real VWAP), not here.

        # Squeeze breakout: Bollinger Band width contracting then expanding
        bb_upper = technicals.get("bb_upper", 0)
        bb_lower = technicals.get("bb_lower", 0)
        bb_width = (bb_upper - bb_lower) / price if price > 0 else 0
        if bb_width < 0.03 and (price > bb_upper or price < bb_lower):
            signals.append("squeeze_breakout")

        # RSI extremes
        rsi = technicals.get("rsi", 50)
        if rsi > 70:
            signals.append("rsi_overbought")
        elif rsi < 30:
            signals.append("rsi_oversold")

        # Gap fill detection: price gapped and is now filling
        if len(close) > 2:
            prev_close_2 = float(close.iloc[-3])
            gap_up = float(low.iloc[-2]) > prev_close_2
            gap_down = float(high.iloc[-2]) < prev_close_2
            if gap_up and price < float(low.iloc[-2]):
                signals.append("gap_fill")
            elif gap_down and price > float(high.iloc[-2]):
                signals.append("gap_fill")

        return signals
