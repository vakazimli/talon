"""Vision-based chart analyzer.

Renders a hardened, annotated, multi-timeframe chart and asks Claude
Opus 4.7 (or any vision-capable model) to judge whether it's a tradeable
short-term setup.

Why dual-pane?
  Single-timeframe charts overfit to noise. By giving the model both a
  daily-bar context (~3 months) and a recent intraday view (~3 days at
  15-minute bars) in one image, it can spot daily-vs-intraday divergence
  (e.g. clean daily uptrend, sloppy intraday rollover) that a single
  pane would miss.

Why annotated?
  Drawing EMA20 / EMA50 / Bollinger bands and an RSI sub-pane on the
  chart means the model reads the actual technical state visually —
  it is not asked to derive RSI from raw bars (which it can't).

Hardening:
  * Tight system prompt: no price invention, reason before verdict,
    default to PASS on uncertainty.
  * Post-validation cross-checks the model's key_levels against the
    numerical technicals we already have. Mismatched levels downgrade
    chart_score and surface a 'inconsistency' note.

Disabled by default in `config/settings.yaml` because vision requests
are ~$0.05 each. Enable via `chart_analyst.enabled: true`.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

VISION_SYSTEM_PROMPT = (
    "You are a disciplined short-term options day trader looking at a "
    "candlestick chart. Your default answer is PASS unless the setup "
    "is clean and obvious. Be honest. Do NOT invent price levels — only "
    "reference levels visible on the chart or already provided in the "
    "technical context. If you can't read the chart for any reason, "
    "return chart_score=0 and verdict=PASS. Respond ONLY with valid JSON, "
    "no markdown fences, no commentary."
)

VISION_PROMPT_TEMPLATE = """Two charts of {ticker} are below: a 3-month DAILY chart on the left,
and a recent INTRADAY chart on the right. EMA20 (orange) and EMA50 (red)
are drawn on both panes. Bollinger Bands shaded on daily. RSI(14) shown in the lower sub-pane on each.

Setup we're evaluating:
  Direction: {direction}
  Setup type: {setup_type}
  Numerical technicals (use these as the source of truth for prices):
    Last close: {price}
    RSI: {rsi}
    EMA20: {ema20}   EMA50: {ema50}
    VWAP: {vwap}
    BB upper: {bb_upper}    BB lower: {bb_lower}
    Volume z-score: {vol_zscore}

Step 1: REASONING — list 3-5 short observations from the chart.
Step 2: Identify support and resistance ONLY from levels visible on the
        chart and reasonably aligned with the numerical technicals above.
        If you can't, set them to null.
Step 3: Verdict.

Schema (return exactly this, JSON only, no markdown):
{{
  "observations": ["<obs1>", "<obs2>", "<obs3>"],
  "pattern": "<short label e.g. 'flag breakout', 'choppy range', 'failed retest'>",
  "key_levels": {{
    "support": <number or null>,
    "resistance": <number or null>
  }},
  "daily_intraday_aligned": <true|false>,
  "chart_score": <int 0-100, where 100 = textbook setup>,
  "concerns": "<one sentence: what could go wrong>",
  "verdict": "TAKE" | "WAIT" | "PASS"
}}

Rules:
- Default to PASS when in doubt or chart looks choppy/range-bound.
- Set verdict=PASS if daily and intraday disagree (e.g. daily uptrend but intraday rolling over against the proposed direction).
- chart_score must be <= 50 unless the setup is clean."""


@dataclass
class VisionAnalysis:
    chart_score: int
    pattern: str
    key_levels: dict
    daily_intraday_aligned: bool
    observations: list = field(default_factory=list)
    concerns: str = ""
    verdict: str = "PASS"
    raw: dict = field(default_factory=dict)
    inconsistencies: list = field(default_factory=list)


def _normalize_ohlcv(df):
    """Return a DataFrame with Title-case OHLCV columns + DatetimeIndex."""
    import pandas as pd
    if df is None or df.empty:
        return None
    cols = {c.lower(): c for c in df.columns}
    needed = ["open", "high", "low", "close", "volume"]
    for col in needed:
        if col not in cols:
            return None
    out = df.copy()
    rename = {}
    for col in needed:
        target = col.capitalize()
        if cols[col] != target:
            rename[cols[col]] = target
    if rename:
        out = out.rename(columns=rename)
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    return out[["Open", "High", "Low", "Close", "Volume"]]


def render_dual_chart_png(
    daily_df,
    intraday_df,
    ticker: str,
    title_extra: str = "",
    figsize: tuple[float, float] = (16, 9),
    dpi: int = 130,
) -> bytes:
    """Render a side-by-side annotated chart: daily on the left,
    intraday on the right. EMA20/EMA50, Bollinger Bands on daily, and an
    RSI(14) sub-pane on each. Returns PNG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplfinance as mpf
    import pandas as pd

    daily = _normalize_ohlcv(daily_df)
    intraday = _normalize_ohlcv(intraday_df)
    if daily is None and intraday is None:
        raise ValueError("Both daily and intraday DataFrames are empty")
    if daily is None:
        daily = intraday
    if intraday is None:
        intraday = daily

    style = mpf.make_mpf_style(
        base_mpf_style="charles",
        rc={"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 9},
    )

    fig, axes = plt.subplots(
        2, 2,
        figsize=figsize,
        dpi=dpi,
        gridspec_kw={"height_ratios": [3, 1], "width_ratios": [1, 1]},
    )

    def _plot_pane(df, price_ax, rsi_ax, title, with_bb=True):
        addplots = []
        if len(df) >= 20:
            ema20 = df["Close"].ewm(span=20, adjust=False).mean()
            addplots.append(mpf.make_addplot(ema20, ax=price_ax, color="#ff8800", width=1.0))
        if len(df) >= 50:
            ema50 = df["Close"].ewm(span=50, adjust=False).mean()
            addplots.append(mpf.make_addplot(ema50, ax=price_ax, color="#cc0000", width=1.0))
        if with_bb and len(df) >= 20:
            ma = df["Close"].rolling(20).mean()
            std = df["Close"].rolling(20).std()
            upper = ma + 2 * std
            lower = ma - 2 * std
            addplots.append(mpf.make_addplot(upper, ax=price_ax, color="#888888", width=0.6, linestyle="--"))
            addplots.append(mpf.make_addplot(lower, ax=price_ax, color="#888888", width=0.6, linestyle="--"))
        # RSI in its own axis below the candle plot
        if len(df) >= 15:
            delta = df["Close"].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, 1e-9)
            rsi = 100 - (100 / (1 + rs))
            addplots.append(mpf.make_addplot(rsi, ax=rsi_ax, color="#3355cc", width=1.0))

        mpf.plot(
            df,
            type="candle",
            style=style,
            volume=False,
            addplot=addplots if addplots else None,
            ax=price_ax,
            axtitle=title,
            tight_layout=True,
        )
        rsi_ax.set_ylim(0, 100)
        rsi_ax.axhline(70, color="#bb0000", linewidth=0.6, linestyle=":")
        rsi_ax.axhline(30, color="#00aa00", linewidth=0.6, linestyle=":")
        rsi_ax.set_ylabel("RSI", fontsize=8)
        rsi_ax.grid(True, axis="y", alpha=0.3)

    _plot_pane(daily, axes[0][0], axes[1][0],
               title=f"{ticker} DAILY (3mo)  {title_extra}", with_bb=True)
    _plot_pane(intraday, axes[0][1], axes[1][1],
               title=f"{ticker} INTRADAY (recent)", with_bb=False)

    fig.suptitle(f"{ticker} {title_extra}".strip(), fontsize=12, y=0.995)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def render_chart_png(
    df,
    ticker: str,
    title_extra: str = "",
    ema_periods: tuple[int, ...] = (20, 50),
    figsize: tuple[float, float] = (10, 6),
) -> bytes:
    """Backwards-compatible single-pane chart. Used for testing and as
    fallback when only one timeframe is available."""
    return render_dual_chart_png(df, df, ticker, title_extra, figsize, dpi=120)


def _validate_levels(analysis: dict, technicals: dict, last_price: float) -> list[str]:
    """Cross-check the model's key_levels against numerical technicals.

    Hallucinated price levels are the #1 vision-LLM failure mode. We
    require reported support/resistance to be (a) within 5% of last
    price and (b) within 3% of at least one of {ema20, ema50, vwap,
    bb_lower, bb_upper}. Anything else gets logged and surfaced.
    """
    issues: list[str] = []
    if not last_price or last_price <= 0:
        return issues

    anchors = []
    for k in ("ema20", "ema50", "vwap", "bb_lower", "bb_upper"):
        v = technicals.get(k)
        if isinstance(v, (int, float)) and v > 0:
            anchors.append((k, float(v)))

    levels = analysis.get("key_levels") or {}
    for label in ("support", "resistance"):
        lvl = levels.get(label)
        if lvl is None:
            continue
        try:
            lvl_f = float(lvl)
        except (TypeError, ValueError):
            continue
        # within 5% of last price?
        if abs(lvl_f - last_price) / last_price > 0.05:
            issues.append(
                f"{label} ${lvl_f:.2f} is more than 5% from last close ${last_price:.2f}"
            )
            continue
        # near any technical anchor (3%)?
        if anchors and not any(abs(lvl_f - a) / a < 0.03 for _, a in anchors):
            anchor_str = ", ".join(f"{n}={a:.2f}" for n, a in anchors)
            issues.append(
                f"{label} ${lvl_f:.2f} not near any technical anchor ({anchor_str})"
            )
    return issues


class VisionAnalyst:
    """Orchestrates chart rendering + vision LLM call + validation."""

    def __init__(self, orchestrator, config: dict | None = None):
        self.orchestrator = orchestrator
        cfg = (config or {}).get("chart_analyst", {}) if isinstance(config, dict) else {}
        self.enabled = bool(cfg.get("enabled", False))

    def analyze(
        self,
        ticker: str,
        direction: str,
        setup_type: str,
        df_for_chart,
        technicals: dict,
        daily_df=None,
    ) -> Optional[VisionAnalysis]:
        if not self.enabled:
            return None
        try:
            png = render_dual_chart_png(
                daily_df if daily_df is not None else df_for_chart,
                df_for_chart,
                ticker,
                title_extra=f"({setup_type}/{direction})",
            )
        except Exception:
            logger.exception("Chart rendering failed for %s", ticker)
            return None

        b64 = base64.b64encode(png).decode("ascii")
        last_price = 0.0
        try:
            last_price = float(df_for_chart["close"].iloc[-1]) if "close" in df_for_chart.columns \
                else float(df_for_chart["Close"].iloc[-1])
        except Exception:
            pass

        prompt = VISION_PROMPT_TEMPLATE.format(
            ticker=ticker,
            direction=direction,
            setup_type=setup_type,
            price=f"{last_price:.2f}" if last_price else "?",
            rsi=technicals.get("rsi", "?"),
            ema20=technicals.get("ema20", "?"),
            ema50=technicals.get("ema50", "?"),
            vwap=technicals.get("vwap", "?"),
            bb_upper=technicals.get("bb_upper", "?"),
            bb_lower=technicals.get("bb_lower", "?"),
            vol_zscore=technicals.get("volume_zscore", "?"),
        )

        result = self.orchestrator.call_model_vision(
            "chart_analyst",
            text_prompt=prompt,
            image_b64_png=b64,
            system_prompt=VISION_SYSTEM_PROMPT,
        )
        if result is None or not result.content:
            return None

        text = result.content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.error("Vision analyst returned non-JSON: %s", text[:200])
            return None

        chart_score = int(data.get("chart_score") or 0)
        verdict = str(data.get("verdict") or "PASS").upper()

        # Cross-validate model price levels against numerical technicals.
        # Hallucinated levels downgrade the score; egregious mismatches
        # flip the verdict to PASS.
        issues = _validate_levels(data, technicals, last_price)
        if issues:
            logger.warning(
                "Vision %s key_levels issues: %s", ticker, "; ".join(issues)
            )
            penalty = 8 * len(issues)
            chart_score = max(0, chart_score - penalty)
            if len(issues) >= 2 and verdict == "TAKE":
                logger.info(
                    "Vision %s: 2+ key-level inconsistencies; downgrading TAKE -> WAIT.",
                    ticker,
                )
                verdict = "WAIT"

        # Daily-vs-intraday disagreement is itself a strong PASS signal.
        if data.get("daily_intraday_aligned") is False and verdict == "TAKE":
            logger.info(
                "Vision %s: daily/intraday disagreement; downgrading TAKE -> WAIT.",
                ticker,
            )
            verdict = "WAIT"

        return VisionAnalysis(
            chart_score=chart_score,
            pattern=str(data.get("pattern") or ""),
            key_levels=data.get("key_levels") or {},
            daily_intraday_aligned=bool(data.get("daily_intraday_aligned", True)),
            observations=list(data.get("observations") or []),
            concerns=str(data.get("concerns") or ""),
            verdict=verdict,
            raw=data,
            inconsistencies=issues,
        )
