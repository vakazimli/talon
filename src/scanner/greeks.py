"""Black-Scholes Greeks calculator. No external API needed."""

import math

from scipy.stats import norm

RISK_FREE_RATE = 0.045


def compute_greeks(
    underlying_price: float,
    strike: float,
    dte_days: int,
    iv: float,
    option_type: str = "call",
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:
    """Compute delta, gamma, theta from Black-Scholes.

    Args:
        underlying_price: Current price of the underlying
        strike: Option strike price
        dte_days: Days to expiration
        iv: Implied volatility (as decimal, e.g. 0.25 for 25%)
        option_type: "call" or "put"
        risk_free_rate: Annual risk-free rate
    """
    if dte_days <= 0:
        dte_days = 0.25  # quarter-day for 0DTE
    if iv <= 0 or underlying_price <= 0 or strike <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0}

    T = dte_days / 365.0
    S = underlying_price
    K = strike
    r = risk_free_rate
    sigma = iv

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option_type == "call":
        delta = norm.cdf(d1)
        theta_term = -S * norm.pdf(d1) * sigma / (2 * sqrt_T) - r * K * math.exp(-r * T) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta_term = -S * norm.pdf(d1) * sigma / (2 * sqrt_T) + r * K * math.exp(-r * T) * norm.cdf(-d2)

    gamma = norm.pdf(d1) / (S * sigma * sqrt_T)
    theta = theta_term / 365.0  # per-day theta

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
    }


def bs_price(
    underlying_price: float,
    strike: float,
    dte_days: float,
    iv: float,
    option_type: str = "call",
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """Black-Scholes price of a European option. Used by the backtest engine
    to value synthetic contracts (no historical options chain on free data).

    Falls back to intrinsic value when inputs are degenerate.
    """
    is_call = option_type == "call"
    if underlying_price <= 0 or strike <= 0:
        return 0.0
    if dte_days <= 0 or iv <= 0:
        intrinsic = (
            max(0.0, underlying_price - strike) if is_call
            else max(0.0, strike - underlying_price)
        )
        return round(intrinsic, 4)

    T = dte_days / 365.0
    S = underlying_price
    K = strike
    r = risk_free_rate
    sigma = iv
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if is_call:
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return round(max(price, 0.0), 4)
