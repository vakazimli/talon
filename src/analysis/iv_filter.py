"""IV richness filter — rejects setups where options are overpriced
relative to the underlying's recent historical volatility.

Without a paid IV-history feed we approximate with HV20: 20-day
realized volatility of daily log returns. Implied vol > 1.5 * HV20 is
"elevated" — premium too rich for the directional move we expect.
This is a coarse heuristic but catches the worst overpriced setups.
"""

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# IV/HV ratios above this are "elevated" — penalize or reject.
IV_HV_REJECT_RATIO = 1.8
IV_HV_PENALTY_RATIO = 1.4
PENALTY_POINTS = 12.0


@dataclass
class IVCheck:
    iv: float
    hv20: float
    ratio: float
    elevated: bool
    rich: bool          # true if extremely rich (rejection-worthy)


def hv20_from_returns(closes) -> float:
    """20-day annualized historical volatility from a closing-price series.

    `closes` may be a pandas Series, list, or any iterable of floats.
    Returns 0 if insufficient data.
    """
    try:
        import pandas as pd
        s = pd.Series(closes).dropna()
        if len(s) < 21:
            return 0.0
        log_ret = (s.pct_change().add(1).apply(math.log)).dropna().tail(20)
        if len(log_ret) < 5:
            return 0.0
        daily_std = float(log_ret.std())
        return daily_std * math.sqrt(252)
    except Exception:
        logger.exception("hv20 computation failed")
        return 0.0


def evaluate_iv(best_iv: float, hv20: float) -> IVCheck:
    """Compare implied vol to historical vol and classify."""
    if hv20 <= 0 or best_iv <= 0:
        return IVCheck(iv=best_iv, hv20=hv20, ratio=0.0, elevated=False, rich=False)
    ratio = best_iv / hv20
    elevated = ratio >= IV_HV_PENALTY_RATIO
    rich = ratio >= IV_HV_REJECT_RATIO
    return IVCheck(iv=best_iv, hv20=hv20, ratio=ratio, elevated=elevated, rich=rich)
