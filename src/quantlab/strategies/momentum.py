"""Cross-sectional momentum signals with volatility-aware position sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_momentum_positions(
    prices: pd.DataFrame,
    *,
    lookback: int = 126,
    volatility_lookback: int = 20,
    selection_fraction: float = 0.2,
    target_gross: float = 1.0,
) -> pd.DataFrame:
    """Long recent winners and short losers with inverse-volatility sizing."""
    if lookback < 2 or volatility_lookback < 2:
        raise ValueError("lookbacks must be at least two")
    if not 0.0 < selection_fraction <= 0.5:
        raise ValueError("selection_fraction must be in (0, 0.5]")
    if target_gross <= 0.0:
        raise ValueError("target_gross must be positive")
    if prices.shape[1] < 2:
        raise ValueError("at least two assets are required")
    if (prices <= 0.0).any().any():
        raise ValueError("prices must be positive")

    returns = prices.pct_change(fill_method=None)
    momentum = prices.pct_change(lookback, fill_method=None)
    ranks = momentum.rank(axis=1, pct=True, method="average")
    long_mask = ranks >= (1.0 - selection_fraction)
    short_mask = ranks <= selection_fraction
    volatility = returns.rolling(volatility_lookback).std(ddof=1).clip(lower=1e-6)
    raw = long_mask.astype(float).div(volatility) - short_mask.astype(float).div(volatility)

    long_weights = raw.clip(lower=0.0)
    short_weights = -raw.clip(upper=0.0)
    long_weights = long_weights.div(long_weights.sum(axis=1).replace(0.0, np.nan), axis=0)
    short_weights = short_weights.div(short_weights.sum(axis=1).replace(0.0, np.nan), axis=0)
    positions = 0.5 * target_gross * (long_weights - short_weights)
    return positions.replace([np.inf, -np.inf], np.nan).fillna(0.0)
