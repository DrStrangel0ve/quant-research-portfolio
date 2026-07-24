"""Time-series momentum and volatility-targeting signals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def time_series_momentum_positions(
    prices: pd.DataFrame,
    *,
    signal_lookback: int = 126,
    volatility_lookback: int = 20,
    annual_volatility_target: float = 0.12,
    periods_per_year: int = 252,
    leverage_cap: float = 2.0,
) -> pd.DataFrame:
    """Direction from trailing returns, sized to an ex-ante volatility target."""
    if signal_lookback < 2 or volatility_lookback < 2:
        raise ValueError("lookbacks must be at least two")
    if annual_volatility_target <= 0.0 or leverage_cap <= 0.0:
        raise ValueError("risk target and leverage cap must be positive")
    returns = prices.pct_change(fill_method=None)
    momentum = prices.pct_change(signal_lookback, fill_method=None)
    direction = pd.DataFrame(
        np.sign(momentum.to_numpy()),
        index=prices.index,
        columns=prices.columns,
    )
    annualized_vol = returns.rolling(volatility_lookback).std(ddof=1)
    annualized_vol *= np.sqrt(periods_per_year)
    scale = (annual_volatility_target / annualized_vol.clip(lower=1e-6)).clip(
        upper=leverage_cap
    )
    positions = pd.DataFrame(
        direction.to_numpy() * scale.to_numpy() / prices.shape[1],
        index=prices.index,
        columns=prices.columns,
    )
    return positions.replace([np.inf, -np.inf], np.nan).fillna(0.0)
