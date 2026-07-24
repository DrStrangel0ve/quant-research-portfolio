"""Rolling pairs-trading research without full-sample parameter leakage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairsSignal:
    positions: pd.DataFrame
    hedge_ratio: pd.Series
    spread: pd.Series
    zscore: pd.Series


def rolling_pairs_signal(
    prices: pd.DataFrame,
    *,
    dependent: str,
    independent: str,
    hedge_lookback: int = 60,
    zscore_lookback: int = 20,
    entry_zscore: float = 2.0,
    exit_zscore: float = 0.5,
) -> PairsSignal:
    """Build a stateful mean-reversion signal using rolling OLS on log prices.

    The signal at time ``t`` uses data through ``t`` only. Feed these target
    positions to ``run_backtest`` with its default one-period execution lag.
    """
    if dependent not in prices or independent not in prices:
        raise KeyError("dependent and independent columns must be present")
    if hedge_lookback < 3 or zscore_lookback < 3:
        raise ValueError("lookbacks must be at least three")
    if not 0.0 <= exit_zscore < entry_zscore:
        raise ValueError("require 0 <= exit_zscore < entry_zscore")
    selected = prices[[dependent, independent]].astype(float)
    if (selected <= 0.0).any().any():
        raise ValueError("prices must be positive")

    log_prices = pd.DataFrame(
        np.log(selected.to_numpy()),
        index=selected.index,
        columns=selected.columns,
    )
    x = log_prices[independent]
    y = log_prices[dependent]
    covariance = x.rolling(hedge_lookback).cov(y)
    variance = x.rolling(hedge_lookback).var(ddof=1)
    hedge_ratio = (covariance / variance.replace(0.0, np.nan)).rename("hedge_ratio")
    rolling_x_mean = x.rolling(hedge_lookback).mean()
    rolling_y_mean = y.rolling(hedge_lookback).mean()
    intercept = rolling_y_mean - hedge_ratio * rolling_x_mean
    spread = (y - intercept - hedge_ratio * x).rename("spread")
    spread_mean = spread.rolling(zscore_lookback).mean()
    spread_std = spread.rolling(zscore_lookback).std(ddof=1).replace(0.0, np.nan)
    zscore = ((spread - spread_mean) / spread_std).rename("zscore")

    state = 0
    states: list[int] = []
    for value in zscore:
        if np.isnan(value):
            state = 0
        elif state == 0 and value >= entry_zscore:
            state = -1
        elif state == 0 and value <= -entry_zscore:
            state = 1
        elif state != 0 and abs(value) <= exit_zscore:
            state = 0
        states.append(state)

    state_series = pd.Series(states, index=prices.index, dtype=float)
    dependent_weight = state_series
    independent_weight = -state_series * hedge_ratio
    positions = pd.DataFrame(
        {dependent: dependent_weight, independent: independent_weight},
        index=prices.index,
    ).fillna(0.0)
    gross = positions.abs().sum(axis=1).replace(0.0, np.nan)
    positions = positions.div(gross, axis=0).fillna(0.0)
    return PairsSignal(
        positions=positions,
        hedge_ratio=hedge_ratio,
        spread=spread,
        zscore=zscore,
    )
