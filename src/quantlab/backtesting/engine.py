"""A transparent, vectorized multi-asset backtest engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.core.metrics import performance_summary
from quantlab.core.validation import (
    require_datetime_index,
    require_finite,
    require_same_shape_and_labels,
)


@dataclass(frozen=True)
class CostModel:
    """Linear execution costs, expressed in basis points of turnover."""

    commission_bps: float = 0.0
    half_spread_bps: float = 0.0
    slippage_bps: float = 0.0

    def __post_init__(self) -> None:
        values = (self.commission_bps, self.half_spread_bps, self.slippage_bps)
        if any(value < 0.0 for value in values):
            raise ValueError("cost parameters cannot be negative")

    @property
    def total_rate(self) -> float:
        return (self.commission_bps + self.half_spread_bps + self.slippage_bps) / 10_000


@dataclass(frozen=True)
class BacktestResult:
    """Full audit trail for a backtest."""

    returns: pd.DataFrame
    target_positions: pd.DataFrame
    held_positions: pd.DataFrame
    gross_returns: pd.Series
    turnover: pd.Series
    costs: pd.Series
    net_returns: pd.Series
    equity_curve: pd.Series
    metrics: dict[str, float]


def run_backtest(
    asset_returns: pd.DataFrame,
    target_positions: pd.DataFrame,
    *,
    costs: CostModel | None = None,
    signal_lag: int = 1,
    periods_per_year: int = 252,
) -> BacktestResult:
    """Run a close-to-close backtest with lagged holdings and linear costs.

    ``target_positions[t]`` is information available after period ``t``. With
    the default one-period lag, it earns ``asset_returns[t + 1]``. This explicit
    lag makes accidental same-bar look-ahead difficult.
    """
    if signal_lag < 1:
        raise ValueError("signal_lag must be at least one")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    require_datetime_index(asset_returns, "asset_returns")
    require_datetime_index(target_positions, "target_positions")
    require_same_shape_and_labels(asset_returns, target_positions)
    require_finite(asset_returns, "asset_returns")
    require_finite(target_positions, "target_positions")

    cost_model = costs or CostModel()
    held_positions = target_positions.shift(signal_lag).fillna(0.0)
    gross_returns = (held_positions * asset_returns).sum(axis=1)

    prior_positions = held_positions.shift(1).fillna(0.0)
    turnover = (held_positions - prior_positions).abs().sum(axis=1)
    realized_costs = turnover * cost_model.total_rate
    net_returns = gross_returns - realized_costs

    if (net_returns <= -1.0).any():
        raise ValueError("portfolio lost 100% or more in one period")
    equity_curve = (1.0 + net_returns).cumprod()
    metrics = dict(performance_summary(net_returns, periods_per_year))
    gross_total_return = float(np.prod(1.0 + gross_returns.to_numpy()) - 1.0)
    metrics.update(
        {
            "average_turnover": float(turnover.mean()),
            "total_cost": float(realized_costs.sum()),
            "gross_total_return": gross_total_return,
        }
    )

    return BacktestResult(
        returns=asset_returns.copy(),
        target_positions=target_positions.copy(),
        held_positions=held_positions,
        gross_returns=gross_returns.rename("gross_return"),
        turnover=turnover.rename("turnover"),
        costs=realized_costs.rename("cost"),
        net_returns=net_returns.rename("net_return"),
        equity_curve=equity_curve.rename("equity"),
        metrics=metrics,
    )


def inverse_volatility_weights(
    returns: pd.DataFrame,
    lookback: int = 20,
    volatility_floor: float = 1e-6,
) -> pd.DataFrame:
    """Scale assets inversely to trailing volatility with unit gross exposure."""
    if lookback < 2:
        raise ValueError("lookback must be at least two")
    rolling_vol = returns.rolling(lookback).std(ddof=1).clip(lower=volatility_floor)
    inverse_vol = 1.0 / rolling_vol
    denominator = inverse_vol.sum(axis=1).replace(0.0, np.nan)
    return inverse_vol.div(denominator, axis=0).fillna(0.0)
