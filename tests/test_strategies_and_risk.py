import numpy as np
import pandas as pd
import pytest

from quantlab.data.synthetic import cointegrated_pair, correlated_asset_prices
from quantlab.risk.portfolio import (
    equal_risk_contribution_weights,
    risk_contributions,
    rolling_risk_parity_positions,
    shrink_covariance,
)
from quantlab.strategies.momentum import cross_sectional_momentum_positions
from quantlab.strategies.pairs import rolling_pairs_signal
from quantlab.strategies.trend import time_series_momentum_positions


def test_pairs_signal_does_not_change_past_when_future_changes() -> None:
    prices = cointegrated_pair(n_periods=400, seed=20)
    original = rolling_pairs_signal(
        prices,
        dependent="Dependent",
        independent="Independent",
        hedge_lookback=60,
        zscore_lookback=20,
    )
    changed = prices.copy()
    changed.iloc[300:] *= 4.0
    perturbed = rolling_pairs_signal(
        changed,
        dependent="Dependent",
        independent="Independent",
        hedge_lookback=60,
        zscore_lookback=20,
    )
    pd.testing.assert_frame_equal(original.positions.iloc[:300], perturbed.positions.iloc[:300])


def test_cross_sectional_momentum_is_dollar_neutral_when_active() -> None:
    prices = correlated_asset_prices(n_assets=10, n_periods=300, seed=21)
    positions = cross_sectional_momentum_positions(
        prices,
        lookback=30,
        volatility_lookback=20,
        selection_fraction=0.2,
    )
    active = positions.abs().sum(axis=1) > 0.0
    assert np.allclose(positions.loc[active].sum(axis=1), 0.0)
    assert np.allclose(positions.loc[active].abs().sum(axis=1), 1.0)


def test_trend_positions_respect_gross_leverage_cap() -> None:
    prices = correlated_asset_prices(n_assets=5, n_periods=300, seed=22)
    positions = time_series_momentum_positions(prices, leverage_cap=1.5)
    assert (positions.abs().sum(axis=1) <= 1.5 + 1e-12).all()


def test_equal_risk_contribution_solver_equalizes_components() -> None:
    covariance = pd.DataFrame(
        [[0.04, 0.006], [0.006, 0.09]],
        index=["low", "high"],
        columns=["low", "high"],
    )
    weights = equal_risk_contribution_weights(covariance)
    contributions = risk_contributions(weights.to_numpy(), covariance.to_numpy())
    assert weights.sum() == pytest.approx(1.0)
    assert contributions[0] == pytest.approx(contributions[1], rel=1e-6)
    assert weights["low"] > weights["high"]


def test_shrink_covariance_preserves_diagonal() -> None:
    prices = correlated_asset_prices(n_assets=4, n_periods=200, seed=23)
    returns = prices.pct_change(fill_method=None).dropna()
    sample = returns.cov()
    shrunk = shrink_covariance(returns, shrinkage=0.5)
    assert np.allclose(np.diag(sample), np.diag(shrunk))


def test_rolling_risk_parity_uses_only_trailing_data() -> None:
    prices = correlated_asset_prices(n_assets=4, n_periods=300, seed=24)
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    original = rolling_risk_parity_positions(returns, lookback=40, rebalance_frequency=10)
    changed = returns.copy()
    changed.iloc[200:] *= 10.0
    perturbed = rolling_risk_parity_positions(changed, lookback=40, rebalance_frequency=10)
    pd.testing.assert_frame_equal(original.iloc[:200], perturbed.iloc[:200])
