"""Synthetic market generators used by examples and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def correlated_asset_prices(
    *,
    n_assets: int,
    n_periods: int,
    seed: int,
    annual_drift: float = 0.06,
    annual_volatility: float = 0.20,
    correlation: float = 0.3,
) -> pd.DataFrame:
    """Generate correlated daily GBM prices with heterogeneous trends."""
    if n_assets < 2 or n_periods < 2:
        raise ValueError("at least two assets and periods are required")
    lower_bound = -1.0 / (n_assets - 1)
    if not lower_bound < correlation < 1.0:
        raise ValueError("correlation is invalid for an equicorrelation matrix")
    rng = np.random.default_rng(seed)
    covariance = np.full((n_assets, n_assets), correlation)
    np.fill_diagonal(covariance, 1.0)
    shocks = rng.multivariate_normal(np.zeros(n_assets), covariance, size=n_periods)
    asset_drifts = np.linspace(annual_drift - 0.04, annual_drift + 0.04, n_assets)
    asset_volatility = np.linspace(annual_volatility * 0.7, annual_volatility * 1.3, n_assets)
    log_returns = (
        (asset_drifts - 0.5 * asset_volatility**2) / 252
        + shocks * asset_volatility / np.sqrt(252)
    )
    prices = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    index = pd.bdate_range("2015-01-01", periods=n_periods)
    columns = [f"Asset_{index + 1:02d}" for index in range(n_assets)]
    return pd.DataFrame(prices, index=index, columns=columns)


def cointegrated_pair(
    *,
    n_periods: int,
    seed: int,
    hedge_ratio: float = 1.25,
    mean_reversion: float = 0.12,
) -> pd.DataFrame:
    """Generate two positive price series with a stationary log-price spread."""
    if n_periods < 2 or hedge_ratio <= 0.0:
        raise ValueError("n_periods and hedge_ratio must be positive")
    if not 0.0 < mean_reversion < 1.0:
        raise ValueError("mean_reversion must be between zero and one")
    rng = np.random.default_rng(seed)
    common = np.cumsum(0.0002 + 0.012 * rng.standard_normal(n_periods))
    spread = np.zeros(n_periods)
    for position in range(1, n_periods):
        spread[position] = (
            (1.0 - mean_reversion) * spread[position - 1]
            + 0.012 * rng.standard_normal()
        )
    independent = np.exp(4.5 + common)
    dependent = np.exp(0.2 + hedge_ratio * np.log(independent) + spread) / 100
    date_index = pd.bdate_range("2015-01-01", periods=n_periods)
    return pd.DataFrame(
        {"Dependent": dependent, "Independent": independent},
        index=date_index,
    )


def regime_switching_prices(
    *,
    n_assets: int,
    n_periods: int,
    seed: int,
) -> pd.DataFrame:
    """Generate assets whose expected returns change across three regimes."""
    if n_assets < 2 or n_periods < 30:
        raise ValueError("at least two assets and 30 periods are required")
    rng = np.random.default_rng(seed)
    regimes = np.repeat(np.arange(3), np.ceil(n_periods / 3))[:n_periods].astype(int)
    base_loadings = np.linspace(-1.0, 1.0, n_assets)
    regime_sign = np.array([1.0, -0.5, 0.8])
    expected = regime_sign[regimes, None] * base_loadings[None, :] * 0.0005
    market = 0.008 * rng.standard_normal((n_periods, 1))
    idiosyncratic = 0.010 * rng.standard_normal((n_periods, n_assets))
    log_returns = expected + 0.45 * market + idiosyncratic
    prices = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    index = pd.bdate_range("2012-01-02", periods=n_periods)
    columns = [f"Asset_{index + 1:02d}" for index in range(n_assets)]
    return pd.DataFrame(prices, index=index, columns=columns)


def simulate_garch_returns(
    *,
    n_periods: int,
    seed: int,
    omega: float = 2e-6,
    alpha: float = 0.08,
    beta: float = 0.90,
) -> pd.Series:
    """Generate Gaussian GARCH(1,1) returns."""
    if n_periods < 2 or omega <= 0.0 or alpha < 0.0 or beta < 0.0:
        raise ValueError("invalid GARCH simulation parameters")
    if alpha + beta >= 1.0:
        raise ValueError("stationary simulation requires alpha + beta < 1")
    rng = np.random.default_rng(seed)
    variance = np.empty(n_periods)
    values = np.empty(n_periods)
    variance[0] = omega / (1.0 - alpha - beta)
    values[0] = np.sqrt(variance[0]) * rng.standard_normal()
    for index in range(1, n_periods):
        variance[index] = omega + alpha * values[index - 1] ** 2 + beta * variance[index - 1]
        values[index] = np.sqrt(variance[index]) * rng.standard_normal()
    dates = pd.bdate_range("2010-01-01", periods=n_periods)
    return pd.Series(values, index=dates, name="return")
