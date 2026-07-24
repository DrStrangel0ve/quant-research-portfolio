"""Portfolio risk estimators and allocation algorithms."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def shrink_covariance(
    returns: pd.DataFrame,
    *,
    shrinkage: float = 0.2,
) -> pd.DataFrame:
    """Shrink the sample covariance toward a diagonal target."""
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be between zero and one")
    covariance = returns.astype(float).cov()
    diagonal = np.diag(np.diag(covariance.to_numpy()))
    shrunk = (1.0 - shrinkage) * covariance.to_numpy() + shrinkage * diagonal
    return pd.DataFrame(shrunk, index=covariance.index, columns=covariance.columns)


def portfolio_volatility(weights: FloatArray, covariance: FloatArray) -> float:
    """Portfolio standard deviation."""
    variance = float(weights @ covariance @ weights)
    return float(np.sqrt(max(variance, 0.0)))


def risk_contributions(weights: FloatArray, covariance: FloatArray) -> FloatArray:
    """Component contributions to total portfolio volatility."""
    volatility = portfolio_volatility(weights, covariance)
    if volatility == 0.0:
        return np.zeros_like(weights)
    marginal = covariance @ weights / volatility
    return np.asarray(weights * marginal, dtype=np.float64)


def equal_risk_contribution_weights(
    covariance: pd.DataFrame,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
) -> pd.Series:
    """Long-only equal-risk-contribution weights via multiplicative updates."""
    matrix = covariance.to_numpy(dtype=float)
    n_assets = len(covariance)
    if matrix.shape != (n_assets, n_assets) or n_assets == 0:
        raise ValueError("covariance must be a non-empty square matrix")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("covariance must be symmetric")
    if np.linalg.eigvalsh(matrix).min() < -1e-10:
        raise ValueError("covariance must be positive semidefinite")

    weights = np.full(n_assets, 1.0 / n_assets)
    for _ in range(max_iterations):
        contributions = risk_contributions(weights, matrix)
        target = contributions.sum() / n_assets
        if np.max(np.abs(contributions - target)) < tolerance:
            break
        safe = np.maximum(contributions, 1e-16)
        weights *= target / safe
        weights = np.maximum(weights, 1e-16)
        weights /= weights.sum()
    else:
        raise RuntimeError("risk-parity solver did not converge")

    return pd.Series(weights, index=covariance.index, name="weight")


def marginal_expected_shortfall(
    returns: pd.DataFrame,
    weights: pd.Series,
    *,
    confidence: float = 0.95,
) -> pd.Series:
    """Average asset return in portfolio tail states, reported as positive loss."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    aligned_weights = weights.reindex(returns.columns)
    if aligned_weights.isna().any():
        raise ValueError("weights must cover every return column")
    portfolio_returns = returns @ aligned_weights
    cutoff = portfolio_returns.quantile(1.0 - confidence)
    tail = returns.loc[portfolio_returns <= cutoff]
    return (-tail.mean()).rename("marginal_expected_shortfall")


def rolling_risk_parity_positions(
    returns: pd.DataFrame,
    *,
    lookback: int = 60,
    rebalance_frequency: int = 21,
    shrinkage: float = 0.2,
) -> pd.DataFrame:
    """Trailing equal-risk-contribution allocations held between rebalances."""
    if lookback < 3 or rebalance_frequency <= 0:
        raise ValueError("lookback must be >= 3 and rebalance_frequency must be positive")
    positions = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    for position in range(lookback, len(returns), rebalance_frequency):
        trailing = returns.iloc[position - lookback : position]
        covariance = shrink_covariance(trailing, shrinkage=shrinkage)
        positions.iloc[position] = equal_risk_contribution_weights(covariance)
    return positions.ffill().fillna(0.0)
