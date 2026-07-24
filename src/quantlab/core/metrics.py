"""Performance and risk statistics with explicit annualization assumptions."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _clean_returns(returns: pd.Series) -> pd.Series:
    cleaned = returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        raise ValueError("returns must contain at least one finite observation")
    return cleaned


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Geometrically annualize simple periodic returns."""
    values = _clean_returns(returns)
    if (values <= -1.0).any():
        raise ValueError("simple returns cannot be less than or equal to -100%")
    growth = float(np.prod(1.0 + values.to_numpy()))
    return float(growth ** (periods_per_year / len(values)) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized sample volatility."""
    values = _clean_returns(returns)
    if len(values) < 2:
        return 0.0
    return float(values.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized arithmetic Sharpe ratio."""
    values = _clean_returns(returns)
    periodic_rf = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = values - periodic_rf
    volatility = float(excess.std(ddof=1))
    if volatility == 0.0 or np.isnan(volatility):
        return 0.0
    return float(excess.mean() / volatility * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sortino ratio using target downside deviation."""
    values = _clean_returns(returns)
    target_periodic = (1.0 + target_return) ** (1.0 / periods_per_year) - 1.0
    excess = values - target_periodic
    downside = np.minimum(excess.to_numpy(), 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
    if downside_deviation == 0.0:
        return 0.0
    return float(excess.mean() / downside_deviation * np.sqrt(periods_per_year))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Return the underwater equity curve."""
    values = _clean_returns(returns)
    equity = (1.0 + values).cumprod()
    running_max = equity.cummax()
    return equity / running_max - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Most negative peak-to-trough drawdown."""
    return float(drawdown_series(returns).min())


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical loss VaR as a positive number."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    values = _clean_returns(returns)
    return float(-np.quantile(values.to_numpy(), 1.0 - confidence))


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical expected shortfall as a positive loss number."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    values = _clean_returns(returns)
    cutoff = np.quantile(values.to_numpy(), 1.0 - confidence)
    tail = values[values <= cutoff]
    return float(-tail.mean())


def performance_summary(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> Mapping[str, float]:
    """Compute a compact recruiter-facing performance scorecard."""
    values = _clean_returns(returns)
    total_return = float(np.prod(1.0 + values.to_numpy()) - 1.0)
    return {
        "total_return": total_return,
        "annualized_return": annualized_return(values, periods_per_year),
        "annualized_volatility": annualized_volatility(values, periods_per_year),
        "sharpe_ratio": sharpe_ratio(values, periods_per_year=periods_per_year),
        "sortino_ratio": sortino_ratio(values, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(values),
        "var_95": value_at_risk(values),
        "expected_shortfall_95": expected_shortfall(values),
        "hit_rate": float((values > 0.0).mean()),
    }
