import numpy as np
import pandas as pd
import pytest

from quantlab.core.metrics import (
    annualized_return,
    drawdown_series,
    expected_shortfall,
    max_drawdown,
    performance_summary,
    sharpe_ratio,
    value_at_risk,
)


def test_annualized_return_uses_geometric_compounding() -> None:
    returns = pd.Series([0.01] * 12)
    assert annualized_return(returns, periods_per_year=12) == pytest.approx(1.01**12 - 1.0)


def test_drawdown_tracks_peak_to_trough_loss() -> None:
    returns = pd.Series([0.10, -0.20, 0.05])
    drawdowns = drawdown_series(returns)
    assert drawdowns.iloc[0] == pytest.approx(0.0)
    assert max_drawdown(returns) == pytest.approx(-0.20)


def test_zero_volatility_has_zero_sharpe() -> None:
    assert sharpe_ratio(pd.Series([0.0, 0.0, 0.0])) == 0.0


def test_tail_statistics_are_positive_losses() -> None:
    returns = pd.Series(np.linspace(-0.10, 0.10, 101))
    assert value_at_risk(returns) > 0.0
    assert expected_shortfall(returns) >= value_at_risk(returns)


def test_summary_rejects_impossible_simple_returns() -> None:
    with pytest.raises(ValueError, match="100%"):
        performance_summary(pd.Series([0.01, -1.0]))
