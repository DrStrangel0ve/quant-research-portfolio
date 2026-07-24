import numpy as np
import pandas as pd
import pytest

from quantlab.backtesting.engine import CostModel, run_backtest
from quantlab.backtesting.walk_forward import expanding_window_splits


def _frame(values: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(
        values,
        index=pd.bdate_range("2024-01-01", periods=len(values)),
        columns=["asset"],
    )


def test_positions_are_lagged_before_earning_returns() -> None:
    returns = _frame([[0.10], [0.20], [0.30]])
    targets = _frame([[1.0], [0.0], [0.0]])
    result = run_backtest(returns, targets)
    assert result.gross_returns.tolist() == pytest.approx([0.0, 0.20, 0.0])


def test_costs_are_charged_on_entry_and_exit_turnover() -> None:
    returns = _frame([[0.0], [0.0], [0.0]])
    targets = _frame([[1.0], [0.0], [0.0]])
    result = run_backtest(returns, targets, costs=CostModel(commission_bps=10.0))
    assert result.turnover.tolist() == pytest.approx([0.0, 1.0, 1.0])
    assert result.costs.sum() == pytest.approx(0.002)


def test_backtest_requires_identical_labels() -> None:
    returns = _frame([[0.0], [0.0]])
    positions = returns.rename(columns={"asset": "other"})
    with pytest.raises(ValueError, match="columns"):
        run_backtest(returns, positions)


def test_walk_forward_split_is_chronological_and_embargoed() -> None:
    splits = list(
        expanding_window_splits(
            30,
            min_train_size=10,
            test_size=5,
            step_size=5,
            embargo=2,
        )
    )
    assert len(splits) == 3
    for split in splits:
        assert split.train.max() + 2 < split.test.min()
        assert np.intersect1d(split.train, split.test).size == 0
