from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from quantlab.backtesting.engine import CostModel, run_backtest
from quantlab.data.synthetic import correlated_asset_prices
from quantlab.reporting.artifacts import write_summary
from quantlab.risk.portfolio import rolling_risk_parity_positions


def main() -> None:
    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)
    prices = correlated_asset_prices(n_assets=8, n_periods=2_000, seed=1010)
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    risk_parity = rolling_risk_parity_positions(
        returns,
        lookback=60,
        rebalance_frequency=21,
        shrinkage=0.25,
    )
    equal_weight = pd.DataFrame(
        1.0 / prices.shape[1],
        index=prices.index,
        columns=prices.columns,
    )
    costs = CostModel(commission_bps=0.5, half_spread_bps=1.0, slippage_bps=0.5)
    risk_result = run_backtest(returns, risk_parity, costs=costs)
    equal_result = run_backtest(returns, equal_weight, costs=costs)
    pd.DataFrame(
        {
            "risk_parity": risk_result.equity_curve,
            "equal_weight": equal_result.equity_curve,
        }
    ).plot(title="Risk parity versus equal weight")
    plt.tight_layout()
    plt.savefig(output / "portfolio_comparison.png", dpi=160)
    plt.close()
    write_summary(
        output / "summary.json",
        {
            **{f"risk_parity_{key}": value for key, value in risk_result.metrics.items()},
            "equal_weight_sharpe_ratio": equal_result.metrics["sharpe_ratio"],
            "equal_weight_max_drawdown": equal_result.metrics["max_drawdown"],
        },
    )


if __name__ == "__main__":
    main()
