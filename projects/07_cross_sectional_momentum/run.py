from pathlib import Path

import matplotlib.pyplot as plt

from quantlab.backtesting.engine import CostModel, run_backtest
from quantlab.data.synthetic import regime_switching_prices
from quantlab.reporting.artifacts import write_summary
from quantlab.strategies.momentum import cross_sectional_momentum_positions


def main() -> None:
    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)
    prices = regime_switching_prices(n_assets=20, n_periods=2_500, seed=7007)
    positions = cross_sectional_momentum_positions(
        prices,
        lookback=126,
        volatility_lookback=30,
        selection_fraction=0.2,
    )
    result = run_backtest(
        prices.pct_change(fill_method=None).fillna(0.0),
        positions,
        costs=CostModel(commission_bps=0.5, half_spread_bps=1.5, slippage_bps=1.0),
    )
    result.equity_curve.plot(title="Cross-sectional momentum net equity")
    plt.tight_layout()
    plt.savefig(output / "equity_curve.png", dpi=160)
    plt.close()
    write_summary(output / "summary.json", result.metrics)


if __name__ == "__main__":
    main()
