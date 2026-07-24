from pathlib import Path

import matplotlib.pyplot as plt

from quantlab.backtesting.engine import CostModel, run_backtest
from quantlab.data.synthetic import cointegrated_pair
from quantlab.reporting.artifacts import write_summary
from quantlab.strategies.pairs import rolling_pairs_signal


def main() -> None:
    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)
    prices = cointegrated_pair(n_periods=2_000, seed=6006)
    signal = rolling_pairs_signal(
        prices,
        dependent="Dependent",
        independent="Independent",
        hedge_lookback=90,
        zscore_lookback=30,
    )
    result = run_backtest(
        prices.pct_change(fill_method=None).fillna(0.0),
        signal.positions,
        costs=CostModel(commission_bps=0.5, half_spread_bps=1.0, slippage_bps=0.5),
    )
    result.equity_curve.plot(title="Pairs strategy net equity")
    plt.tight_layout()
    plt.savefig(output / "equity_curve.png", dpi=160)
    plt.close()
    write_summary(output / "summary.json", result.metrics)


if __name__ == "__main__":
    main()
