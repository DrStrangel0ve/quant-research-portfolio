from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from quantlab.backtesting.engine import CostModel, run_backtest
from quantlab.backtesting.walk_forward import expanding_window_splits
from quantlab.data.synthetic import regime_switching_prices
from quantlab.reporting.artifacts import write_summary
from quantlab.strategies.trend import time_series_momentum_positions


def main() -> None:
    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)
    prices = regime_switching_prices(n_assets=8, n_periods=2_800, seed=8008)
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    candidate_lookbacks = (21, 63, 126)
    candidate_positions = {
        lookback: time_series_momentum_positions(prices, signal_lookback=lookback)
        for lookback in candidate_lookbacks
    }
    out_of_sample = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    selections: list[dict[str, int | float]] = []
    for split_number, split in enumerate(
        expanding_window_splits(
            len(prices),
            min_train_size=756,
            test_size=252,
            embargo=1,
        ),
        start=1,
    ):
        scores: dict[int, float] = {}
        for lookback, positions in candidate_positions.items():
            training_result = run_backtest(
                returns.iloc[split.train],
                positions.iloc[split.train],
                costs=CostModel(commission_bps=0.5, half_spread_bps=1.0, slippage_bps=1.0),
            )
            scores[lookback] = training_result.metrics["sharpe_ratio"]
        selected = max(scores, key=scores.get)
        out_of_sample.iloc[split.test] = candidate_positions[selected].iloc[split.test]
        selections.append(
            {
                "split": split_number,
                "selected_lookback": selected,
                "training_sharpe": scores[selected],
            }
        )

    result = run_backtest(
        returns,
        out_of_sample,
        costs=CostModel(commission_bps=0.5, half_spread_bps=1.0, slippage_bps=1.0),
    )
    pd.DataFrame(selections).to_csv(output / "model_selections.csv", index=False)
    result.equity_curve.plot(title="Walk-forward trend: stitched OOS equity")
    plt.tight_layout()
    plt.savefig(output / "oos_equity_curve.png", dpi=160)
    plt.close()
    write_summary(
        output / "summary.json",
        {
            **result.metrics,
            "walk_forward_splits": len(selections),
            "candidate_models": len(candidate_lookbacks),
        },
    )


if __name__ == "__main__":
    main()
