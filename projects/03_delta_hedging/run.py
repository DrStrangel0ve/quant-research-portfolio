from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantlab.derivatives.hedging import simulate_delta_hedge
from quantlab.reporting.artifacts import write_summary


def main() -> None:
    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)
    records: list[dict[str, float | int]] = []
    for rebalances in (12, 52, 252):
        result = simulate_delta_hedge(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.02,
            implied_volatility=0.20,
            realized_volatility=0.22,
            n_rebalances=rebalances,
            n_paths=10_000,
            transaction_cost_bps=2.0,
            rng=np.random.default_rng(3000 + rebalances),
        )
        records.append(
            {
                "rebalances": rebalances,
                "mean_error": result.mean_error,
                "error_std": result.error_std,
                "expected_shortfall_95": result.expected_shortfall_95,
                "mean_cost": float(result.costs.mean()),
            }
        )
    results = pd.DataFrame(records)
    results.to_csv(output / "frequency_comparison.csv", index=False)
    results.plot(x="rebalances", y=["error_std", "mean_cost"], marker="o")
    plt.title("Hedge precision versus trading cost")
    plt.ylabel("Value per option")
    plt.tight_layout()
    plt.savefig(output / "frequency_tradeoff.png", dpi=160)
    plt.close()
    finest = results.iloc[-1]
    write_summary(
        output / "summary.json",
        {
            "finest_rebalance_count": int(finest["rebalances"]),
            "finest_mean_error": float(finest["mean_error"]),
            "finest_error_std": float(finest["error_std"]),
            "finest_expected_shortfall_95": float(finest["expected_shortfall_95"]),
            "finest_mean_cost": float(finest["mean_cost"]),
        },
    )


if __name__ == "__main__":
    main()
