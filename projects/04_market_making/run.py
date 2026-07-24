from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantlab.reporting.artifacts import write_summary
from quantlab.simulation.market_making import MarketMakerParameters, simulate_market_maker


def main() -> None:
    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)
    records: list[dict[str, float]] = []
    for risk_aversion in (0.03, 0.10, 0.30):
        result = simulate_market_maker(
            parameters=MarketMakerParameters(risk_aversion=risk_aversion),
            horizon=1.0,
            n_steps=600,
            n_paths=5_000,
            rng=np.random.default_rng(4000 + int(100 * risk_aversion)),
        )
        records.append(
            {
                "risk_aversion": risk_aversion,
                "mean_pnl": result.mean_pnl,
                "pnl_std": result.pnl_std,
                "inventory_var_95": result.inventory_var_95,
            }
        )
    results = pd.DataFrame(records)
    results.to_csv(output / "risk_aversion_sweep.csv", index=False)
    fig, left = plt.subplots()
    right = left.twinx()
    left.plot(results["risk_aversion"], results["pnl_std"], marker="o", color="tab:blue")
    right.plot(
        results["risk_aversion"],
        results["inventory_var_95"],
        marker="s",
        color="tab:orange",
    )
    left.set_xlabel("Risk aversion")
    left.set_ylabel("Terminal P&L standard deviation", color="tab:blue")
    right.set_ylabel("|Inventory| 95th percentile", color="tab:orange")
    fig.tight_layout()
    fig.savefig(output / "inventory_risk.png", dpi=160)
    plt.close(fig)
    middle = results.iloc[1]
    write_summary(
        output / "summary.json",
        {
            "baseline_risk_aversion": float(middle["risk_aversion"]),
            "baseline_mean_pnl": float(middle["mean_pnl"]),
            "baseline_pnl_std": float(middle["pnl_std"]),
            "baseline_inventory_var_95": float(middle["inventory_var_95"]),
        },
    )


if __name__ == "__main__":
    main()
