from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantlab.derivatives.black_scholes import (
    black_scholes_price,
    european_option_monte_carlo,
)
from quantlab.reporting.artifacts import write_summary


def main() -> None:
    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)
    analytic = black_scholes_price(
        spot=100.0, strike=105.0, maturity=1.0, rate=0.03, volatility=0.20
    )
    records: list[dict[str, float | int | bool]] = []
    for n_paths in (1_000, 5_000, 20_000, 100_000):
        for antithetic in (False, True):
            estimate = european_option_monte_carlo(
                spot=100.0,
                strike=105.0,
                maturity=1.0,
                rate=0.03,
                volatility=0.20,
                n_paths=n_paths,
                rng=np.random.default_rng(10_000 + n_paths + int(antithetic)),
                antithetic=antithetic,
            )
            records.append(
                {
                    "n_paths": n_paths,
                    "antithetic": antithetic,
                    "estimate": estimate.estimate,
                    "standard_error": estimate.standard_error,
                    "absolute_error": abs(estimate.estimate - analytic),
                }
            )
    results = pd.DataFrame(records)
    results.to_csv(output / "convergence.csv", index=False)
    for label, group in results.groupby("antithetic"):
        plt.loglog(
            group["n_paths"],
            group["absolute_error"],
            marker="o",
            label=f"antithetic={label}",
        )
    plt.xlabel("Simulated paths")
    plt.ylabel("Absolute pricing error")
    plt.title("European call Monte Carlo convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "convergence.png", dpi=160)
    plt.close()
    best = results.loc[results["n_paths"].idxmax()]
    write_summary(
        output / "summary.json",
        {
            "analytic_price": analytic,
            "largest_run_estimate": float(best["estimate"]),
            "largest_run_absolute_error": float(best["absolute_error"]),
            "largest_run_standard_error": float(best["standard_error"]),
            "seeded": True,
        },
    )


if __name__ == "__main__":
    main()
