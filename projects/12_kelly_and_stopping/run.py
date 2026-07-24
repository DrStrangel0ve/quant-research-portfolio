from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantlab.probability.games import (
    kelly_fraction,
    secretary_game,
    simulate_kelly_wealth,
)
from quantlab.reporting.artifacts import write_summary


def main() -> None:
    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)
    optimal = kelly_fraction(
        win_probability=0.55,
        net_win_multiple=1.0,
        net_loss_multiple=1.0,
    )
    wealth_records: list[dict[str, float | str]] = []
    for label, fraction in (
        ("half_kelly", 0.5 * optimal),
        ("full_kelly", optimal),
        ("over_bet", 2.0 * optimal),
    ):
        wealth = simulate_kelly_wealth(
            initial_wealth=1.0,
            fraction=fraction,
            win_probability=0.55,
            net_win_multiple=1.0,
            net_loss_multiple=1.0,
            n_bets=250,
            n_paths=50_000,
            rng=np.random.default_rng(1200 + int(fraction * 1_000)),
        )
        wealth_records.append(
            {
                "policy": label,
                "fraction": fraction,
                "median_terminal_wealth": float(np.median(wealth)),
                "mean_log_terminal_wealth": float(np.log(wealth).mean()),
                "loss_probability": float((wealth < 1.0).mean()),
            }
        )
    wealth_table = pd.DataFrame(wealth_records)
    wealth_table.to_csv(output / "kelly_comparison.csv", index=False)

    fractions = np.linspace(0.05, 0.80, 31)
    success_rates = [
        secretary_game(
            n_candidates=100,
            sample_fraction=float(fraction),
            n_trials=10_000,
            rng=np.random.default_rng(12_000 + index),
        ).estimate
        for index, fraction in enumerate(fractions)
    ]
    best_index = int(np.argmax(success_rates))
    plt.plot(fractions, success_rates)
    plt.axvline(1.0 / np.e, linestyle="--", color="black", label="1/e")
    plt.xlabel("Initial rejection fraction")
    plt.ylabel("Probability of selecting the best")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "secretary_rule.png", dpi=160)
    plt.close()
    full_kelly = wealth_table.loc[wealth_table["policy"] == "full_kelly"].iloc[0]
    write_summary(
        output / "summary.json",
        {
            "kelly_fraction": optimal,
            "full_kelly_mean_log_terminal_wealth": float(
                full_kelly["mean_log_terminal_wealth"]
            ),
            "full_kelly_loss_probability": float(full_kelly["loss_probability"]),
            "best_secretary_sample_fraction": float(fractions[best_index]),
            "best_secretary_success_rate": float(success_rates[best_index]),
            "theoretical_secretary_fraction": float(1.0 / np.e),
        },
    )


if __name__ == "__main__":
    main()
