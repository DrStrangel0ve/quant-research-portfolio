from pathlib import Path

import numpy as np

from quantlab.probability.games import (
    gambler_ruin_probability,
    monty_hall,
    simulate_gambler_ruin,
)
from quantlab.reporting.artifacts import write_summary


def main() -> None:
    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)
    exact = gambler_ruin_probability(
        initial_wealth=10,
        target_wealth=25,
        win_probability=0.48,
    )
    simulated = simulate_gambler_ruin(
        initial_wealth=10,
        target_wealth=25,
        win_probability=0.48,
        n_trials=100_000,
        rng=np.random.default_rng(1111),
    )
    stay = monty_hall(switch=False, n_trials=100_000, rng=np.random.default_rng(1112))
    switch = monty_hall(switch=True, n_trials=100_000, rng=np.random.default_rng(1113))
    write_summary(
        output / "summary.json",
        {
            "ruin_exact_success_probability": exact,
            "ruin_simulated_success_probability": simulated.estimate,
            "ruin_simulation_standard_error": simulated.standard_error,
            "monty_stay_win_rate": stay.estimate,
            "monty_switch_win_rate": switch.estimate,
        },
    )


if __name__ == "__main__":
    main()
