from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from quantlab.derivatives.black_scholes import black_scholes_price
from quantlab.derivatives.heston import HestonParameters, simulate_heston
from quantlab.reporting.artifacts import write_summary


def main() -> None:
    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)
    parameters = HestonParameters(
        mean_reversion=2.0,
        long_run_variance=0.04,
        vol_of_variance=0.5,
        correlation=-0.7,
        initial_variance=0.04,
    )
    spots, variances = simulate_heston(
        spot=100.0,
        drift=0.02,
        maturity=1.0,
        n_steps=252,
        n_paths=30_000,
        parameters=parameters,
        rng=np.random.default_rng(2026),
    )
    strikes = np.arange(75.0, 126.0, 5.0)
    implied_volatility: list[float] = []
    for strike in strikes:
        option_price = np.exp(-0.02) * np.maximum(spots[-1] - strike, 0.0).mean()
        implied_volatility.append(
            brentq(
                lambda volatility,
                selected_strike=strike,
                target_price=option_price: (
                    black_scholes_price(
                        spot=100.0,
                        strike=float(selected_strike),
                        maturity=1.0,
                        rate=0.02,
                        volatility=volatility,
                    )
                    - target_price
                ),
                0.01,
                2.0,
            )
        )
    plt.plot(strikes, implied_volatility, marker="o")
    plt.xlabel("Strike")
    plt.ylabel("Black-Scholes implied volatility")
    plt.title("Heston terminal-distribution volatility skew")
    plt.tight_layout()
    plt.savefig(output / "implied_volatility.png", dpi=160)
    plt.close()
    write_summary(
        output / "summary.json",
        {
            "paths": spots.shape[1],
            "steps": spots.shape[0] - 1,
            "feller_ratio": parameters.feller_ratio,
            "feller_condition_satisfied": parameters.feller_ratio >= 1.0,
            "mean_terminal_spot": float(spots[-1].mean()),
            "mean_terminal_variance": float(variances[-1].mean()),
            "atm_implied_volatility": implied_volatility[len(implied_volatility) // 2],
        },
    )


if __name__ == "__main__":
    main()
