from pathlib import Path

import matplotlib.pyplot as plt

from quantlab.data.synthetic import simulate_garch_returns
from quantlab.reporting.artifacts import write_summary
from quantlab.time_series.garch import fit_garch_11, forecast_variance


def main() -> None:
    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)
    returns = simulate_garch_returns(
        n_periods=3_000,
        seed=9009,
        omega=2e-6,
        alpha=0.08,
        beta=0.90,
    )
    fit = fit_garch_11(returns)
    forecast = forecast_variance(fit, horizon=30)
    fit.conditional_variance.iloc[-500:].plot(label="conditional variance")
    returns.pow(2).iloc[-500:].plot(alpha=0.35, label="squared return")
    plt.legend()
    plt.title("Estimated GARCH variance")
    plt.tight_layout()
    plt.savefig(output / "conditional_variance.png", dpi=160)
    plt.close()
    forecast.to_csv(output / "variance_forecast.csv", header=True)
    write_summary(
        output / "summary.json",
        {
            "converged": fit.converged,
            "estimated_omega": fit.omega,
            "estimated_alpha": fit.alpha,
            "estimated_beta": fit.beta,
            "estimated_persistence": fit.persistence,
            "true_persistence": 0.98,
            "log_likelihood": fit.log_likelihood,
            "forecast_day_30": float(forecast.iloc[-1]),
        },
    )


if __name__ == "__main__":
    main()
