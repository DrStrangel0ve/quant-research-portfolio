"""Monte Carlo primitives with deterministic random-number injection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MonteCarloEstimate:
    estimate: float
    standard_error: float
    confidence_low: float
    confidence_high: float
    n_paths: int


def geometric_brownian_motion(
    *,
    spot: float,
    drift: float,
    volatility: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    antithetic: bool = False,
) -> FloatArray:
    """Simulate exact-discretization GBM paths, including the initial spot."""
    if spot <= 0.0 or maturity <= 0.0:
        raise ValueError("spot and maturity must be positive")
    if volatility < 0.0:
        raise ValueError("volatility cannot be negative")
    if n_steps <= 0 or n_paths <= 0:
        raise ValueError("n_steps and n_paths must be positive")

    draws_needed = (n_paths + 1) // 2 if antithetic else n_paths
    shocks = rng.standard_normal((n_steps, draws_needed))
    if antithetic:
        shocks = np.concatenate([shocks, -shocks], axis=1)[:, :n_paths]

    dt = maturity / n_steps
    increments = (drift - 0.5 * volatility**2) * dt
    increments = increments + volatility * np.sqrt(dt) * shocks
    log_paths = np.vstack([np.zeros(n_paths), np.cumsum(increments, axis=0)])
    return np.asarray(spot * np.exp(log_paths), dtype=np.float64)


def estimate_discounted_payoff(
    discounted_payoffs: FloatArray,
    confidence: float = 0.95,
) -> MonteCarloEstimate:
    """Summarize discounted pathwise payoffs with a normal confidence interval."""
    if discounted_payoffs.ndim != 1 or len(discounted_payoffs) < 2:
        raise ValueError("discounted_payoffs must contain at least two values")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    values = np.asarray(discounted_payoffs, dtype=float)
    estimate = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))

    from scipy.stats import norm

    critical_value = float(norm.ppf(0.5 + confidence / 2.0))
    margin = critical_value * standard_error
    return MonteCarloEstimate(
        estimate=estimate,
        standard_error=standard_error,
        confidence_low=estimate - margin,
        confidence_high=estimate + margin,
        n_paths=len(values),
    )
