"""Heston stochastic-volatility simulation using full-truncation Euler."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class HestonParameters:
    mean_reversion: float
    long_run_variance: float
    vol_of_variance: float
    correlation: float
    initial_variance: float

    def __post_init__(self) -> None:
        positive = (
            self.mean_reversion,
            self.long_run_variance,
            self.vol_of_variance,
            self.initial_variance,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("Heston variance parameters must be positive")
        if not -1.0 <= self.correlation <= 1.0:
            raise ValueError("correlation must lie between -1 and 1")

    @property
    def feller_ratio(self) -> float:
        """Values >= 1 satisfy the Feller positivity condition."""
        return 2.0 * self.mean_reversion * self.long_run_variance / self.vol_of_variance**2


def simulate_heston(
    *,
    spot: float,
    drift: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
    parameters: HestonParameters,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    """Simulate spot and variance paths with full-truncation Euler."""
    if spot <= 0.0 or maturity <= 0.0:
        raise ValueError("spot and maturity must be positive")
    if n_steps <= 0 or n_paths <= 0:
        raise ValueError("n_steps and n_paths must be positive")

    dt = maturity / n_steps
    sqrt_dt = np.sqrt(dt)
    spots = np.empty((n_steps + 1, n_paths), dtype=float)
    variances = np.empty_like(spots)
    spots[0] = spot
    variances[0] = parameters.initial_variance

    independent_spot = rng.standard_normal((n_steps, n_paths))
    independent_variance = rng.standard_normal((n_steps, n_paths))
    spot_shocks = (
        parameters.correlation * independent_variance
        + np.sqrt(1.0 - parameters.correlation**2) * independent_spot
    )

    for step in range(n_steps):
        variance = np.maximum(variances[step], 0.0)
        next_variance = (
            variances[step]
            + parameters.mean_reversion * (parameters.long_run_variance - variance) * dt
            + parameters.vol_of_variance * np.sqrt(variance) * sqrt_dt * independent_variance[step]
        )
        variances[step + 1] = np.maximum(next_variance, 0.0)
        log_increment = (drift - 0.5 * variance) * dt
        log_increment += np.sqrt(variance) * sqrt_dt * spot_shocks[step]
        spots[step + 1] = spots[step] * np.exp(log_increment)

    return (
        np.asarray(spots, dtype=np.float64),
        np.asarray(variances, dtype=np.float64),
    )
