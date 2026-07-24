"""Small, inspectable GARCH(1,1) maximum-likelihood implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GarchFit:
    omega: float
    alpha: float
    beta: float
    conditional_variance: pd.Series
    last_squared_return: float
    log_likelihood: float
    converged: bool

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def unconditional_variance(self) -> float:
        return self.omega / (1.0 - self.persistence)


def _variance_path(values: FloatArray, omega: float, alpha: float, beta: float) -> FloatArray:
    variances = np.empty_like(values)
    variances[0] = max(float(np.var(values, ddof=1)), 1e-10)
    for index in range(1, len(values)):
        variances[index] = omega + alpha * values[index - 1] ** 2 + beta * variances[index - 1]
    return variances


def fit_garch_11(returns: pd.Series) -> GarchFit:
    """Fit zero-mean Gaussian GARCH(1,1) to decimal returns."""
    values = returns.astype(float).dropna().to_numpy()
    if len(values) < 30:
        raise ValueError("at least 30 observations are required")
    if not np.isfinite(values).all():
        raise ValueError("returns must be finite")

    sample_variance = max(float(np.var(values, ddof=1)), 1e-8)

    def decode(parameters: FloatArray) -> tuple[float, float, float]:
        omega = float(np.exp(parameters[0]))
        alpha_scale = float(np.exp(parameters[1]))
        beta_scale = float(np.exp(parameters[2]))
        denominator = 1.0 + alpha_scale + beta_scale
        alpha = 0.999 * alpha_scale / denominator
        beta = 0.999 * beta_scale / denominator
        return omega, alpha, beta

    def objective(parameters: FloatArray) -> float:
        omega, alpha, beta = decode(parameters)
        variance = np.maximum(_variance_path(values, omega, alpha, beta), 1e-12)
        likelihood_terms = np.log(2.0 * np.pi) + np.log(variance) + values**2 / variance
        return float(0.5 * likelihood_terms.sum())

    initial_alpha = 0.08
    initial_beta = 0.87
    initial_residual = 0.999 - initial_alpha - initial_beta
    initial = np.array(
        [
            np.log(sample_variance * (1.0 - initial_alpha - initial_beta)),
            np.log(initial_alpha / initial_residual),
            np.log(initial_beta / initial_residual),
        ]
    )
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=[(-30.0, 0.0), (-12.0, 12.0), (-12.0, 12.0)],
        options={"maxiter": 2_000, "ftol": 1e-12, "gtol": 1e-8},
    )
    omega, alpha, beta = decode(result.x)
    variance = _variance_path(values, omega, alpha, beta)
    index = returns.dropna().index
    return GarchFit(
        omega=omega,
        alpha=alpha,
        beta=beta,
        conditional_variance=pd.Series(variance, index=index, name="conditional_variance"),
        last_squared_return=float(values[-1] ** 2),
        log_likelihood=-float(result.fun),
        converged=bool(result.success),
    )


def forecast_variance(fit: GarchFit, horizon: int) -> pd.Series:
    """Closed-form multi-step variance forecast."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    persistence = fit.persistence
    long_run = fit.unconditional_variance
    last_variance = float(fit.conditional_variance.iloc[-1])
    first_forecast = (
        fit.omega + fit.alpha * fit.last_squared_return + fit.beta * last_variance
    )
    forecasts = [
        long_run + persistence**step * (first_forecast - long_run)
        for step in range(horizon)
    ]
    return pd.Series(forecasts, index=pd.RangeIndex(1, horizon + 1), name="variance_forecast")
