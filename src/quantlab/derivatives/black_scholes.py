"""Black-Scholes analytics and Monte Carlo European option pricing."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.stats import norm

from quantlab.simulation.monte_carlo import (
    MonteCarloEstimate,
    estimate_discounted_payoff,
    geometric_brownian_motion,
)

OptionType = Literal["call", "put"]


def _validate_inputs(
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
    option_type: OptionType,
) -> None:
    if spot <= 0.0 or strike <= 0.0 or maturity <= 0.0:
        raise ValueError("spot, strike, and maturity must be positive")
    if volatility <= 0.0:
        raise ValueError("volatility must be positive")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")


def black_scholes_price(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    option_type: OptionType = "call",
) -> float:
    """Closed-form European option value."""
    _validate_inputs(spot, strike, maturity, volatility, option_type)
    sqrt_t = np.sqrt(maturity)
    d1 = (
        np.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility**2) * maturity
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    discounted_spot = spot * np.exp(-dividend_yield * maturity)
    discounted_strike = strike * np.exp(-rate * maturity)
    if option_type == "call":
        return float(discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2))
    return float(discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1))


def black_scholes_delta(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    option_type: OptionType = "call",
) -> float:
    """Closed-form spot delta."""
    _validate_inputs(spot, strike, maturity, volatility, option_type)
    d1 = (
        np.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility**2) * maturity
    ) / (volatility * np.sqrt(maturity))
    discount = np.exp(-dividend_yield * maturity)
    if option_type == "call":
        return float(discount * norm.cdf(d1))
    return float(discount * (norm.cdf(d1) - 1.0))


def european_option_monte_carlo(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    n_paths: int,
    rng: np.random.Generator,
    dividend_yield: float = 0.0,
    option_type: OptionType = "call",
    antithetic: bool = True,
) -> MonteCarloEstimate:
    """Risk-neutral Monte Carlo value of a European option."""
    _validate_inputs(spot, strike, maturity, volatility, option_type)
    paths = geometric_brownian_motion(
        spot=spot,
        drift=rate - dividend_yield,
        volatility=volatility,
        maturity=maturity,
        n_steps=1,
        n_paths=n_paths,
        rng=rng,
        antithetic=antithetic,
    )
    terminal = paths[-1]
    if option_type == "call":
        payoff = np.maximum(terminal - strike, 0.0)
    else:
        payoff = np.maximum(strike - terminal, 0.0)
    discounted = np.asarray(np.exp(-rate * maturity) * payoff, dtype=np.float64)
    return estimate_discounted_payoff(discounted)
