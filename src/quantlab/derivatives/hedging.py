"""Discrete delta-hedging experiments with transaction costs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from quantlab.derivatives.black_scholes import black_scholes_delta, black_scholes_price
from quantlab.simulation.monte_carlo import geometric_brownian_motion


@dataclass(frozen=True)
class HedgingResult:
    errors: pd.Series
    costs: pd.Series
    mean_error: float
    error_std: float
    expected_shortfall_95: float


def simulate_delta_hedge(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    implied_volatility: float,
    realized_volatility: float,
    n_rebalances: int,
    n_paths: int,
    transaction_cost_bps: float,
    rng: np.random.Generator,
) -> HedgingResult:
    """Simulate a self-financing short-call hedge under GBM.

    The dealer receives the Black-Scholes premium, buys delta shares, accrues
    cash at the risk-free rate, and pays proportional costs on each rebalance.
    """
    if transaction_cost_bps < 0.0:
        raise ValueError("transaction_cost_bps cannot be negative")
    paths = geometric_brownian_motion(
        spot=spot,
        drift=rate,
        volatility=realized_volatility,
        maturity=maturity,
        n_steps=n_rebalances,
        n_paths=n_paths,
        rng=rng,
        antithetic=True,
    )
    dt = maturity / n_rebalances
    premium = black_scholes_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=implied_volatility,
    )
    initial_delta = black_scholes_delta(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=implied_volatility,
    )
    shares = np.full(n_paths, initial_delta)
    initial_cost = transaction_cost_bps / 10_000 * abs(initial_delta) * spot
    cash = np.full(n_paths, premium - initial_delta * spot - initial_cost)
    accumulated_costs = np.full(n_paths, initial_cost)

    for step in range(1, n_rebalances):
        cash *= np.exp(rate * dt)
        time_remaining = maturity - step * dt
        d1 = (
            np.log(paths[step] / strike)
            + (rate + 0.5 * implied_volatility**2) * time_remaining
        ) / (implied_volatility * np.sqrt(time_remaining))
        new_delta = np.asarray(
            norm.cdf(d1),
            dtype=float,
        )
        trade = new_delta - shares
        trading_cost = transaction_cost_bps / 10_000 * np.abs(trade) * paths[step]
        cash -= trade * paths[step] + trading_cost
        accumulated_costs += trading_cost
        shares = new_delta

    cash *= np.exp(rate * dt)
    terminal = paths[-1]
    payoff = np.maximum(terminal - strike, 0.0)
    final_liquidation_cost = transaction_cost_bps / 10_000 * np.abs(shares) * terminal
    errors = cash + shares * terminal - payoff - final_liquidation_cost
    accumulated_costs += final_liquidation_cost
    error_series = pd.Series(errors, name="hedging_error")
    cost_series = pd.Series(accumulated_costs, name="transaction_cost")
    tail_cutoff = error_series.quantile(0.05)
    expected_shortfall = -float(error_series[error_series <= tail_cutoff].mean())
    return HedgingResult(
        errors=error_series,
        costs=cost_series,
        mean_error=float(error_series.mean()),
        error_std=float(error_series.std(ddof=1)),
        expected_shortfall_95=expected_shortfall,
    )
