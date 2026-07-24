"""Avellaneda-Stoikov-inspired inventory-aware market-making simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketMakerParameters:
    volatility: float = 2.0
    risk_aversion: float = 0.1
    fill_intensity: float = 1.5
    fill_decay: float = 1.0
    initial_mid: float = 100.0
    max_inventory: int = 10

    def __post_init__(self) -> None:
        positive = (
            self.volatility,
            self.risk_aversion,
            self.fill_intensity,
            self.fill_decay,
            self.initial_mid,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("market-making parameters must be positive")
        if self.max_inventory <= 0:
            raise ValueError("max_inventory must be positive")


@dataclass(frozen=True)
class MarketMakingResult:
    paths: pd.DataFrame
    terminal_pnl: pd.Series
    terminal_inventory: pd.Series
    mean_pnl: float
    pnl_std: float
    inventory_var_95: float


def simulate_market_maker(
    *,
    parameters: MarketMakerParameters,
    horizon: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> MarketMakingResult:
    """Simulate reservation-price quoting, fills, inventory, and mark-to-market P&L."""
    if horizon <= 0.0 or n_steps <= 0 or n_paths <= 0:
        raise ValueError("horizon, n_steps, and n_paths must be positive")

    dt = horizon / n_steps
    mid = np.full(n_paths, parameters.initial_mid)
    cash = np.zeros(n_paths)
    inventory = np.zeros(n_paths, dtype=int)
    total_fills = np.zeros(n_paths, dtype=int)

    for step in range(n_steps):
        time_remaining = horizon - step * dt
        reservation = (
            mid
            - inventory
            * parameters.risk_aversion
            * parameters.volatility**2
            * time_remaining
        )
        half_spread = (
            0.5
            * parameters.risk_aversion
            * parameters.volatility**2
            * time_remaining
            + np.log1p(parameters.risk_aversion / parameters.fill_decay)
            / parameters.risk_aversion
        )
        bid = reservation - half_spread
        ask = reservation + half_spread
        bid_distance = np.maximum(mid - bid, 0.0)
        ask_distance = np.maximum(ask - mid, 0.0)
        bid_probability = 1.0 - np.exp(
            -parameters.fill_intensity * np.exp(-parameters.fill_decay * bid_distance) * dt
        )
        ask_probability = 1.0 - np.exp(
            -parameters.fill_intensity * np.exp(-parameters.fill_decay * ask_distance) * dt
        )

        bid_fills = (rng.random(n_paths) < bid_probability) & (
            inventory < parameters.max_inventory
        )
        ask_fills = (rng.random(n_paths) < ask_probability) & (
            inventory > -parameters.max_inventory
        )
        inventory += bid_fills.astype(int) - ask_fills.astype(int)
        cash -= bid * bid_fills
        cash += ask * ask_fills
        total_fills += bid_fills.astype(int) + ask_fills.astype(int)

        mid += parameters.volatility * np.sqrt(dt) * rng.standard_normal(n_paths)

    terminal_pnl_values = cash + inventory * mid
    paths = pd.DataFrame(
        {
            "terminal_mid": mid,
            "terminal_cash": cash,
            "terminal_inventory": inventory,
            "fill_count": total_fills,
            "terminal_pnl": terminal_pnl_values,
        }
    )
    terminal_pnl = paths["terminal_pnl"]
    terminal_inventory = paths["terminal_inventory"]
    return MarketMakingResult(
        paths=paths,
        terminal_pnl=terminal_pnl,
        terminal_inventory=terminal_inventory,
        mean_pnl=float(terminal_pnl.mean()),
        pnl_std=float(terminal_pnl.std(ddof=1)),
        inventory_var_95=float(np.quantile(np.abs(terminal_inventory), 0.95)),
    )
