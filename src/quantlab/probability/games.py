"""Classic probability games with analytic benchmarks and seeded simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SimulationRate:
    successes: int
    trials: int

    @property
    def estimate(self) -> float:
        return self.successes / self.trials

    @property
    def standard_error(self) -> float:
        probability = self.estimate
        return float(np.sqrt(probability * (1.0 - probability) / self.trials))


def gambler_ruin_probability(
    *,
    initial_wealth: int,
    target_wealth: int,
    win_probability: float,
) -> float:
    """Exact probability of hitting the target before ruin."""
    if not 0 < initial_wealth < target_wealth:
        raise ValueError("require 0 < initial_wealth < target_wealth")
    if not 0.0 < win_probability < 1.0:
        raise ValueError("win_probability must be between zero and one")
    if np.isclose(win_probability, 0.5):
        return initial_wealth / target_wealth
    loss_probability = 1.0 - win_probability
    ratio = loss_probability / win_probability
    return float((1.0 - ratio**initial_wealth) / (1.0 - ratio**target_wealth))


def simulate_gambler_ruin(
    *,
    initial_wealth: int,
    target_wealth: int,
    win_probability: float,
    n_trials: int,
    rng: np.random.Generator,
) -> SimulationRate:
    """Monte Carlo estimate using vectorized absorbing random walks."""
    gambler_ruin_probability(
        initial_wealth=initial_wealth,
        target_wealth=target_wealth,
        win_probability=win_probability,
    )
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    wealth = np.full(n_trials, initial_wealth, dtype=int)
    active = np.ones(n_trials, dtype=bool)
    while active.any():
        indices = np.flatnonzero(active)
        outcomes = rng.random(len(indices)) < win_probability
        wealth[indices] += np.where(outcomes, 1, -1)
        active = (wealth > 0) & (wealth < target_wealth)
    return SimulationRate(successes=int((wealth == target_wealth).sum()), trials=n_trials)


def kelly_fraction(
    *,
    win_probability: float,
    net_win_multiple: float,
    net_loss_multiple: float = 1.0,
) -> float:
    """Log-optimal bankroll fraction for a binary gamble."""
    if not 0.0 < win_probability < 1.0:
        raise ValueError("win_probability must be between zero and one")
    if net_win_multiple <= 0.0 or net_loss_multiple <= 0.0:
        raise ValueError("payoff multiples must be positive")
    loss_probability = 1.0 - win_probability
    fraction = (
        win_probability / net_loss_multiple
        - loss_probability / net_win_multiple
    )
    return float(max(fraction, 0.0))


def simulate_kelly_wealth(
    *,
    initial_wealth: float,
    fraction: float,
    win_probability: float,
    net_win_multiple: float,
    net_loss_multiple: float,
    n_bets: int,
    n_paths: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Simulate terminal bankrolls under fixed-fraction betting."""
    if initial_wealth <= 0.0 or n_bets <= 0 or n_paths <= 0:
        raise ValueError("wealth, n_bets, and n_paths must be positive")
    if not 0.0 <= fraction <= 1.0 / net_loss_multiple:
        raise ValueError("fraction risks more than the entire bankroll")
    outcomes = rng.random((n_bets, n_paths)) < win_probability
    multipliers = np.where(
        outcomes,
        1.0 + fraction * net_win_multiple,
        1.0 - fraction * net_loss_multiple,
    )
    return np.asarray(initial_wealth * multipliers.prod(axis=0), dtype=float)


def monty_hall(
    *,
    switch: bool,
    n_trials: int,
    rng: np.random.Generator,
) -> SimulationRate:
    """Simulate the canonical three-door Monty Hall game."""
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    prize = rng.integers(0, 3, size=n_trials)
    initial_choice = rng.integers(0, 3, size=n_trials)
    if switch:
        successes = int((initial_choice != prize).sum())
    else:
        successes = int((initial_choice == prize).sum())
    return SimulationRate(successes=successes, trials=n_trials)


def secretary_game(
    *,
    n_candidates: int,
    sample_fraction: float,
    n_trials: int,
    rng: np.random.Generator,
) -> SimulationRate:
    """Estimate success of the reject-then-select-best-so-far rule."""
    if n_candidates < 2 or n_trials <= 0:
        raise ValueError("at least two candidates and one trial are required")
    if not 0.0 < sample_fraction < 1.0:
        raise ValueError("sample_fraction must be between zero and one")
    sample_count = min(max(int(n_candidates * sample_fraction), 1), n_candidates - 1)
    successes = 0
    for _ in range(n_trials):
        ordering = rng.permutation(n_candidates)
        threshold = ordering[:sample_count].max()
        eligible = np.flatnonzero(ordering[sample_count:] > threshold)
        selected_index = (
            sample_count + int(eligible[0]) if len(eligible) else n_candidates - 1
        )
        successes += int(ordering[selected_index] == n_candidates - 1)
    return SimulationRate(successes=successes, trials=n_trials)
