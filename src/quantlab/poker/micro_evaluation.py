"""Duplicate-poker evaluation for Royal Micro Hold'em policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quantlab.poker.deep_cfr import MicroPolicy
from quantlab.poker.micro_holdem import (
    MicroDeal,
    MicroHoldemState,
    initial_micro_state,
    sample_micro_state,
)


@dataclass(frozen=True)
class MicroMatchResult:
    """Seat-swapped paired estimate for the first named policy."""

    mean_big_blinds: float
    standard_error: float
    ci95_low: float
    ci95_high: float
    duplicate_pairs: int


@dataclass(frozen=True)
class PolicyImprovementResult:
    """Paired difference between candidate and baseline against one opponent."""

    candidate_mean_big_blinds: float
    baseline_mean_big_blinds: float
    mean_improvement_big_blinds: float
    standard_error: float
    ci95_low: float
    ci95_high: float
    duplicate_pairs: int


def duplicate_micro_match(
    policy_a: MicroPolicy,
    policy_b: MicroPolicy,
    *,
    duplicate_pairs: int,
    rng: np.random.Generator,
) -> MicroMatchResult:
    """Play identical cards in both seat orientations and form a paired CI."""
    if duplicate_pairs <= 1:
        raise ValueError("duplicate_pairs must exceed one")
    pair_values = np.empty(duplicate_pairs, dtype=np.float64)
    for index in range(duplicate_pairs):
        root = sample_micro_state(rng)
        pair_values[index] = _duplicate_pair_value(policy_a, policy_b, root, rng)
    mean, standard_error, ci95_low, ci95_high = _paired_summary(pair_values)
    return MicroMatchResult(
        mean_big_blinds=mean,
        standard_error=standard_error,
        ci95_low=ci95_low,
        ci95_high=ci95_high,
        duplicate_pairs=duplicate_pairs,
    )


def paired_policy_improvement(
    candidate: MicroPolicy,
    baseline: MicroPolicy,
    opponent: MicroPolicy,
    *,
    duplicate_pairs: int,
    rng: np.random.Generator,
) -> PolicyImprovementResult:
    """Estimate candidate-minus-baseline value on identical deals and RNG streams."""
    if duplicate_pairs <= 1:
        raise ValueError("duplicate_pairs must exceed one")
    candidate_values = np.empty(duplicate_pairs, dtype=np.float64)
    baseline_values = np.empty(duplicate_pairs, dtype=np.float64)
    differences = np.empty(duplicate_pairs, dtype=np.float64)
    for index in range(duplicate_pairs):
        root = sample_micro_state(rng)
        rollout_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        candidate_values[index] = _duplicate_pair_value(
            candidate,
            opponent,
            root,
            np.random.default_rng(rollout_seed),
        )
        baseline_values[index] = _duplicate_pair_value(
            baseline,
            opponent,
            root,
            np.random.default_rng(rollout_seed),
        )
        differences[index] = candidate_values[index] - baseline_values[index]
    mean, standard_error, ci95_low, ci95_high = _paired_summary(differences)
    return PolicyImprovementResult(
        candidate_mean_big_blinds=float(candidate_values.mean()),
        baseline_mean_big_blinds=float(baseline_values.mean()),
        mean_improvement_big_blinds=mean,
        standard_error=standard_error,
        ci95_low=ci95_low,
        ci95_high=ci95_high,
        duplicate_pairs=duplicate_pairs,
    )


def play_micro_hand(
    state: MicroHoldemState,
    policies: tuple[MicroPolicy, MicroPolicy],
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Sample one complete hand from a behavioral-policy profile."""
    while not state.is_terminal:
        probabilities = policies[state.current_player].probabilities(state)
        legal = state.legal_actions()
        local = np.asarray([probabilities[int(action)] for action in legal], dtype=float)
        if not np.isclose(local.sum(), 1.0):
            raise ValueError("policy probabilities must sum to one over legal actions")
        action = legal[int(rng.choice(len(legal), p=local))]
        state = state.apply(action)
    return state.payoffs()


def _duplicate_pair_value(
    policy_a: MicroPolicy,
    policy_b: MicroPolicy,
    root: MicroHoldemState,
    rng: np.random.Generator,
) -> float:
    first = play_micro_hand(root, (policy_a, policy_b), rng)[0]
    mirrored = MicroDeal(
        (root.deal.hole_cards[1], root.deal.hole_cards[0]),
        root.deal.board,
    )
    second = play_micro_hand(
        initial_micro_state(mirrored, button=1 - root.button),
        (policy_b, policy_a),
        rng,
    )[1]
    return (first + second) / 2.0


def _paired_summary(
    values: NDArray[np.float64],
) -> tuple[float, float, float, float]:
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
    half_width = 1.96 * standard_error
    return mean, standard_error, mean - half_width, mean + half_width
