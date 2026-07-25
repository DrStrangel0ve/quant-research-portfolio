"""Duplicate-poker evaluation for Royal Micro Hold'em policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
        pair_values[index] = (first + second) / 2.0

    mean = float(pair_values.mean())
    standard_error = float(pair_values.std(ddof=1) / np.sqrt(duplicate_pairs))
    half_width = 1.96 * standard_error
    return MicroMatchResult(
        mean_big_blinds=mean,
        standard_error=standard_error,
        ci95_low=mean - half_width,
        ci95_high=mean + half_width,
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
