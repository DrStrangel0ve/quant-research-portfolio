"""Belief-aware rollout resolving for Royal Micro Hold'em.

This module is a transparent bridge toward continual resolving.  It filters an
opponent range through the blueprint's likelihood of observed actions, then
evaluates each legal root action with common-random-number rollouts.  It is not
a safe subgame solver and does not provide ReBeL's theoretical guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from quantlab.poker.deep_cfr import MicroPolicy, policy_probabilities_batch
from quantlab.poker.micro_evaluation import play_micro_hand
from quantlab.poker.micro_holdem import (
    ACTION_COUNT,
    DECK_SIZE,
    MicroDeal,
    MicroHoldemState,
    initial_micro_state,
)


@dataclass(frozen=True)
class RangeEstimate:
    """Posterior opponent hole-card combinations and normalized masses."""

    combos: tuple[tuple[int, int], ...]
    probabilities: NDArray[np.float64]
    effective_sample_size: float


class BeliefRolloutResolver:
    """Choose a legal action by posterior-weighted blueprint rollouts."""

    def __init__(
        self,
        blueprint: MicroPolicy,
        *,
        rollouts_per_action: int = 96,
        seed: int = 15_101,
    ) -> None:
        if rollouts_per_action <= 0:
            raise ValueError("rollouts_per_action must be positive")
        self.blueprint = blueprint
        self.rollouts_per_action = rollouts_per_action
        self.rng = np.random.default_rng(seed)
        self.last_action_values = np.zeros(ACTION_COUNT, dtype=np.float64)
        self.last_range: RangeEstimate | None = None

    def probabilities(self, state: MicroHoldemState) -> NDArray[np.float64]:
        """Return a pure policy selecting the highest estimated root action."""
        observer = state.current_player
        posterior = infer_opponent_range(state, observer=observer, blueprint=self.blueprint)
        self.last_range = posterior
        samples = sample_counterfactual_states(
            state,
            observer,
            posterior,
            self.rng,
            count=self.rollouts_per_action,
        )
        rollout_seeds = self.rng.integers(
            0,
            np.iinfo(np.int64).max,
            size=self.rollouts_per_action,
        )
        action_values = np.full(ACTION_COUNT, -np.inf, dtype=np.float64)
        for action in state.legal_actions():
            values = np.empty(self.rollouts_per_action, dtype=np.float64)
            for index, sampled_state in enumerate(samples):
                child = sampled_state.apply(action)
                values[index] = (
                    child.payoffs()[observer]
                    if child.is_terminal
                    else play_micro_hand(
                        child,
                        (self.blueprint, self.blueprint),
                        np.random.default_rng(int(rollout_seeds[index])),
                    )[observer]
                )
            action_values[int(action)] = values.mean()
        self.last_action_values = action_values
        best = max(state.legal_actions(), key=lambda action: action_values[int(action)])
        probabilities = np.zeros(ACTION_COUNT, dtype=np.float64)
        probabilities[int(best)] = 1.0
        return probabilities


def infer_opponent_range(
    state: MicroHoldemState,
    *,
    observer: int,
    blueprint: MicroPolicy,
) -> RangeEstimate:
    """Bayes-filter opponent combos using public action likelihoods."""
    if observer not in (0, 1):
        raise ValueError("observer must be player zero or one")
    known = set(state.deal.hole_cards[observer]) | set(state.visible_board)
    combos = tuple(combinations((card for card in range(DECK_SIZE) if card not in known), 2))
    replays = [
        _state_for_combo(
            state,
            observer,
            combo,
            np.random.default_rng(index),
            replay_history=False,
        )
        for index, combo in enumerate(combos)
    ]
    weights = np.ones(len(combos), dtype=np.float64)
    for _, observed_action in state.history:
        if replays[0].current_player != observer:
            probabilities = policy_probabilities_batch(blueprint, replays)
            weights *= np.maximum(probabilities[:, int(observed_action)], 1e-9)
        replays = [replay.apply(observed_action) for replay in replays]
    total = weights.sum()
    probabilities = (
        weights / total
        if total > 0.0
        else np.full(len(combos), 1.0 / len(combos), dtype=np.float64)
    )
    effective_sample_size = float(1.0 / np.square(probabilities).sum())
    return RangeEstimate(combos, probabilities, effective_sample_size)


def sample_counterfactual_states(
    state: MicroHoldemState,
    observer: int,
    posterior: RangeEstimate,
    rng: np.random.Generator,
    *,
    count: int,
) -> list[MicroHoldemState]:
    """Systematically sample the posterior so low-variance mass is represented."""
    positions = (float(rng.random()) + np.arange(count)) / count
    indices = np.searchsorted(np.cumsum(posterior.probabilities), positions, side="right")
    return [
        _state_for_combo(state, observer, posterior.combos[int(index)], rng)
        for index in indices
    ]


def _state_for_combo(
    state: MicroHoldemState,
    observer: int,
    opponent_combo: tuple[int, int],
    rng: np.random.Generator,
    *,
    replay_history: bool = True,
) -> MicroHoldemState:
    hole_cards: list[tuple[int, int]] = [(0, 1), (2, 3)]
    hole_cards[observer] = state.deal.hole_cards[observer]
    hole_cards[1 - observer] = opponent_combo
    excluded = set(hole_cards[0]) | set(hole_cards[1])
    if state.visible_board:
        board = state.deal.board
    else:
        available = [card for card in range(DECK_SIZE) if card not in excluded]
        sampled = rng.choice(available, size=3, replace=False)
        board = (int(sampled[0]), int(sampled[1]), int(sampled[2]))
    deal = MicroDeal((hole_cards[0], hole_cards[1]), board)
    replay = initial_micro_state(deal, button=state.button)
    if replay_history:
        for _, action in state.history:
            replay = replay.apply(action)
    return replay
