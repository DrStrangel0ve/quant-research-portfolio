"""Policies and transparent baseline agents for Leduc Hold'em."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from quantlab.poker.leduc import Action, LeducState

ACTION_COUNT = len(Action)


class Policy(Protocol):
    """A behavioral policy over the legal actions at a game state."""

    def probabilities(self, state: LeducState) -> NDArray[np.float64]:
        """Return a normalized four-action probability vector."""


@dataclass(frozen=True)
class TabularPolicy:
    """Average CFR policy indexed by RLCard-compatible information states."""

    table: Mapping[str, Sequence[float]]
    information_mode: Literal["compact", "perfect_recall"] = "compact"

    def probabilities(self, state: LeducState) -> NDArray[np.float64]:
        legal = state.legal_actions()
        if not legal:
            raise ValueError("a terminal state has no policy")
        key = state.information_state(
            perfect_recall=self.information_mode == "perfect_recall"
        )
        raw = np.asarray(self.table.get(key, (0.0,) * ACTION_COUNT))
        probabilities = np.zeros(ACTION_COUNT, dtype=float)
        legal_indices = np.asarray([int(action) for action in legal], dtype=int)
        probabilities[legal_indices] = np.maximum(raw[legal_indices], 0.0)
        total = probabilities.sum()
        if total <= 0.0:
            probabilities[legal_indices] = 1.0 / len(legal)
        else:
            probabilities /= total
        return probabilities


@dataclass(frozen=True)
class RandomBot:
    """Uniformly random over legal actions."""

    def probabilities(self, state: LeducState) -> NDArray[np.float64]:
        return _preferred_action_policy(state, ())


@dataclass(frozen=True)
class CallingStationBot:
    """Never folds and never raises: call a bet, otherwise check."""

    def probabilities(self, state: LeducState) -> NDArray[np.float64]:
        return _preferred_action_policy(state, (Action.CALL, Action.CHECK))


@dataclass(frozen=True)
class AggressiveBot:
    """Raise whenever allowed, then call or check."""

    def probabilities(self, state: LeducState) -> NDArray[np.float64]:
        return _preferred_action_policy(
            state,
            (Action.RAISE, Action.CALL, Action.CHECK, Action.FOLD),
        )


def sample_action(
    policy: Policy,
    state: LeducState,
    rng: np.random.Generator,
) -> Action:
    """Sample one legal action from a policy."""
    probabilities = policy.probabilities(state)
    return Action(int(rng.choice(ACTION_COUNT, p=probabilities)))


def _preferred_action_policy(
    state: LeducState,
    preferences: tuple[Action, ...],
) -> NDArray[np.float64]:
    legal = state.legal_actions()
    if not legal:
        raise ValueError("a terminal state has no policy")
    probabilities = np.zeros(ACTION_COUNT, dtype=float)
    for action in preferences:
        if action in legal:
            probabilities[int(action)] = 1.0
            return probabilities
    for action in legal:
        probabilities[int(action)] = 1.0 / len(legal)
    return probabilities
