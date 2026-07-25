"""A from-scratch chance-sampled CFR+ implementation for Leduc Hold'em."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from quantlab.poker.agents import ACTION_COUNT, TabularPolicy
from quantlab.poker.leduc import Deal, LeducState, initial_state


class CFRPlusTrainer:
    """Alternating chance-sampled CFR+ with linearly weighted averaging.

    No poker or learning framework is used by the trainer. Each iteration
    samples a blind assignment and three distinct cards, then traverses every
    legal action for each updating player. Positive regret matching produces
    the current policy; cumulative regrets are floored at zero as in CFR+.
    """

    def __init__(
        self,
        *,
        seed: int = 13_001,
        information_mode: Literal["compact", "perfect_recall"] = "perfect_recall",
    ) -> None:
        self.seed = seed
        self.information_mode = information_mode
        self.rng = np.random.default_rng(seed)
        self.iteration = 0
        self.regrets: dict[str, NDArray[np.float64]] = defaultdict(
            lambda: np.zeros(ACTION_COUNT, dtype=float)
        )
        self.strategy_sums: dict[str, NDArray[np.float64]] = defaultdict(
            lambda: np.zeros(ACTION_COUNT, dtype=float)
        )

    def train(self, iterations: int) -> TabularPolicy:
        """Run additional iterations and return the linearly averaged policy."""
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        for _ in range(iterations):
            self.iteration += 1
            state = self._sample_initial_state()
            for traverser in (0, 1):
                self._traverse(
                    state,
                    traverser=traverser,
                    reach=(1.0, 1.0),
                )
        return self.average_policy()

    def average_policy(self) -> TabularPolicy:
        """Normalize accumulated realization-weighted strategies."""
        table: dict[str, list[float]] = {}
        for key, strategy_sum in self.strategy_sums.items():
            total = strategy_sum.sum()
            if total > 0.0:
                table[key] = (strategy_sum / total).tolist()
            else:
                table[key] = [0.0] * ACTION_COUNT
        return TabularPolicy(table, information_mode=self.information_mode)

    def save(self, path: Path) -> None:
        """Save a reproducible, human-readable training checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "format": "quantlab-leduc-cfr-plus-v1",
            "seed": self.seed,
            "iterations": self.iteration,
            "information_mode": self.information_mode,
            "algorithm": {
                "name": "chance-sampled CFR+",
                "regret_floor": 0.0,
                "average_weighting": "linear",
            },
            "policy": self.average_policy().table,
            "regrets": {key: values.tolist() for key, values in self.regrets.items()},
            "strategy_sums": {
                key: values.tolist() for key, values in self.strategy_sums.items()
            },
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CFRPlusTrainer:
        """Restore a checkpoint for evaluation or continued training."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "quantlab-leduc-cfr-plus-v1":
            raise ValueError("unsupported checkpoint format")
        trainer = cls(
            seed=int(payload["seed"]),
            information_mode=payload.get("information_mode", "compact"),
        )
        trainer.iteration = int(payload["iterations"])
        for key, values in payload["regrets"].items():
            trainer.regrets[key] = np.asarray(values, dtype=float)
        for key, values in payload["strategy_sums"].items():
            trainer.strategy_sums[key] = np.asarray(values, dtype=float)
        return trainer

    def _sample_initial_state(self) -> LeducState:
        cards = self.rng.choice(6, size=3, replace=False)
        deal = Deal(
            (int(cards[0]), int(cards[1])),
            int(cards[2]),
        )
        small_blind = int(self.rng.integers(0, 2))
        return initial_state(deal, small_blind=small_blind)

    def _traverse(
        self,
        state: LeducState,
        *,
        traverser: int,
        reach: tuple[float, float],
    ) -> float:
        if state.is_terminal:
            return state.payoffs()[traverser]

        player = state.current_player
        key = state.information_state(
            perfect_recall=self.information_mode == "perfect_recall"
        )
        strategy = self._regret_matching(key, state)
        legal = state.legal_actions()
        action_values = np.zeros(ACTION_COUNT, dtype=float)
        node_value = 0.0

        for action in legal:
            next_reach = list(reach)
            next_reach[player] *= strategy[int(action)]
            value = self._traverse(
                state.apply(action),
                traverser=traverser,
                reach=(next_reach[0], next_reach[1]),
            )
            action_values[int(action)] = value
            node_value += strategy[int(action)] * value

        if player == traverser:
            opponent_reach = reach[1 - player]
            regret_delta = opponent_reach * (action_values - node_value)
            legal_indices = np.asarray([int(action) for action in legal], dtype=int)
            updated = self.regrets[key].copy()
            updated[legal_indices] = np.maximum(
                updated[legal_indices] + regret_delta[legal_indices],
                0.0,
            )
            self.regrets[key] = updated
            self.strategy_sums[key] += self.iteration * reach[player] * strategy
        return float(node_value)

    def _regret_matching(
        self,
        key: str,
        state: LeducState,
    ) -> NDArray[np.float64]:
        legal = state.legal_actions()
        legal_indices = np.asarray([int(action) for action in legal], dtype=int)
        strategy = np.zeros(ACTION_COUNT, dtype=float)
        positive = np.maximum(self.regrets[key][legal_indices], 0.0)
        total = positive.sum()
        if total > 0.0:
            strategy[legal_indices] = positive / total
        else:
            strategy[legal_indices] = 1.0 / len(legal)
        return strategy
