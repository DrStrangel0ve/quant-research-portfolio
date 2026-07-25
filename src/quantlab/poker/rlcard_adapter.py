"""Optional bridge to RLCard's published pretrained Leduc CFR checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from quantlab.poker.agents import TabularPolicy


def load_rlcard_reference_policy() -> TabularPolicy:
    """Convert RLCard's bundled CFR checkpoint into QuantLab's policy format.

    RLCard is deliberately an optional benchmark dependency: the game engine,
    trainer, evaluator, and playable artifact do not rely on it.
    """
    try:
        import rlcard.models
    except ImportError as error:
        raise ImportError(
            "install the poker benchmark extra with `pip install -e .[poker]`"
        ) from error

    model = rlcard.models.load("leduc-holdem-cfr")
    agent = model.agents[0]
    average_policy: Mapping[bytes, Any] = agent.average_policy
    table: dict[str, list[float]] = {}
    for observation_bytes, weights in average_policy.items():
        observation = np.frombuffer(observation_bytes, dtype=np.float64)
        key = information_key_from_observation(observation)
        table[key] = np.asarray(weights, dtype=float).tolist()
    return TabularPolicy(table)


def information_key_from_observation(observation: np.ndarray[Any, Any]) -> str:
    """Decode RLCard's documented 36-element Leduc observation vector."""
    if observation.shape != (36,):
        raise ValueError("RLCard Leduc observations must contain 36 elements")
    hand_locations = np.flatnonzero(observation[:3])
    board_locations = np.flatnonzero(observation[3:6])
    my_chip_locations = np.flatnonzero(observation[6:21])
    opponent_chip_locations = np.flatnonzero(observation[21:36])
    if len(hand_locations) != 1 or len(my_chip_locations) != 1:
        raise ValueError("observation does not encode exactly one hand and chip count")
    if len(opponent_chip_locations) != 1 or len(board_locations) > 1:
        raise ValueError("observation has an invalid board or opponent chip count")
    hand = int(hand_locations[0])
    board = int(board_locations[0]) if len(board_locations) else -1
    my_chips = int(my_chip_locations[0])
    opponent_chips = int(opponent_chip_locations[0])
    return f"h{hand}|b{board}|m{my_chips}|o{opponent_chips}"
