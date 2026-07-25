"""Information-state features for Royal Micro Hold'em neural policies."""

from __future__ import annotations

from itertools import permutations

import numpy as np
from numpy.typing import NDArray

from quantlab.poker.micro_holdem import (
    ACTION_COUNT,
    MAX_RAISES_PER_STREET,
    STARTING_STACK,
    MicroHoldemState,
    card_rank,
    card_suit,
)

CARD_SLOTS = 5
CARD_FEATURES = 10
BOARD_MASK_SIZE = 3
SCALAR_FEATURES = 12
MAX_HISTORY = 8
HISTORY_FEATURES = 7

CARD_OFFSET = 0
BOARD_MASK_OFFSET = CARD_SLOTS * CARD_FEATURES
SCALAR_OFFSET = BOARD_MASK_OFFSET + BOARD_MASK_SIZE
LEGAL_MASK_OFFSET = SCALAR_OFFSET + SCALAR_FEATURES
HISTORY_OFFSET = LEGAL_MASK_OFFSET + ACTION_COUNT
FEATURE_DIM = HISTORY_OFFSET + MAX_HISTORY * HISTORY_FEATURES
LEGAL_MASK_SLICE = slice(LEGAL_MASK_OFFSET, LEGAL_MASK_OFFSET + ACTION_COUNT)


def encode_information_state(
    state: MicroHoldemState,
    *,
    player: int | None = None,
) -> NDArray[np.float32]:
    """Encode only information observable to ``player`` into 126 floats."""
    observer = state.current_player if player is None else player
    if observer not in (0, 1):
        raise ValueError("player must be zero or one")
    hole, board = canonical_cards(
        state.deal.hole_cards[observer],
        state.visible_board,
    )
    features = np.zeros(FEATURE_DIM, dtype=np.float32)

    for slot, card in enumerate((*hole, *board)):
        start = CARD_OFFSET + slot * CARD_FEATURES
        features[start + card_rank(card)] = 1.0
        features[start + 6 + card_suit(card)] = 1.0
    for index in range(len(board)):
        features[BOARD_MASK_OFFSET + index] = 1.0

    opponent = 1 - observer
    scalars = (
        float(state.street),
        float(observer == state.button),
        state.pot / (2.0 * STARTING_STACK),
        state.contributions[observer] / STARTING_STACK,
        state.contributions[opponent] / STARTING_STACK,
        state.stacks[observer] / STARTING_STACK,
        state.stacks[opponent] / STARTING_STACK,
        state.street_contributions[observer] / STARTING_STACK,
        state.street_contributions[opponent] / STARTING_STACK,
        state.to_call(observer) / STARTING_STACK,
        state.raises / MAX_RAISES_PER_STREET,
        len(state.history) / MAX_HISTORY,
    )
    features[SCALAR_OFFSET : SCALAR_OFFSET + SCALAR_FEATURES] = scalars
    for action in state.legal_actions():
        features[LEGAL_MASK_OFFSET + int(action)] = 1.0

    for index, (street, action) in enumerate(state.history[-MAX_HISTORY:]):
        start = HISTORY_OFFSET + index * HISTORY_FEATURES
        features[start + int(action)] = 1.0
        features[start + ACTION_COUNT + street] = 1.0
    return features


def canonical_cards(
    hole_cards: tuple[int, int],
    board: tuple[int, ...],
) -> tuple[tuple[int, int], tuple[int, ...]]:
    """Canonicalize suit-isomorphic observations by exhaustive relabeling.

    Four suits have only 24 permutations. Taking the lexicographically smallest
    transformed representation avoids subtle order-dependent suit mappings and
    makes the feature vector exactly invariant to every global suit permutation.
    """
    if len(board) not in (0, 3):
        raise ValueError("the visible board must contain zero or three cards")
    best: tuple[int, ...] | None = None
    for suit_map in permutations(range(4)):
        transformed_hole = tuple(
            sorted(card_rank(card) * 4 + suit_map[card_suit(card)] for card in hole_cards)
        )
        transformed_board = tuple(
            sorted(card_rank(card) * 4 + suit_map[card_suit(card)] for card in board)
        )
        candidate = (*transformed_hole, *transformed_board)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise RuntimeError("suit canonicalization did not produce a representation")
    return (best[0], best[1]), tuple(best[2:])
