"""A bounded heads-up no-limit Hold'em research game.

Royal Micro Hold'em keeps the strategically important mechanics of heads-up
no-limit poker while remaining small enough for laptop-scale neural CFR
experiments:

* a 24-card deck (9 through A in four suits),
* two private cards per player and a three-card flop,
* 20-chip effective stacks with one/two-chip blinds,
* preflop and flop betting, and
* fold, check/call, half-pot, pot, and all-in actions.

The game is intentionally synthetic.  It is a scaling benchmark between Leduc
and full heads-up no-limit Texas Hold'em, not a claim about real-money HUNL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import IntEnum

import numpy as np

RANK_NAMES = ("9", "T", "J", "Q", "K", "A")
SUIT_NAMES = ("c", "d", "h", "s")
DECK_SIZE = 24
STARTING_STACK = 20
BIG_BLIND = 2
MAX_RAISES_PER_STREET = 2


class MicroAction(IntEnum):
    """Fixed action abstraction used by the engine and neural networks."""

    FOLD = 0
    CHECK_CALL = 1
    HALF_POT = 2
    POT = 3
    ALL_IN = 4


ACTION_NAMES = {
    MicroAction.FOLD: "fold",
    MicroAction.CHECK_CALL: "check/call",
    MicroAction.HALF_POT: "half-pot",
    MicroAction.POT: "pot",
    MicroAction.ALL_IN: "all-in",
}
ACTION_COUNT = len(MicroAction)


@dataclass(frozen=True)
class MicroDeal:
    """A complete chance outcome whose hidden cards stay private to the engine."""

    hole_cards: tuple[tuple[int, int], tuple[int, int]]
    board: tuple[int, int, int]

    def __post_init__(self) -> None:
        cards = (*self.hole_cards[0], *self.hole_cards[1], *self.board)
        if len(cards) != 7 or len(set(cards)) != 7:
            raise ValueError("a deal must contain seven distinct cards")
        if any(card not in range(DECK_SIZE) for card in cards):
            raise ValueError("card ids must be integers from zero through twenty-three")


@dataclass(frozen=True)
class MicroHoldemState:
    """Immutable public game state plus a complete hidden chance outcome."""

    deal: MicroDeal
    button: int
    current_player: int
    street: int
    contributions: tuple[int, int]
    street_contributions: tuple[int, int]
    acted: tuple[bool, bool] = (False, False)
    raises: int = 0
    folded_player: int | None = None
    showdown: bool = False
    history: tuple[tuple[int, MicroAction], ...] = ()

    @property
    def is_terminal(self) -> bool:
        return self.folded_player is not None or self.showdown

    @property
    def pot(self) -> int:
        return sum(self.contributions)

    @property
    def stacks(self) -> tuple[int, int]:
        return (
            STARTING_STACK - self.contributions[0],
            STARTING_STACK - self.contributions[1],
        )

    @property
    def visible_board(self) -> tuple[int, ...]:
        if self.street == 0 and not self.showdown:
            return ()
        return self.deal.board

    def to_call(self, player: int | None = None) -> int:
        observer = self.current_player if player is None else player
        _validate_player(observer)
        return max(self.street_contributions) - self.street_contributions[observer]

    def legal_actions(self) -> tuple[MicroAction, ...]:
        if self.is_terminal:
            return ()
        player = self.current_player
        opponent = 1 - player
        stack = self.stacks[player]
        outstanding = self.to_call(player)
        actions: list[MicroAction] = []
        if outstanding > 0:
            actions.extend((MicroAction.FOLD, MicroAction.CHECK_CALL))
        else:
            actions.append(MicroAction.CHECK_CALL)

        can_raise = (
            stack > outstanding
            and self.stacks[opponent] > 0
            and self.raises < MAX_RAISES_PER_STREET
        )
        if not can_raise:
            return tuple(actions)

        all_in_amount = stack
        seen_amounts: set[int] = set()
        for action in (MicroAction.HALF_POT, MicroAction.POT):
            amount = self._chips_for(action)
            if amount <= outstanding or amount >= all_in_amount or amount in seen_amounts:
                continue
            actions.append(action)
            seen_amounts.add(amount)
        if all_in_amount > outstanding:
            actions.append(MicroAction.ALL_IN)
        return tuple(actions)

    def apply(self, action: MicroAction) -> MicroHoldemState:
        """Apply one abstract action and return the resulting immutable state."""
        if action not in self.legal_actions():
            legal = ", ".join(ACTION_NAMES[item] for item in self.legal_actions())
            raise ValueError(f"{ACTION_NAMES.get(action, str(action))} is illegal; choose {legal}")

        player = self.current_player
        opponent = 1 - player
        history = (*self.history, (self.street, action))
        if action == MicroAction.FOLD:
            return replace(
                self,
                current_player=opponent,
                folded_player=player,
                history=history,
            )

        contributions = list(self.contributions)
        street_contributions = list(self.street_contributions)
        chips = self._chips_for(action)
        contributions[player] += chips
        street_contributions[player] += chips
        acted = list(self.acted)
        raises = self.raises
        if action in (MicroAction.HALF_POT, MicroAction.POT, MicroAction.ALL_IN):
            acted = [False, False]
            acted[player] = True
            raises += 1
        else:
            acted[player] = True

        advanced = replace(
            self,
            current_player=opponent,
            contributions=(contributions[0], contributions[1]),
            street_contributions=(street_contributions[0], street_contributions[1]),
            acted=(acted[0], acted[1]),
            raises=raises,
            history=history,
        )
        balanced = advanced.street_contributions[0] == advanced.street_contributions[1]
        if not (balanced and all(advanced.acted)):
            return advanced
        if 0 in advanced.stacks or self.street == 1:
            return replace(advanced, showdown=True)
        return replace(
            advanced,
            current_player=1 - self.button,
            street=1,
            street_contributions=(0, 0),
            acted=(False, False),
            raises=0,
        )

    def payoffs(self) -> tuple[float, float]:
        """Return zero-sum net utility in big-blind units."""
        if not self.is_terminal:
            raise ValueError("payoffs are only defined at terminal states")
        winners: tuple[int, ...]
        if self.folded_player is not None:
            winners = (1 - self.folded_player,)
        else:
            scores = tuple(
                evaluate_five((*self.deal.hole_cards[player], *self.deal.board))
                for player in (0, 1)
            )
            best = max(scores)
            winners = tuple(player for player, score in enumerate(scores) if score == best)

        shares = [0.0, 0.0]
        for winner in winners:
            shares[winner] = self.pot / len(winners)
        values = tuple(
            (shares[player] - self.contributions[player]) / BIG_BLIND
            for player in (0, 1)
        )
        return values[0], values[1]

    def _chips_for(self, action: MicroAction) -> int:
        player = self.current_player
        stack = self.stacks[player]
        outstanding = min(self.to_call(player), stack)
        if action == MicroAction.CHECK_CALL:
            return outstanding
        if action == MicroAction.ALL_IN:
            return stack
        if action not in (MicroAction.HALF_POT, MicroAction.POT):
            return 0
        pot_after_call = self.pot + outstanding
        fraction = 0.5 if action == MicroAction.HALF_POT else 1.0
        raise_increment = max(1, math.ceil(fraction * pot_after_call))
        return min(stack, outstanding + raise_increment)


def initial_micro_state(deal: MicroDeal, *, button: int) -> MicroHoldemState:
    """Post one/two-chip blinds; the button/small blind acts first preflop."""
    _validate_player(button)
    contributions = [2, 2]
    contributions[button] = 1
    return MicroHoldemState(
        deal=deal,
        button=button,
        current_player=button,
        street=0,
        contributions=(contributions[0], contributions[1]),
        street_contributions=(contributions[0], contributions[1]),
    )


def sample_micro_state(
    rng: np.random.Generator,
    *,
    button: int | None = None,
) -> MicroHoldemState:
    """Sample a complete seven-card deal and optional button assignment."""
    cards = rng.choice(DECK_SIZE, size=7, replace=False)
    deal = MicroDeal(
        (
            (int(cards[0]), int(cards[1])),
            (int(cards[2]), int(cards[3])),
        ),
        (int(cards[4]), int(cards[5]), int(cards[6])),
    )
    assigned_button = int(rng.integers(0, 2)) if button is None else button
    return initial_micro_state(deal, button=assigned_button)


def evaluate_five(cards: tuple[int, ...]) -> tuple[int, ...]:
    """Evaluate exactly five cards; larger tuples compare as stronger hands."""
    if len(cards) != 5 or len(set(cards)) != 5:
        raise ValueError("evaluate_five requires five distinct cards")
    ranks = [card_rank(card) for card in cards]
    suits = [card_suit(card) for card in cards]
    counts = {rank: ranks.count(rank) for rank in set(ranks)}
    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    flush = len(set(suits)) == 1
    unique_ranks = sorted(set(ranks))
    straight = len(unique_ranks) == 5 and unique_ranks[-1] - unique_ranks[0] == 4
    straight_high = unique_ranks[-1] if straight else -1

    if straight and flush:
        return (8, straight_high)
    if groups[0][0] == 4:
        return (7, groups[0][1], groups[1][1])
    if groups[0][0] == 3 and groups[1][0] == 2:
        return (6, groups[0][1], groups[1][1])
    if flush:
        return (5, *sorted(ranks, reverse=True))
    if straight:
        return (4, straight_high)
    if groups[0][0] == 3:
        kickers = sorted((rank for rank in ranks if rank != groups[0][1]), reverse=True)
        return (3, groups[0][1], *kickers)
    pairs = sorted((rank for rank, count in counts.items() if count == 2), reverse=True)
    if len(pairs) == 2:
        kicker = next(rank for rank, count in counts.items() if count == 1)
        return (2, pairs[0], pairs[1], kicker)
    if len(pairs) == 1:
        kickers = sorted((rank for rank in ranks if rank != pairs[0]), reverse=True)
        return (1, pairs[0], *kickers)
    return (0, *sorted(ranks, reverse=True))


def card_rank(card: int) -> int:
    """Return the zero-based rank 9=0 through A=5."""
    _validate_card(card)
    return card // 4


def card_suit(card: int) -> int:
    """Return the zero-based suit c=0, d=1, h=2, s=3."""
    _validate_card(card)
    return card % 4


def card_label(card: int) -> str:
    """Return a compact rank/suit label such as ``Ah``."""
    return f"{RANK_NAMES[card_rank(card)]}{SUIT_NAMES[card_suit(card)]}"


def _validate_card(card: int) -> None:
    if card not in range(DECK_SIZE):
        raise ValueError("card must be an integer from zero through twenty-three")


def _validate_player(player: int) -> None:
    if player not in (0, 1):
        raise ValueError("player must be zero or one")
