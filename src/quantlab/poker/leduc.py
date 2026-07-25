"""RLCard-compatible heads-up limit Leduc Hold'em.

The six-card deck contains two suits of each rank (J, Q, K). Each player
receives one private card, one public card is revealed after the first betting
round, and a public/private rank match makes a pair. Payoffs are expressed in
big-blind units, matching RLCard's environment.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import IntEnum
from itertools import permutations


class Action(IntEnum):
    """Action identifiers intentionally match RLCard's Leduc environment."""

    CALL = 0
    RAISE = 1
    FOLD = 2
    CHECK = 3


RANK_NAMES = ("J", "Q", "K")
ACTION_NAMES = {
    Action.CALL: "call",
    Action.RAISE: "raise",
    Action.FOLD: "fold",
    Action.CHECK: "check",
}


@dataclass(frozen=True)
class Deal:
    """A complete chance outcome; the public card remains hidden until round two."""

    private_cards: tuple[int, int]
    public_card: int

    def __post_init__(self) -> None:
        cards = (*self.private_cards, self.public_card)
        if len(set(cards)) != 3 or any(card not in range(6) for card in cards):
            raise ValueError("a deal must contain three distinct cards from the six-card deck")


@dataclass(frozen=True)
class LeducState:
    """Immutable game state suitable for tree traversal and exact evaluation."""

    deal: Deal
    small_blind: int
    current_player: int
    round_index: int
    contributions: tuple[int, int]
    raises: int = 0
    non_raise_actions: int = 0
    folded_player: int | None = None
    showdown: bool = False
    history: tuple[Action, ...] = ()

    @property
    def is_terminal(self) -> bool:
        return self.folded_player is not None or self.showdown

    @property
    def public_rank(self) -> int | None:
        if self.round_index == 0 and not self.showdown:
            return None
        return card_rank(self.deal.public_card)

    def private_rank(self, player: int) -> int:
        _validate_player(player)
        return card_rank(self.deal.private_cards[player])

    def information_state(
        self,
        player: int | None = None,
        *,
        perfect_recall: bool = False,
    ) -> str:
        """Return a compact or public-history-aware information-state key."""
        observer = self.current_player if player is None else player
        _validate_player(observer)
        opponent = 1 - observer
        board = -1 if self.public_rank is None else self.public_rank
        compact = (
            f"h{self.private_rank(observer)}|b{board}|"
            f"m{self.contributions[observer]}|o{self.contributions[opponent]}"
        )
        if not perfect_recall:
            return compact
        history = ",".join(str(int(action)) for action in self.history)
        return f"{compact}|r{self.round_index}|t{history}"

    def legal_actions(self) -> tuple[Action, ...]:
        if self.is_terminal:
            return ()
        player = self.current_player
        opponent = 1 - player
        facing_bet = self.contributions[player] < self.contributions[opponent]
        actions: list[Action] = []
        if facing_bet:
            actions.append(Action.CALL)
        if self.raises < 2:
            actions.append(Action.RAISE)
        actions.append(Action.FOLD)
        if not facing_bet:
            actions.append(Action.CHECK)
        return tuple(actions)

    def apply(self, action: Action) -> LeducState:
        """Apply one legal action and return a new state."""
        if action not in self.legal_actions():
            legal = ", ".join(ACTION_NAMES[item] for item in self.legal_actions())
            raise ValueError(f"{ACTION_NAMES.get(action, str(action))} is illegal; choose {legal}")

        player = self.current_player
        opponent = 1 - player
        contributions = list(self.contributions)
        raises = self.raises
        non_raise_actions = self.non_raise_actions

        if action == Action.CALL:
            contributions[player] = contributions[opponent]
            non_raise_actions += 1
        elif action == Action.RAISE:
            raise_amount = 2 if self.round_index == 0 else 4
            contributions[player] = max(contributions) + raise_amount
            raises += 1
            non_raise_actions = 1
        elif action == Action.FOLD:
            return replace(
                self,
                current_player=opponent,
                folded_player=player,
                history=(*self.history, action),
            )
        else:
            non_raise_actions += 1

        advanced = replace(
            self,
            current_player=opponent,
            contributions=(contributions[0], contributions[1]),
            raises=raises,
            non_raise_actions=non_raise_actions,
            history=(*self.history, action),
        )
        if non_raise_actions < 2:
            return advanced
        if self.round_index == 0:
            return replace(
                advanced,
                round_index=1,
                raises=0,
                non_raise_actions=0,
            )
        return replace(advanced, showdown=True)

    def payoffs(self) -> tuple[float, float]:
        """Return zero-sum terminal utilities in big-blind units."""
        if not self.is_terminal:
            raise ValueError("payoffs are only defined at terminal states")
        pot = sum(self.contributions)
        winners: tuple[int, ...]
        if self.folded_player is not None:
            winners = (1 - self.folded_player,)
        else:
            scores = tuple(_showdown_score(self, player) for player in (0, 1))
            best = max(scores)
            winners = tuple(player for player, score in enumerate(scores) if score == best)

        shares = [0.0, 0.0]
        for winner in winners:
            shares[winner] = pot / len(winners)
        payoffs = tuple(
            (shares[player] - self.contributions[player]) / 2.0 for player in (0, 1)
        )
        return payoffs[0], payoffs[1]


def card_rank(card: int) -> int:
    """Map card ids (two suits per rank) to J=0, Q=1, K=2."""
    if card not in range(6):
        raise ValueError("card must be an integer from zero through five")
    return card // 2


def initial_state(deal: Deal, *, small_blind: int) -> LeducState:
    """Create a game with one-chip/two-chip blinds and the small blind acting."""
    _validate_player(small_blind)
    contributions = [2, 2]
    contributions[small_blind] = 1
    return LeducState(
        deal=deal,
        small_blind=small_blind,
        current_player=small_blind,
        round_index=0,
        contributions=(contributions[0], contributions[1]),
    )


def chance_outcomes() -> Iterator[tuple[LeducState, float]]:
    """Enumerate all 240 equiprobable ordered deals and blind assignments."""
    probability = 1.0 / 240.0
    for private_zero, private_one, public in permutations(range(6), 3):
        deal = Deal((private_zero, private_one), public)
        for small_blind in (0, 1):
            yield initial_state(deal, small_blind=small_blind), probability


def _showdown_score(state: LeducState, player: int) -> tuple[int, int]:
    private = state.private_rank(player)
    public = card_rank(state.deal.public_card)
    return int(private == public), private


def _validate_player(player: int) -> None:
    if player not in (0, 1):
        raise ValueError("player must be zero or one")
