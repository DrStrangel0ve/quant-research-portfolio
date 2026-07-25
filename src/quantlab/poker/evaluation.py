"""Exact and Monte Carlo evaluation for heads-up Leduc policies."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from quantlab.poker.agents import Policy, sample_action
from quantlab.poker.leduc import Action, Deal, LeducState, chance_outcomes, initial_state


@dataclass(frozen=True)
class BestResponseResult:
    """An exact deterministic best response and its game value."""

    player: int
    value: float
    actions: Mapping[str, Action]


@dataclass(frozen=True)
class MatchResult:
    """Paired duplicate-poker estimate for the first named policy."""

    mean_big_blinds: float
    standard_error: float
    ci95_low: float
    ci95_high: float
    duplicate_pairs: int


def expected_value(policy_zero: Policy, policy_one: Policy) -> float:
    """Exactly enumerate chance and return player zero's expected utility."""
    value = 0.0
    for state, probability in chance_outcomes():
        value += probability * _profile_value(state, (policy_zero, policy_one), player=0)
    return float(value)


def best_response(opponent: Policy, *, player: int) -> BestResponseResult:
    """Compute an exact pure best response by backward induction over information sets.

    Counterfactual reach weights include chance and the opponent's actions while
    excluding the responding player's own reach, as required by imperfect-
    information best-response evaluation.
    """
    if player not in (0, 1):
        raise ValueError("player must be zero or one")
    grouped: dict[str, list[tuple[LeducState, float]]] = defaultdict(list)
    for root, chance_probability in chance_outcomes():
        _collect_response_histories(
            root,
            responding_player=player,
            opponent=opponent,
            counterfactual_reach=chance_probability,
            grouped=grouped,
        )

    depths = {
        key: max(len(state.history) for state, _ in histories)
        for key, histories in grouped.items()
    }
    chosen: dict[str, Action] = {}
    for key in sorted(grouped, key=depths.__getitem__, reverse=True):
        histories = grouped[key]
        legal = histories[0][0].legal_actions()
        if any(state.legal_actions() != legal for state, _ in histories):
            raise RuntimeError(f"inconsistent legal actions in information state {key}")
        action_values = {
            action: sum(
                reach
                * _response_continuation(
                    state.apply(action),
                    responding_player=player,
                    opponent=opponent,
                    chosen=chosen,
                )
                for state, reach in histories
            )
            for action in legal
        }
        chosen[key] = max(action_values, key=action_values.__getitem__)

    value = 0.0
    for root, probability in chance_outcomes():
        value += probability * _response_continuation(
            root,
            responding_player=player,
            opponent=opponent,
            chosen=chosen,
        )
    return BestResponseResult(player=player, value=float(value), actions=chosen)


def exploitability(policy: Policy) -> float:
    """Return exact two-player exploitability in big blinds per hand.

    This is half of NashConv: the mean gain obtained by independently replacing
    either seat with its exact best response.
    """
    response_zero = best_response(policy, player=0).value
    response_one = best_response(policy, player=1).value
    return float((response_zero + response_one) / 2.0)


def duplicate_match(
    policy_a: Policy,
    policy_b: Policy,
    *,
    duplicate_pairs: int,
    rng: np.random.Generator,
) -> MatchResult:
    """Seat-swap identical deals and report a paired 95% confidence interval."""
    if duplicate_pairs <= 1:
        raise ValueError("duplicate_pairs must exceed one")
    pair_values = np.empty(duplicate_pairs, dtype=float)
    for index in range(duplicate_pairs):
        cards = rng.choice(6, size=3, replace=False)
        small_blind = int(rng.integers(0, 2))
        deal = Deal((int(cards[0]), int(cards[1])), int(cards[2]))
        first = _play_sampled(
            initial_state(deal, small_blind=small_blind),
            (policy_a, policy_b),
            rng,
        )[0]
        mirrored_deal = Deal((deal.private_cards[1], deal.private_cards[0]), deal.public_card)
        second = _play_sampled(
            initial_state(mirrored_deal, small_blind=1 - small_blind),
            (policy_b, policy_a),
            rng,
        )[1]
        pair_values[index] = (first + second) / 2.0

    mean = float(pair_values.mean())
    standard_error = float(pair_values.std(ddof=1) / np.sqrt(duplicate_pairs))
    half_width = 1.96 * standard_error
    return MatchResult(
        mean_big_blinds=mean,
        standard_error=standard_error,
        ci95_low=mean - half_width,
        ci95_high=mean + half_width,
        duplicate_pairs=duplicate_pairs,
    )


def _profile_value(
    state: LeducState,
    policies: tuple[Policy, Policy],
    *,
    player: int,
) -> float:
    if state.is_terminal:
        return state.payoffs()[player]
    probabilities = policies[state.current_player].probabilities(state)
    return float(
        sum(
            probabilities[int(action)]
            * _profile_value(state.apply(action), policies, player=player)
            for action in state.legal_actions()
        )
    )


def _collect_response_histories(
    state: LeducState,
    *,
    responding_player: int,
    opponent: Policy,
    counterfactual_reach: float,
    grouped: dict[str, list[tuple[LeducState, float]]],
) -> None:
    if state.is_terminal:
        return
    if state.current_player == responding_player:
        grouped[_response_information_state(state)].append((state, counterfactual_reach))
        for action in state.legal_actions():
            _collect_response_histories(
                state.apply(action),
                responding_player=responding_player,
                opponent=opponent,
                counterfactual_reach=counterfactual_reach,
                grouped=grouped,
            )
        return

    probabilities = opponent.probabilities(state)
    for action in state.legal_actions():
        probability = probabilities[int(action)]
        if probability > 0.0:
            _collect_response_histories(
                state.apply(action),
                responding_player=responding_player,
                opponent=opponent,
                counterfactual_reach=counterfactual_reach * probability,
                grouped=grouped,
            )


def _response_continuation(
    state: LeducState,
    *,
    responding_player: int,
    opponent: Policy,
    chosen: Mapping[str, Action],
) -> float:
    if state.is_terminal:
        return state.payoffs()[responding_player]
    if state.current_player == responding_player:
        key = _response_information_state(state)
        if key not in chosen:
            raise RuntimeError(f"best-response action for {key} has not been solved")
        return _response_continuation(
            state.apply(chosen[key]),
            responding_player=responding_player,
            opponent=opponent,
            chosen=chosen,
        )

    probabilities = opponent.probabilities(state)
    return float(
        sum(
            probabilities[int(action)]
            * _response_continuation(
                state.apply(action),
                responding_player=responding_player,
                opponent=opponent,
                chosen=chosen,
            )
            for action in state.legal_actions()
        )
    )


def _play_sampled(
    state: LeducState,
    policies: tuple[Policy, Policy],
    rng: np.random.Generator,
) -> tuple[float, float]:
    while not state.is_terminal:
        action = sample_action(policies[state.current_player], state, rng)
        state = state.apply(action)
    return state.payoffs()


def _response_information_state(state: LeducState) -> str:
    """Perfect-recall key for a best responder that observes the public history."""
    return state.information_state(perfect_recall=True)
