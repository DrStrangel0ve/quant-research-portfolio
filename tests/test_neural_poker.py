from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

from quantlab.poker.deep_cfr import (
    CallingStationMicroPolicy,
    DeepCFRTrainer,
    NeuralPolicy,
    PokerNetwork,
    RandomMicroPolicy,
    ReplaySample,
    ReservoirBuffer,
    TrainingConfig,
    load_browser_strategy,
)
from quantlab.poker.features import FEATURE_DIM, canonical_cards, encode_information_state
from quantlab.poker.micro_evaluation import duplicate_micro_match
from quantlab.poker.micro_holdem import (
    MicroAction,
    MicroDeal,
    card_rank,
    card_suit,
    evaluate_five,
    initial_micro_state,
    sample_micro_state,
)
from quantlab.poker.resolver import BeliefRolloutResolver, infer_opponent_range


def test_micro_deal_rejects_duplicate_cards() -> None:
    with pytest.raises(ValueError, match="distinct"):
        MicroDeal(((0, 0), (2, 3)), (4, 5, 6))


def test_preflop_call_and_big_blind_check_reveal_flop() -> None:
    state = initial_micro_state(
        MicroDeal(((0, 5), (10, 15)), (1, 6, 11)),
        button=0,
    )
    assert state.contributions == (1, 2)
    assert state.current_player == 0
    assert MicroAction.FOLD in state.legal_actions()
    state = state.apply(MicroAction.CHECK_CALL)
    assert MicroAction.FOLD not in state.legal_actions()
    state = state.apply(MicroAction.CHECK_CALL)
    assert state.street == 1
    assert state.current_player == 1
    assert state.visible_board == (1, 6, 11)


def test_abstract_raise_sizes_are_distinct_and_capped() -> None:
    state = sample_micro_state(np.random.default_rng(2), button=0)
    children = {
        action: state.apply(action).contributions[0]
        for action in state.legal_actions()
        if action in (MicroAction.HALF_POT, MicroAction.POT, MicroAction.ALL_IN)
    }
    assert len(set(children.values())) == len(children)
    assert children[MicroAction.HALF_POT] < children[MicroAction.POT]
    assert children[MicroAction.POT] < children[MicroAction.ALL_IN]
    assert children[MicroAction.ALL_IN] == 20


def test_all_in_call_runs_board_and_preserves_zero_sum() -> None:
    state = sample_micro_state(np.random.default_rng(3), button=0)
    state = state.apply(MicroAction.ALL_IN)
    state = state.apply(MicroAction.CHECK_CALL)
    assert state.showdown
    assert len(state.visible_board) == 3
    assert sum(state.payoffs()) == pytest.approx(0.0)


def test_five_card_evaluator_orders_major_hand_classes() -> None:
    straight_flush = (4, 8, 12, 16, 20)
    quads = (0, 1, 2, 3, 20)
    full_house = (0, 1, 2, 4, 5)
    flush = (0, 4, 8, 12, 20)
    straight = (0, 5, 10, 15, 16)
    trips = (0, 1, 2, 8, 20)
    two_pair = (0, 1, 4, 5, 20)
    pair = (0, 1, 8, 12, 20)
    high_card = (0, 5, 10, 15, 20)
    scores = [
        evaluate_five(hand)
        for hand in (
            straight_flush,
            quads,
            full_house,
            flush,
            straight,
            trips,
            two_pair,
            pair,
            high_card,
        )
    ]
    assert scores == sorted(scores, reverse=True)


def test_suit_canonicalization_is_invariant_to_all_permutations() -> None:
    hole = (1, 14)
    board = (5, 10, 19)
    expected = canonical_cards(hole, board)
    for suit_map in permutations(range(4)):
        transformed = tuple(
            card_rank(card) * 4 + suit_map[card_suit(card)]
            for card in (*hole, *board)
        )
        assert canonical_cards(
            (transformed[0], transformed[1]),
            transformed[2:],
        ) == expected


def test_feature_vector_contains_only_observable_cards() -> None:
    first = initial_micro_state(
        MicroDeal(((0, 5), (10, 15)), (1, 6, 11)),
        button=0,
    )
    second = initial_micro_state(
        MicroDeal(((0, 5), (12, 17)), (2, 7, 13)),
        button=0,
    )
    assert encode_information_state(first).shape == (FEATURE_DIM,)
    assert np.array_equal(encode_information_state(first), encode_information_state(second))


def test_reservoir_sampling_is_bounded() -> None:
    rng = np.random.default_rng(4)
    buffer = ReservoirBuffer(5, rng)
    for index in range(100):
        buffer.add(
            ReplaySample(
                np.full(FEATURE_DIM, index, dtype=np.float32),
                np.zeros(5, dtype=np.float32),
                1.0,
            )
        )
    assert len(buffer) == 5
    assert buffer.seen == 100


def test_neural_policy_masks_illegal_actions() -> None:
    state = sample_micro_state(np.random.default_rng(5), button=0)
    policy = NeuralPolicy(PokerNetwork(hidden_size=16))
    probabilities = policy.probabilities(state)
    assert probabilities.sum() == pytest.approx(1.0)
    for action in MicroAction:
        if action not in state.legal_actions():
            assert probabilities[int(action)] == 0.0


def test_tiny_deep_cfr_run_collects_both_memories(tmp_path: Path) -> None:
    trainer = DeepCFRTrainer(
        TrainingConfig(
            iterations=3,
            traversals_per_player=1,
            train_every=3,
            advantage_steps=1,
            strategy_steps=1,
            batch_size=8,
            memory_capacity=500,
            hidden_size=16,
            seed=6,
        )
    )
    policy = trainer.train()
    assert len(trainer.advantage_memories[0]) > 0
    assert len(trainer.advantage_memories[1]) > 0
    assert len(trainer.strategy_memory) > 0
    state = sample_micro_state(np.random.default_rng(7))
    assert policy.probabilities(state).sum() == pytest.approx(1.0)
    checkpoint = tmp_path / "tiny.pt"
    export = tmp_path / "tiny.json"
    trainer.save_checkpoint(checkpoint)
    trainer.export_strategy_json(export)
    restored = NeuralPolicy.from_checkpoint(checkpoint)
    assert restored.probabilities(state) == pytest.approx(policy.probabilities(state))
    assert load_browser_strategy(export)["network"]["input_size"] == FEATURE_DIM


def test_duplicate_random_match_is_centered_near_zero() -> None:
    result = duplicate_micro_match(
        RandomMicroPolicy(),
        RandomMicroPolicy(),
        duplicate_pairs=300,
        rng=np.random.default_rng(8),
    )
    assert result.ci95_low <= result.mean_big_blinds <= result.ci95_high
    assert abs(result.mean_big_blinds) < 0.5


def test_range_inference_excludes_known_cards_and_normalizes() -> None:
    state = sample_micro_state(np.random.default_rng(9), button=0)
    blueprint = CallingStationMicroPolicy()
    posterior = infer_opponent_range(state, observer=0, blueprint=blueprint)
    known = set(state.deal.hole_cards[0])
    assert not any(known.intersection(combo) for combo in posterior.combos)
    assert posterior.probabilities.sum() == pytest.approx(1.0)
    assert posterior.effective_sample_size == pytest.approx(len(posterior.combos))


def test_range_inference_replays_public_history_from_the_root() -> None:
    state = sample_micro_state(np.random.default_rng(90), button=0)
    state = state.apply(MicroAction.HALF_POT)
    posterior = infer_opponent_range(
        state,
        observer=state.current_player,
        blueprint=RandomMicroPolicy(),
    )
    assert posterior.probabilities.sum() == pytest.approx(1.0)


def test_belief_rollout_resolver_returns_a_legal_action() -> None:
    state = sample_micro_state(np.random.default_rng(10), button=0)
    resolver = BeliefRolloutResolver(
        CallingStationMicroPolicy(),
        rollouts_per_action=2,
        seed=11,
    )
    probabilities = resolver.probabilities(state)
    selected = MicroAction(int(np.argmax(probabilities)))
    assert selected in state.legal_actions()
