from pathlib import Path

import numpy as np
import pytest

from quantlab.poker.agents import RandomBot, TabularPolicy
from quantlab.poker.cfr import CFRPlusTrainer
from quantlab.poker.evaluation import duplicate_match, expected_value, exploitability
from quantlab.poker.leduc import Action, Deal, chance_outcomes, initial_state
from quantlab.poker.rlcard_adapter import load_rlcard_reference_policy


def test_deal_rejects_duplicate_cards() -> None:
    with pytest.raises(ValueError, match="distinct"):
        Deal((0, 0), 1)


def test_blinds_and_first_action_match_rlcard_rules() -> None:
    state = initial_state(Deal((0, 2), 4), small_blind=1)
    assert state.contributions == (2, 1)
    assert state.current_player == 1
    assert state.legal_actions() == (Action.CALL, Action.RAISE, Action.FOLD)


def test_call_check_reveals_board_and_preserves_first_actor() -> None:
    state = initial_state(Deal((0, 2), 4), small_blind=0)
    state = state.apply(Action.CALL).apply(Action.CHECK)
    assert state.round_index == 1
    assert state.current_player == 0
    assert state.public_rank == 2
    assert state.legal_actions() == (Action.RAISE, Action.FOLD, Action.CHECK)


def test_raise_cap_and_call_end_the_round() -> None:
    state = initial_state(Deal((0, 2), 4), small_blind=0)
    state = state.apply(Action.RAISE).apply(Action.RAISE)
    assert Action.RAISE not in state.legal_actions()
    state = state.apply(Action.CALL)
    assert state.round_index == 1
    assert state.contributions == (6, 6)


def test_fold_payoffs_are_zero_sum_and_scaled_by_big_blind() -> None:
    state = initial_state(Deal((0, 2), 4), small_blind=0).apply(Action.FOLD)
    assert state.payoffs() == (-0.5, 0.5)
    assert sum(state.payoffs()) == pytest.approx(0.0)


def test_public_pair_beats_higher_unpaired_private_card() -> None:
    state = initial_state(Deal((0, 4), 1), small_blind=0)
    state = state.apply(Action.CALL).apply(Action.CHECK)
    state = state.apply(Action.CHECK).apply(Action.CHECK)
    assert state.showdown
    assert state.payoffs()[0] > 0.0
    assert sum(state.payoffs()) == pytest.approx(0.0)


def test_chance_enumeration_is_normalized_and_complete() -> None:
    outcomes = list(chance_outcomes())
    assert len(outcomes) == 240
    assert sum(probability for _, probability in outcomes) == pytest.approx(1.0)


def test_tabular_policy_masks_illegal_actions() -> None:
    state = initial_state(Deal((0, 2), 4), small_blind=0)
    policy = TabularPolicy({state.information_state(): [0.0, 0.0, 0.0, 1.0]})
    probabilities = policy.probabilities(state)
    assert probabilities[int(Action.CHECK)] == 0.0
    assert probabilities.sum() == pytest.approx(1.0)


def test_seeded_training_is_reproducible() -> None:
    first = CFRPlusTrainer(seed=77)
    second = CFRPlusTrainer(seed=77)
    assert first.train(50).table == second.train(50).table


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    trainer = CFRPlusTrainer(seed=88)
    trainer.train(40)
    checkpoint = tmp_path / "policy.json"
    trainer.save(checkpoint)
    restored = CFRPlusTrainer.load(checkpoint)
    assert restored.iteration == 40
    assert restored.information_mode == "perfect_recall"
    assert restored.average_policy().table == trainer.average_policy().table


def test_training_improves_exact_exploitability_and_beats_random() -> None:
    trainer = CFRPlusTrainer(seed=99)
    early = trainer.train(50)
    early_exploitability = exploitability(early)
    mature = trainer.train(450)
    assert exploitability(mature) < early_exploitability
    assert expected_value(mature, RandomBot()) > 0.3


def test_duplicate_match_returns_paired_confidence_interval() -> None:
    result = duplicate_match(
        RandomBot(),
        RandomBot(),
        duplicate_pairs=200,
        rng=np.random.default_rng(123),
    )
    assert result.ci95_low <= result.mean_big_blinds <= result.ci95_high
    assert abs(result.mean_big_blinds) < 0.25


def test_rlcard_reference_checkpoint_decodes_all_information_sets() -> None:
    pytest.importorskip("rlcard")
    reference = load_rlcard_reference_policy()
    assert len(reference.table) == 84


def test_engine_transitions_and_payoffs_match_rlcard() -> None:
    rlcard = pytest.importorskip("rlcard")
    card_ids = {"SJ": 0, "HJ": 1, "SQ": 2, "HQ": 3, "SK": 4, "HK": 5}
    action_ids = {"call": 0, "raise": 1, "fold": 2, "check": 3}
    rng = np.random.default_rng(456)

    for seed in range(10):
        environment = rlcard.make(
            "leduc-holdem",
            config={"seed": seed, "allow_step_back": True},
        )
        environment.reset()
        game = environment.game
        private_cards = tuple(card_ids[player.hand.get_index()] for player in game.players)
        next_public = card_ids[game.dealer.deck[-1].get_index()]
        small_blind = 0 if game.players[0].in_chips == 1 else 1
        state = initial_state(
            Deal((private_cards[0], private_cards[1]), next_public),
            small_blind=small_blind,
        )

        while not state.is_terminal:
            assert state.current_player == game.game_pointer
            assert state.contributions == tuple(player.in_chips for player in game.players)
            rlcard_actions = game.get_legal_actions()
            assert {action.name.lower() for action in state.legal_actions()} == set(
                rlcard_actions
            )
            action_name = str(rng.choice(rlcard_actions))
            state = state.apply(Action(action_ids[action_name]))
            environment.step(action_ids[action_name])

        assert state.payoffs() == pytest.approx(tuple(environment.get_payoffs()))
