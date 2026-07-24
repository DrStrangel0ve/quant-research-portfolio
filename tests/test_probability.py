import numpy as np
import pytest

from quantlab.probability.games import (
    gambler_ruin_probability,
    kelly_fraction,
    monty_hall,
    secretary_game,
    simulate_gambler_ruin,
    simulate_kelly_wealth,
)


def test_fair_gambler_ruin_has_linear_probability() -> None:
    assert gambler_ruin_probability(
        initial_wealth=4,
        target_wealth=10,
        win_probability=0.5,
    ) == pytest.approx(0.4)


def test_simulated_ruin_matches_exact_probability() -> None:
    exact = gambler_ruin_probability(
        initial_wealth=3,
        target_wealth=8,
        win_probability=0.48,
    )
    simulated = simulate_gambler_ruin(
        initial_wealth=3,
        target_wealth=8,
        win_probability=0.48,
        n_trials=50_000,
        rng=np.random.default_rng(12),
    )
    assert abs(simulated.estimate - exact) < 4 * simulated.standard_error


def test_monty_hall_switching_wins_about_two_thirds() -> None:
    result = monty_hall(switch=True, n_trials=50_000, rng=np.random.default_rng(13))
    assert result.estimate == pytest.approx(2.0 / 3.0, abs=0.01)


def test_even_money_kelly_fraction() -> None:
    assert kelly_fraction(
        win_probability=0.55,
        net_win_multiple=1.0,
    ) == pytest.approx(0.10)


def test_full_kelly_beats_overbetting_in_expected_log_wealth() -> None:
    rng = np.random.default_rng(14)
    full = simulate_kelly_wealth(
        initial_wealth=1.0,
        fraction=0.10,
        win_probability=0.55,
        net_win_multiple=1.0,
        net_loss_multiple=1.0,
        n_bets=500,
        n_paths=20_000,
        rng=rng,
    )
    over = simulate_kelly_wealth(
        initial_wealth=1.0,
        fraction=0.20,
        win_probability=0.55,
        net_win_multiple=1.0,
        net_loss_multiple=1.0,
        n_bets=500,
        n_paths=20_000,
        rng=np.random.default_rng(15),
    )
    assert np.log(full).mean() > np.log(over).mean()


def test_secretary_one_over_e_rule() -> None:
    result = secretary_game(
        n_candidates=100,
        sample_fraction=1.0 / np.e,
        n_trials=20_000,
        rng=np.random.default_rng(16),
    )
    assert result.estimate == pytest.approx(1.0 / np.e, abs=0.02)
