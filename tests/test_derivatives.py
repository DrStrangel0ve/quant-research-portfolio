import numpy as np
import pytest

from quantlab.derivatives.black_scholes import (
    black_scholes_delta,
    black_scholes_price,
    european_option_monte_carlo,
)
from quantlab.derivatives.hedging import simulate_delta_hedge
from quantlab.derivatives.heston import HestonParameters, simulate_heston
from quantlab.simulation.monte_carlo import geometric_brownian_motion


def test_black_scholes_known_at_the_money_value() -> None:
    price = black_scholes_price(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
    )
    assert price == pytest.approx(10.4506, abs=1e-4)


def test_put_call_parity() -> None:
    common = {
        "spot": 100.0,
        "strike": 105.0,
        "maturity": 0.75,
        "rate": 0.03,
        "volatility": 0.25,
        "dividend_yield": 0.01,
    }
    call = black_scholes_price(**common, option_type="call")
    put = black_scholes_price(**common, option_type="put")
    parity = 100.0 * np.exp(-0.01 * 0.75) - 105.0 * np.exp(-0.03 * 0.75)
    assert call - put == pytest.approx(parity)


def test_call_and_put_delta_differ_by_discount_factor() -> None:
    common = {
        "spot": 100.0,
        "strike": 100.0,
        "maturity": 1.0,
        "rate": 0.03,
        "volatility": 0.20,
        "dividend_yield": 0.02,
    }
    call = black_scholes_delta(**common, option_type="call")
    put = black_scholes_delta(**common, option_type="put")
    assert call - put == pytest.approx(np.exp(-0.02))


def test_monte_carlo_price_contains_analytic_value() -> None:
    analytic = black_scholes_price(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.03,
        volatility=0.20,
    )
    estimate = european_option_monte_carlo(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.03,
        volatility=0.20,
        n_paths=100_000,
        rng=np.random.default_rng(42),
    )
    assert estimate.confidence_low <= analytic <= estimate.confidence_high


def test_gbm_is_seed_reproducible_and_starts_at_spot() -> None:
    arguments = {
        "spot": 100.0,
        "drift": 0.05,
        "volatility": 0.20,
        "maturity": 1.0,
        "n_steps": 12,
        "n_paths": 20,
    }
    first = geometric_brownian_motion(**arguments, rng=np.random.default_rng(7))
    second = geometric_brownian_motion(**arguments, rng=np.random.default_rng(7))
    assert np.array_equal(first, second)
    assert np.all(first[0] == 100.0)


def test_heston_variance_is_nonnegative() -> None:
    spots, variances = simulate_heston(
        spot=100.0,
        drift=0.02,
        maturity=1.0,
        n_steps=20,
        n_paths=50,
        parameters=HestonParameters(2.0, 0.04, 0.5, -0.7, 0.04),
        rng=np.random.default_rng(9),
    )
    assert spots.shape == variances.shape == (21, 50)
    assert (spots > 0.0).all()
    assert (variances >= 0.0).all()


def test_delta_hedge_returns_finite_distribution() -> None:
    result = simulate_delta_hedge(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.02,
        implied_volatility=0.20,
        realized_volatility=0.20,
        n_rebalances=20,
        n_paths=1_000,
        transaction_cost_bps=0.0,
        rng=np.random.default_rng(10),
    )
    assert len(result.errors) == 1_000
    assert np.isfinite(result.errors).all()
    assert abs(result.mean_error) < 0.25
