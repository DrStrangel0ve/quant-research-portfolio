import numpy as np

from quantlab.data.synthetic import simulate_garch_returns
from quantlab.time_series.garch import fit_garch_11, forecast_variance


def test_garch_fit_is_stationary_and_forecast_is_finite() -> None:
    returns = simulate_garch_returns(n_periods=1_500, seed=30)
    fit = fit_garch_11(returns)
    forecast = forecast_variance(fit, horizon=20)
    assert fit.omega > 0.0
    assert fit.alpha >= 0.0
    assert fit.beta >= 0.0
    assert fit.persistence < 1.0
    assert np.isfinite(forecast).all()
    assert (forecast > 0.0).all()


def test_variance_forecast_mean_reverts() -> None:
    returns = simulate_garch_returns(n_periods=1_500, seed=31)
    fit = fit_garch_11(returns)
    forecast = forecast_variance(fit, horizon=100)
    initial_distance = abs(float(forecast.iloc[0]) - fit.unconditional_variance)
    final_distance = abs(float(forecast.iloc[-1]) - fit.unconditional_variance)
    assert final_distance <= initial_distance + 1e-15
