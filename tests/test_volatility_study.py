from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.research.volatility_study import (
    LAMBDAS,
    TEST_START,
    VARIANCE_FLOOR,
    causal_forecasts,
    daily_variance,
    paired_block_bootstrap,
    qlike,
    run_study,
    select_decay,
)


def daily_fixture() -> pd.Series[float]:
    index = pd.date_range("2023-12-01", "2026-06-30", freq="D", tz="UTC")
    values = 0.0001 * (1.2 + np.sin(np.arange(len(index)) / 23))
    return pd.Series(values, index=index)


def hourly_fixture() -> pd.DataFrame:
    index = pd.date_range("2023-12-01", "2026-06-30 23:00", freq="h", tz="UTC")
    rng = np.random.default_rng(72)
    return pd.DataFrame(
        {"close": 100 * np.exp(np.cumsum(rng.normal(0, 0.002, len(index))))}, index=index
    )


def test_daily_boundary_counts_previous_day_close() -> None:
    index = pd.date_range("2023-12-01 23:00", periods=25, freq="h", tz="UTC")
    bars = pd.DataFrame({"close": np.exp(np.arange(25) * 0.01)}, index=index)
    variance = daily_variance(bars)
    assert pd.isna(variance.iloc[0])
    assert variance.iloc[1] == pytest.approx(24 * 0.01**2)


def test_daily_gap_does_not_turn_multi_hour_return_into_one_hour() -> None:
    index = pd.date_range("2023-12-01", periods=96, freq="h", tz="UTC")
    bars = pd.DataFrame({"close": np.exp(np.arange(96) * 0.001)}, index=index)
    bars = bars.drop(index[47])  # Missing preceding day's final close also spoils day3.
    variance = daily_variance(bars)
    assert variance.isna().tolist() == [True, True, True, False]
    assert variance.iloc[-1] == pytest.approx(24e-6)


@pytest.mark.parametrize("error", ["duplicate", "unsorted", "naive", "offhour", "zero", "nan"])
def test_invalid_hourly_inputs_rejected(error: str) -> None:
    bars = hourly_fixture().iloc[:48].copy()
    if error == "duplicate":
        bars = pd.concat([bars, bars.iloc[-1:]])
    elif error == "unsorted":
        bars = bars.iloc[::-1]
    elif error == "naive":
        bars.index = bars.index.tz_localize(None)
    elif error == "offhour":
        bars.index += pd.Timedelta(minutes=1)
    else:
        bars.iloc[10, 0] = 0 if error == "zero" else np.nan
    with pytest.raises(ValueError):
        daily_variance(bars)


def test_forecasts_are_invariant_to_future_suffix() -> None:
    daily = daily_fixture()
    cut = pd.Timestamp("2026-03-01", tz="UTC")
    original = causal_forecasts(daily)
    changed = daily.copy()
    changed.loc[cut:] *= 1000
    future = causal_forecasts(changed)
    pd.testing.assert_frame_equal(
        original.drop(columns="realized_variance").loc[:cut],
        future.drop(columns="realized_variance").loc[:cut],
    )
    assert (
        original.loc[cut + pd.Timedelta(days=1), "previous_day"]
        != future.loc[cut + pd.Timedelta(days=1), "previous_day"]
    )


def test_rolling_baseline_uses_previous_30_calendar_days() -> None:
    daily = daily_fixture()
    date = pd.Timestamp("2026-02-01", tz="UTC")
    result = causal_forecasts(daily)
    expected = daily.loc[date - pd.Timedelta(days=30) : date - pd.Timedelta(days=1)].mean()
    assert result.loc[date, "rolling_30"] == pytest.approx(expected)
    altered = daily.drop(date - pd.Timedelta(days=20))
    assert pd.isna(causal_forecasts(altered).loc[date, "rolling_30"])


def test_ewma_uses_prior_observation_and_warmup_only() -> None:
    daily = daily_fixture()
    date = pd.Timestamp("2024-01-01", tz="UTC")
    result = causal_forecasts(daily)
    initial = daily.loc["2023-12"].mean()
    for decay in LAMBDAS:
        name = f"ewma_{decay:.2f}"
        assert result.loc[date, name] == pytest.approx(initial)
        assert result.loc[date + pd.Timedelta(days=1), name] == pytest.approx(
            decay * initial + (1 - decay) * daily.loc[date]
        )
    assert result.loc["2024", "training_mean"].isna().all()
    assert result.loc["2025-01-01", "training_mean"] == pytest.approx(daily.loc["2024"].mean())


def test_validation_selection_ignores_all_test_values() -> None:
    frame = causal_forecasts(daily_fixture())
    first, scores = select_decay(frame)
    changed = frame.copy()
    changed.loc[changed.index >= TEST_START, :] = np.nan
    second, second_scores = select_decay(changed)
    assert first == second
    pd.testing.assert_frame_equal(scores, second_scores)
    # Deliberately set one candidate to perfect validation forecasts, proving
    # selection is responsive to validation evidence, not a hardcoded lambda.
    valid = frame.index.year == 2025
    frame.loc[valid, "ewma_0.99"] = frame.loc[valid, "realized_variance"]
    assert select_decay(frame)[0] == 0.99


def test_zero_variance_forecasts_stay_positive_and_finite() -> None:
    daily = daily_fixture() * 0
    forecast = causal_forecasts(daily).loc["2025":].drop(columns="realized_variance")
    assert np.isfinite(forecast.to_numpy()).all()
    assert (forecast.to_numpy() >= VARIANCE_FLOOR).all()
    assert np.isfinite(qlike(np.zeros(2), np.full(2, VARIANCE_FLOOR))).all()


def test_qlike_matches_formula_and_rejects_invalid_inputs() -> None:
    result = qlike(np.array([1.0, 2.0]), np.array([1.0, 1.0]))
    np.testing.assert_allclose(result, [0, 1 - np.log(2)])
    for actual, predicted in [
        ([1.0], [0.0]),
        ([-1.0], [1.0]),
        ([np.nan], [1.0]),
        ([1.0], [1.0, 2.0]),
    ]:
        with pytest.raises(ValueError):
            qlike(np.array(actual), np.array(predicted))


def test_block_bootstrap_deterministic_and_degenerate_interval() -> None:
    left = np.sin(np.arange(181) / 8)
    right = np.cos(np.arange(181) / 7)
    first = paired_block_bootstrap(left, right)
    assert first == paired_block_bootstrap(left, right)
    assert first[0] == pytest.approx((left - right).mean())
    assert first[1] <= first[2]
    assert paired_block_bootstrap(np.full(21, 3.0), np.ones(21)) == (2.0, 2.0, 2.0)


@pytest.mark.parametrize("case", ["length", "short", "nan", "zero_block", "one_resample"])
def test_block_bootstrap_rejects_unpaired_or_invalid_samples(case: str) -> None:
    left, right = np.ones(21), np.ones(21)
    kwargs: dict[str, int] = {}
    if case == "length":
        right = right[:-1]
    elif case == "short":
        left, right = left[:2], right[:2]
    elif case == "nan":
        right[0] = np.nan
    elif case == "zero_block":
        kwargs["block_length"] = 0
    else:
        kwargs["resamples"] = 1
    with pytest.raises(ValueError):
        paired_block_bootstrap(left, right, **kwargs)


def test_end_to_end_fixed_split_counts_and_no_retroactive_selected_forecasts() -> None:
    result = run_study({"example": hourly_fixture()}, resamples=50)
    assert result.daily_counts["example"]["training_days"] == 366
    assert result.daily_counts["example"]["validation_days"] == 365
    assert result.daily_counts["example"]["test_scored_days"] == 181
    assert result.daily_counts["example"]["missing_days"] == 1
    assert len(result.metrics) == 4 and len(result.comparisons) == 3
    assert (result.comparisons["ci_status"] == "computed").all()
    pretest = result.predictions["date"] < TEST_START
    assert result.predictions.loc[pretest, "selected_ewma"].isna().all()


def test_incomplete_test_day_uses_common_sample_and_withholds_invalid_calendar_ci() -> None:
    bars = hourly_fixture().drop(pd.Timestamp("2026-03-05 12:00", tz="UTC"))
    result = run_study({"example": bars}, resamples=20)
    assert result.daily_counts["example"]["test_scored_days"] == 150
    assert result.metrics["test_days"].nunique() == 1
    assert (result.comparisons["ci_status"] == "calendar_gaps").all()
    assert result.comparisons["ci_95_low"].isna().all()
