"""Causal daily variance forecasts with validation-only parameter selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

LAMBDAS = (0.90, 0.94, 0.97, 0.99)
VARIANCE_FLOOR = 1e-12
TRAIN_START = pd.Timestamp("2024-01-01", tz="UTC")
VALIDATION_START = pd.Timestamp("2025-01-01", tz="UTC")
TEST_START = pd.Timestamp("2026-01-01", tz="UTC")
TEST_END = pd.Timestamp("2026-06-30", tz="UTC")
BASELINES = ("rolling_30", "previous_day", "training_mean")


@dataclass
class StudyResult:
    predictions: pd.DataFrame
    validation: pd.DataFrame
    metrics: pd.DataFrame
    comparisons: pd.DataFrame
    selected_lambdas: dict[str, float]
    daily_counts: dict[str, dict[str, int]]


def daily_variance(hourly: pd.DataFrame) -> pd.Series[float]:
    """Sum 24 hourly squared log returns; incomplete days remain missing.

    A close is indexed by its bar's opening hour. Compute returns before daily
    grouping so hour 00 includes the preceding day's final hourly close. A gap
    masks the first return after it instead of treating a multi-hour return as
    one hourly observation. Inputs may contain a genuine exchange data gap.
    """
    if "close" not in hourly or not isinstance(hourly.index, pd.DatetimeIndex):
        raise ValueError("hourly close column and DatetimeIndex required")
    index = hourly.index
    if index.tz is None or index.empty or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("nonempty, sorted, unique, timezone-aware hourly index required")
    index = index.tz_convert("UTC")
    if not (index == index.floor("h")).all():
        raise ValueError("bar opening times must be aligned to whole UTC hours")
    prices = hourly["close"].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("closes must be positive and finite")
    log_close = pd.Series(np.log(prices), index=index)
    returns = log_close.diff()
    contiguous = index.to_series().diff().eq(pd.Timedelta(hours=1))
    returns = returns.where(contiguous)
    squared = returns.pow(2)
    count = squared.resample("D").count()
    result = squared.resample("D").sum(min_count=24).where(count.eq(24))
    result.name = "realized_variance"
    return result


def causal_forecasts(daily: pd.Series[float]) -> pd.DataFrame:
    """Forecast day t before observing its RV; never use a future observation.

    EWMA starts on 2024-01-01 from the observed December 2023 mean. The year
    2024 warms its state and fits the fixed-mean baseline. Missing RV days do
    not update EWMA state; calendar rolling windows require all 30 prior days.
    """
    if not isinstance(daily.index, pd.DatetimeIndex) or daily.index.tz is None:
        raise ValueError("daily index must be timezone-aware")
    index = daily.index.tz_convert("UTC")
    if index.empty or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("daily index must be nonempty, sorted and unique")
    if not (index == index.floor("D")).all():
        raise ValueError("daily index must align to UTC midnight")
    values = daily.to_numpy(dtype=float)
    if np.isinf(values).any() or np.any(values[np.isfinite(values)] < 0):
        raise ValueError("observed variance must be finite and nonnegative or missing")
    rv = pd.Series(values, index=index).reindex(pd.date_range(index[0], index[-1], freq="D"))
    warmup = rv.loc[(rv.index >= pd.Timestamp("2023-12-01", tz="UTC")) & (rv.index < TRAIN_START)]
    if warmup.notna().sum() == 0:
        raise ValueError("observed December 2023 initialization data required")
    result = pd.DataFrame({"realized_variance": rv})
    result["previous_day"] = rv.shift(1).clip(lower=VARIANCE_FLOOR)
    result["rolling_30"] = rv.shift(1).rolling(30, min_periods=30).mean().clip(lower=VARIANCE_FLOOR)
    train = rv.loc[(rv.index >= TRAIN_START) & (rv.index < VALIDATION_START)]
    result["training_mean"] = np.nan
    if train.notna().any():
        result.loc[result.index >= VALIDATION_START, "training_mean"] = max(
            float(train.mean()), VARIANCE_FLOOR
        )
    initial = max(float(warmup.mean()), VARIANCE_FLOOR)
    for decay in LAMBDAS:
        state = initial
        forecasts = np.full(len(rv), np.nan)
        for i, date in enumerate(pd.DatetimeIndex(rv.index)):
            if date < TRAIN_START:
                continue
            forecasts[i] = max(state, VARIANCE_FLOOR)
            observed = rv.iloc[i]
            if pd.notna(observed):
                state = decay * state + (1 - decay) * float(observed)
        result[f"ewma_{decay:.2f}"] = forecasts
    result.index.name = "date"
    return result


def qlike(actual: NDArray[np.float64], forecast: NDArray[np.float64]) -> NDArray[np.float64]:
    """Dimensionless QLIKE, y/h - log(y/h) - 1; lower is better."""
    y, h = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    if y.ndim != 1 or y.shape != h.shape or not np.isfinite(y).all() or not np.isfinite(h).all():
        raise ValueError("equal one-dimensional finite arrays required")
    if np.any(y < 0) or np.any(h <= 0):
        raise ValueError("nonnegative observations and positive forecasts required")
    ratio = np.maximum(y, VARIANCE_FLOOR) / np.maximum(h, VARIANCE_FLOOR)
    result: NDArray[np.float64] = ratio - np.log(ratio) - 1.0
    if not np.isfinite(result).all():
        raise ValueError("loss overflow")
    return result


def select_decay(predictions: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Select among the fixed candidates using 2025 observations only."""
    names = [f"ewma_{decay:.2f}" for decay in LAMBDAS]
    validation = predictions.loc[
        (predictions.index >= VALIDATION_START) & (predictions.index < TEST_START)
    ].dropna(subset=["realized_variance", *names, *BASELINES])
    if len(validation) < 30:
        raise ValueError("at least 30 common eligible validation dates required")
    observed = validation["realized_variance"].to_numpy(dtype=float)
    rows = [
        {
            "lambda": decay,
            "validation_qlike": float(
                qlike(observed, validation[name].to_numpy(dtype=float)).mean()
            ),
            "validation_days": len(validation),
        }
        for decay, name in zip(LAMBDAS, names, strict=True)
    ]
    scores = pd.DataFrame(rows)
    # Stable first-minimum tie break follows the declared candidate order.
    selected = float(scores.iloc[int(np.argmin(scores["validation_qlike"].to_numpy()))]["lambda"])
    return selected, scores


def paired_block_bootstrap(
    candidate_loss: NDArray[np.float64],
    baseline_loss: NDArray[np.float64],
    *,
    block_length: int = 7,
    resamples: int = 2000,
    seed: int = 20260905,
) -> tuple[float, float, float]:
    """Non-circular moving-block percentile interval for a paired mean loss.

    Paired dates must already match and be consecutive calendar days. The
    caller enforces calendar continuity; arrays alone cannot establish dates.
    """
    left, right = np.asarray(candidate_loss, dtype=float), np.asarray(baseline_loss, dtype=float)
    if left.ndim != 1 or left.shape != right.shape or not np.isfinite(left).all():
        raise ValueError("paired loss arrays must be finite one-dimensional arrays of equal size")
    if (
        not np.isfinite(right).all()
        or block_length < 1
        or len(left) < block_length
        or resamples < 2
    ):
        raise ValueError("finite losses, valid block length and >= 2 resamples required")
    differences = left - right
    if not np.isfinite(differences).all():
        raise ValueError("loss difference overflow")
    blocks = (len(left) + block_length - 1) // block_length
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(left) - block_length + 1, size=(resamples, blocks))
    indices = (starts[:, :, None] + np.arange(block_length)).reshape(resamples, -1)[:, : len(left)]
    means = differences[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(differences.mean()), float(low), float(high)


def run_study(hourly: dict[str, pd.DataFrame], *, resamples: int = 2000) -> StudyResult:
    """Run the fixed split; test outcomes never participate in selection."""
    if not hourly:
        raise ValueError("at least one asset required")
    all_predictions: list[pd.DataFrame] = []
    all_validation: list[pd.DataFrame] = []
    metrics: list[dict[str, str | float | int]] = []
    comparisons: list[dict[str, str | float | int]] = []
    selected: dict[str, float] = {}
    counts: dict[str, dict[str, int]] = {}
    for asset, bars in sorted(hourly.items()):
        daily = daily_variance(bars)
        if daily.index[0] > pd.Timestamp("2023-12-01", tz="UTC") or daily.index[-1] < TEST_END:
            raise ValueError("fixed study requires December 2023 through June 2026 coverage")
        frame = causal_forecasts(daily)
        decay, validation = select_decay(frame)
        selected[asset] = decay
        validation.insert(0, "asset", asset)
        all_validation.append(validation)
        frame["selected_ewma"] = frame[f"ewma_{decay:.2f}"].where(frame.index >= TEST_START)
        frame["split"] = np.select(
            [frame.index < TRAIN_START, frame.index < VALIDATION_START, frame.index < TEST_START],
            ["warmup", "train", "validation"],
            default="test",
        )
        frame["asset"] = asset
        all_predictions.append(frame.reset_index())
        candidates = ("selected_ewma", *BASELINES)
        test = frame.loc[(frame.index >= TEST_START) & (frame.index <= TEST_END)]
        common = test.dropna(subset=["realized_variance", *candidates])
        if len(common) < 7:
            raise ValueError("at least seven common eligible held-out dates required")
        # Do not silently pretend nonadjacent dates are adjacent bootstrap days.
        consecutive = common.index.to_series().diff().iloc[1:].eq(pd.Timedelta(days=1)).all()
        actual = common["realized_variance"].to_numpy(dtype=float)
        losses: dict[str, NDArray[np.float64]] = {}
        for model in candidates:
            predicted = common[model].to_numpy(dtype=float)
            losses[model] = qlike(actual, predicted)
            metrics.append(
                {
                    "asset": asset,
                    "model": model,
                    "test_days": len(common),
                    "mean_qlike": float(losses[model].mean()),
                    "mse": float(np.square(actual - predicted).mean()),
                }
            )
        for baseline in BASELINES:
            if consecutive:
                estimate, low, high = paired_block_bootstrap(
                    losses["selected_ewma"], losses[baseline], resamples=resamples
                )
            else:
                estimate = float((losses["selected_ewma"] - losses[baseline]).mean())
                low, high = float("nan"), float("nan")
            comparisons.append(
                {
                    "asset": asset,
                    "baseline": baseline,
                    "test_days": len(common),
                    "qlike_difference": estimate,
                    "ci_95_low": low,
                    "ci_95_high": high,
                    "ci_status": "computed" if consecutive else "calendar_gaps",
                }
            )
        counts[asset] = {
            "hourly_rows": len(bars),
            "daily_calendar_rows": len(daily),
            "complete_days": int(daily.notna().sum()),
            "missing_days": int(daily.isna().sum()),
            "training_days": int(
                daily.loc[TRAIN_START : VALIDATION_START - pd.Timedelta(days=1)].notna().sum()
            ),
            "validation_days": int(validation.iloc[0]["validation_days"]),
            "test_calendar_days": len(test),
            "test_scored_days": len(common),
            "variance_floor_days": int((daily < VARIANCE_FLOOR).sum()),
        }
    return StudyResult(
        pd.concat(all_predictions, ignore_index=True),
        pd.concat(all_validation, ignore_index=True),
        pd.DataFrame(metrics),
        pd.DataFrame(comparisons),
        selected,
        counts,
    )
