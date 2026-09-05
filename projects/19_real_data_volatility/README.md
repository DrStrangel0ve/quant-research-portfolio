# Real-data volatility forecasting

**Question:** does validation-selected exponential smoothing forecast next-day
realized variance better than a prior 30-day mean on BTCUSDT and ETHUSDT?
This is a forecasting study, with no trading strategy or P&L claim.

The study uses hourly Binance spot closes from December 2023 through June 2026.
[DATA.md](DATA.md) documents the official source, checksummed manifest, timestamp
units, and data validation. [RESULTS.md](RESULTS.md) records the actual run,
including unfavorable comparisons.

## Run

From the repository root, after `pip install -e '.[dev]'`:

```bash
# First run: download only the pinned public archives that are absent locally.
python projects/19_real_data_volatility/run.py --download

# Subsequent run: hashes are verified; no network access is needed.
python projects/19_real_data_volatility/run.py
```

Raw ZIP files remain in ignored `data_cache/`. The generated `results/` directory
contains every daily prediction, validation candidate score, held-out score,
paired comparison, data/code fingerprints, and a chart. The recorded results
are committed separately from downloaded raw data and generated artifacts.

## Frozen research protocol

The candidate set and pipeline were fixed before inspecting held-out model
scores in this run. This is a **retrospective held-out calendar period**;
the study was not preregistered and is not a prospective live evaluation.

| Period | Use |
|---|---|
| December 2023 | Initialize lagged features and EWMA state |
| January–December 2024 | EWMA state warmup; fit fixed training-mean baseline |
| January–December 2025 | Select one EWMA decay per asset using mean QLIKE |
| January–June 2026 | Fixed held-out rolling one-day forecast evaluation |

Hourly return `r[t] = log(close[t]) - log(close[t-1])` is calculated **before**
daily grouping. Day D's target is the sum of its 24 squared hourly returns,
including the first hour's return from the preceding day's final hourly close.
This is an hourly-sampled realized-variance proxy. It does not recover the full
intrahour price path or directly observe latent conditional variance.

The first source day lacks the preceding hourly close and is excluded. Any
other incomplete day remains missing; missing prices/returns are never filled
with zero. A gap also invalidates the first return after it. Calendar-based
30-day means require all 30 prior daily observations. EWMA state is unchanged
on missing days. All models use common eligible scoring dates; bootstrap
intervals are withheld if those dates contain calendar gaps.

Models are:

- **Rolling 30-day mean:** mean of D−30 through D−1, excluding D.
- **Previous day:** D−1's observed variance, a persistence baseline.
- **Training mean:** fixed mean of observed 2024 daily variances, available only
  from January 2025 onward.
- **EWMA:** forecast `h[D] = lambda*h[D-1] + (1-lambda)*RV[D-1]`.
  On January 1, 2024, state starts from the observed December 2023 mean.
  The fixed candidates are **0.90, 0.94, 0.97, 0.99**; the first minimum breaks
  ties. Selection uses 2025 QLIKE only, separately for each asset, and is then
  locked. During evaluation the state may incorporate a completed prior test
  day, never the day being predicted. There is no test-period refitting of lambda.

QLIKE is `RV/h - log(RV/h) - 1`, and MSE is `(RV-h)^2`, both on the **variance**
scale. Lower is better. A `1e-12` floor keeps forecasts positive and protects
the QLIKE target logarithm; MSE uses the raw observed target. The recorded run
reports how many observed days meet that floor. QLIKE and MSE are motivated by
the volatility-proxy comparison literature, but its assumptions should not be
presumed valid for every cryptocurrency sample ([Patton, 2011](https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf)).

Paired loss differences are **selected EWMA minus baseline** on the same dates.
The 95% percentile interval uses **2,000 non-circular moving-block bootstrap
resamples**, seven consecutive calendar days per block, seed `20260905`.
Blocks are sampled with replacement, concatenated, and truncated to the
original test length. The interval reflects dependence within that chosen
block length; it does not guarantee coverage under regime change, adjust for
multiple asset/baseline comparisons, or include uncertainty in model selection.

## Evidence and limits

Tests verify day-boundary alignment, missing-hour behavior, causal suffix
invariance, lagged rolling features, validation-only selection, positive finite
forecasts, deterministic paired bootstrap intervals, and equal scoring dates.

BTC and ETH were chosen examples, not a survivorship-free asset universe.
The two assets share one venue and can be highly dependent. Hourly candles
provide neither order-book microstructure nor executable bid/ask prices; the
study therefore does not simulate fills, fees, slippage, or strategy returns.
Six held-out months cover limited regimes. A favorable forecast comparison
would not establish tradable alpha or generalize automatically to equities,
futures, another venue, or a future period.

## Interview discussion

- Explain the difference between a realized-variance proxy and latent conditional variance.
- Reconstruct the first hourly return of a UTC day and why computing returns inside each day is wrong.
- Show the exact information available for a January 2026 forecast.
- Explain why random train/test splitting would contaminate this time-series study.
- Defend the baseline set, QLIKE scale, and paired calendar-block uncertainty estimate.
- Explain an unfavorable model comparison and what a genuinely new holdout would test next.
