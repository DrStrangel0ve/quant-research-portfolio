# Verified seeded results

These numbers were regenerated from a clean local invocation of all 12 project
scripts. Except for analytic identities, they come from synthetic data or
simulation and must not be interpreted as claims of live trading performance.

## Derivatives and simulation

| Experiment | Result | Interpretation |
|---|---:|---|
| European call | Analytic 7.1281; MC 7.1488 | 100,000-path error 0.0208 vs. standard error 0.0398 |
| Heston | ATM IV 18.71% | Negative spot/variance correlation produces a downward strike skew |
| Heston discretization | Feller ratio 0.64 | Positivity condition fails, so full truncation is material and documented |
| Delta hedge | Daily error std 0.598 | More frequent hedging reduced dispersion while mean costs increased |
| Market maker | Mean P&L 1.106, std 1.710 | Baseline mechanism earns modeled spread but retains meaningful inventory risk |
| Limit order book | Median spread 0.020 | 5,000 seeded events preserve a non-crossed final book |

The delta experiment intentionally sets realized volatility above the option's
implied volatility. Its negative average hedging result is therefore expected
for a short-volatility position and is not hidden.

## Backtests and portfolio construction

| Experiment | Net annual return | Net Sharpe | Max drawdown | Total modeled cost |
|---|---:|---:|---:|---:|
| Rolling pairs | 6.09% | 1.37 | -3.38% | 2.39% |
| Cross-sectional momentum | 2.37% | 0.48 | -14.57% | 12.79% |
| Walk-forward trend | -1.59% | -0.40 | -20.29% | 6.97% |
| Rolling risk parity | 9.76% | 0.83 | -19.47% | 0.17% |

These are seeded synthetic experiments. The negative stitched out-of-sample
trend result is retained because a credible research framework must make
failure visible. Risk parity also underperformed equal weight on Sharpe in this
particular sample (0.83 versus 0.96), another useful counterexample to automatic
strategy promotion.

## Statistics and probability

| Experiment | Result | Benchmark |
|---|---:|---:|
| GARCH persistence | 0.9888 estimated | 0.9800 true |
| Gambler's ruin success | 0.19095 simulated | 0.19173 exact |
| Monty Hall switch | 0.66828 | 2/3 |
| Kelly fraction | 0.10 | Binary even-money analytic optimum |
| Secretary sample fraction | 0.375 best grid point | 1/e = 0.36788 |

## Verification command

```bash
python -m ruff check .
python -m mypy
python -m pytest
python scripts/run_all.py
```

At the recorded verification point: 35 tests passed, strict type checking and
linting passed, and statement coverage was 85%.
