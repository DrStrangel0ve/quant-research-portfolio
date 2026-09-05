# Verified seeded results

These numbers were regenerated from clean, seeded project invocations. Except
for analytic identities and exactly enumerated game trees, they come from
synthetic data or simulation and must not be interpreted as claims of live
trading performance.

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

## Imperfect-information game solving

| Experiment | Result | Benchmark |
|---|---:|---|
| CFR+ exploitability | 0.04236 BB/hand | Exact full-history best response |
| CFR+ vs RLCard CFR | +0.13596 BB/hand | Exact enumeration, either seat |
| Duplicate face-off | +0.10673 BB/hand | 10,000 pairs; 95% CI [0.06749, 0.14596] |
| CFR+ vs uniform random | +0.73821 BB/hand | Exact enumeration |
| Neural CFR vs uniform random | +0.61313 BB/hand | 2,000 duplicate pairs; 95% CI [+0.39265, +0.83360] |
| Neural CFR vs calling station | +0.36713 BB/hand | 2,000 duplicate pairs; 95% CI [+0.08077, +0.65348] |
| Neural CFR vs pot pressure | +0.26175 BB/hand | 2,000 duplicate pairs; inconclusive 95% CI [-0.06731, +0.59081] |

The first four poker figures are for six-card heads-up Leduc Hold'em. The neural
figures are for the synthetic 24-card Royal Micro Hold'em benchmark. They
demonstrate a reproducible game-solving pipeline and are not claims about full
no-limit Texas Hold'em or real-money performance.

## Trading systems and execution stress

Project 16 adds a C++20 engine and an independent scan-based reference. Its
Release-active test suite compares trades and FIFO state across 40,000 seeded
events, including cancellation, partial fills, duplicate rejection, and quantity
conservation. Machine-specific benchmark results, workload sizes, timing regions,
and timer-resolution caveats are recorded in the
[C++ project README](../projects/16_cpp_order_book_replay/README.md).

Project 17 compares two fixed quoting policies across six synthetic execution
scenarios. The default experiment uses 48 paired seeds and 1,500 events per seed.
In the baseline run, inventory-aware quoting reduced mean absolute inventory
from 4.006 to 1.206 units, but its mean net P&L difference versus symmetric
quoting was -0.339 model dollars, with a paired 95% interval of [-0.765, 0.087].
The interval does not establish a P&L improvement. This is evidence about the
specified synthetic model, not a real-market return estimate. See the
[execution stress README](../projects/17_execution_stress_lab/README.md) for
all scenarios, seed identifiers, accounting rules, and assumptions.

## Concurrent transport and empirical extensions

[Project 18](../projects/18_market_data_pipeline/RESULTS.md) records five repeats
of a verified 100,000-event synthetic pipeline workload on Linux under WSL2.
Median throughput was 12.94 million events/s with SPSC and 3.38 million with a
mutex queue. SPSC did not win the smaller-burst median p99 comparison; every
repeat and the larger scheduling outliers are retained. UDP failures, snapshot
recovery, queue loss, and stale data are checked separately from the lossless
in-memory benchmark. Results describe this host and workload, not exchange latency.

[Project 19](../projects/19_real_data_volatility/RESULTS.md) extends the portfolio
to public historical BTC/ETH hourly data, with publisher checksums pinned in a
manifest. Model selection uses 2025; held-out evaluation uses January–June 2026.
The study reports forecasting losses and uncertainty rather than simulated
trading profits. Its project report contains the reproduced estimates.

## Verification command

```bash
python -m ruff check .
python -m mypy
python -m pytest
python scripts/run_all.py
python projects/15_neural_poker_solver/run.py
cd projects/14_poker_bot_arena && npm test && npm run lint
```

Python and browser test counts and coverage are regenerated in CI rather than
frozen here. Both stacks enforce linting, typed builds, engine invariants,
policy artifact counts, and deterministic seeded behavior.
