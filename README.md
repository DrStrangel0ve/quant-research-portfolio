# Quantitative Research Portfolio

A collection of reproducible projects in derivatives, market microstructure,
systematic trading, portfolio construction, time-series modelling, and applied
probability. The emphasis is not on impressive-looking in-sample curves; it is
on transparent assumptions, correct temporal alignment, uncertainty estimates,
execution frictions, and experiments that can be challenged in an interview.

> Educational research only. Nothing in this repository is investment advice,
> and synthetic results are not evidence that a strategy will earn live profits.

## Project map

| # | Project | Area | Primary signal |
|---:|---|---|---|
| 01 | [Monte Carlo option pricing](projects/01_option_pricing/README.md) | Derivatives | MC error, variance reduction, analytic benchmark |
| 02 | [Heston stochastic volatility](projects/02_heston_model/README.md) | Derivatives | Correlated spot/variance paths and implied-volatility skew |
| 03 | [Discrete delta hedging](projects/03_delta_hedging/README.md) | Derivatives | Hedging-error distribution vs. frequency and costs |
| 04 | [Inventory-aware market making](projects/04_market_making/README.md) | Microstructure | Quote skew, Poisson fills, terminal inventory and P&L |
| 05 | [Limit order book](projects/05_limit_order_book/README.md) | Microstructure | Price-time priority and agent-based order flow |
| 06 | [Rolling pairs trading](projects/06_pairs_trading/README.md) | Backtesting | Rolling hedge ratio, stateful z-score entries and exits |
| 07 | [Cross-sectional momentum](projects/07_cross_sectional_momentum/README.md) | Backtesting | Long winners/short losers with inverse-vol sizing |
| 08 | [Walk-forward trend following](projects/08_walk_forward_trend/README.md) | Backtesting | Chronological model selection and volatility targeting |
| 09 | [GARCH volatility forecasting](projects/09_garch_forecasting/README.md) | Time series | Constrained maximum likelihood and variance forecasts |
| 10 | [Risk parity portfolio](projects/10_risk_parity/README.md) | Portfolio risk | Shrunk covariance and equal risk contribution |
| 11 | [Ruin and information games](projects/11_probability_games/README.md) | Probability | Gambler's ruin and Monty Hall: exact vs. simulation |
| 12 | [Kelly and optimal stopping](projects/12_kelly_and_stopping/README.md) | Probability | Log-optimal sizing and the secretary problem |
| 13 | [Poker CFR+ lab](projects/13_poker_cfr_lab/README.md) | Game theory | From-scratch CFR+, exact best response, and RLCard face-off |
| 14 | [Playable poker bot arena](projects/14_poker_bot_arena/README.md) | Interactive research | Human play against trained, reference, exact, and heuristic bots |
| 15 | [Neural poker solver](projects/15_neural_poker_solver/README.md) | Game theory / ML | Deep CFR, suit-canonical features, Bayesian ranges, and rollout search |
| 16 | [C++ order book and replay](projects/16_cpp_order_book_replay/README.md) | Trading systems | Integer ticks, indexed cancellation, differential testing, and measured replay performance |
| 17 | [Execution stress lab](projects/17_execution_stress_lab/README.md) | Trading experiments | Queue depletion, delayed order messages, adverse selection, and paired policy evaluation |
| 18 | [Market-data pipeline](projects/18_market_data_pipeline/README.md) | Systems engineering | UDP, bounded concurrency, recovery, freshness, risk checks, and measured tail latency |
| 19 | [Real-data volatility study](projects/19_real_data_volatility/README.md) | Empirical research | Verified public archives, chronological selection, held-out forecast evaluation, and uncertainty |

For systems engineering, start with project 18 and trace a lost packet through
recovery; project 16 supplies its matching engine and reference comparison.
For empirical research, start with project 19 and explain the selection/test
boundary. For execution modeling, project 17 makes queue and latency assumptions
explicit. These projects extend the simpler models earlier in the portfolio.

## Research safeguards

- **No same-bar execution:** strategy signals are lagged before earning returns.
- **Chronological evaluation:** walk-forward splits never shuffle time series.
- **Costs are first-class:** commission, spread, slippage, and turnover appear in
  every strategy scorecard.
- **Reproducibility:** every stochastic experiment receives an explicit seeded
  `numpy.random.Generator`.
- **Uncertainty is reported:** Monte Carlo estimates include standard errors or
  empirical distributions, not only point estimates.
- **Synthetic data is labelled:** offline examples make the code reproducible;
  they are demonstrations, not claims of market alpha.
- **Audit trails:** backtests expose positions, turnover, gross returns, costs,
  net returns, and equity independently.

See [RESEARCH_STANDARDS.md](RESEARCH_STANDARDS.md) for the review checklist.
For a compact audit of the seeded runs, see
[Verified results](docs/RESULTS.md). For role-specific navigation and interview
prompts, see the [Portfolio guide](docs/PORTFOLIO_GUIDE.md).

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
python projects/01_option_pricing/run.py
python projects/13_poker_cfr_lab/run.py
python -m pip install -e ".[neural-poker]"
python projects/15_neural_poker_solver/run.py
python projects/17_execution_stress_lab/run.py
python scripts/run_all.py
```

Projects 16 and 18 use C++20 and CMake independently of the Python package. Their
READMEs contain build, correctness-test, and benchmark commands. Project 19 is
opt-in because it downloads public market-data archives on first use:

```bash
python projects/19_real_data_volatility/run.py --download
# After the verified archives are cached, rerun offline:
python projects/19_real_data_volatility/run.py
```

Each project writes generated artifacts beneath its own `results/` directory.
Those artifacts are intentionally gitignored so a clean checkout always proves
that the documented experiment is reproducible.

## Repository architecture

```text
src/quantlab/        reusable, tested research components
projects/            independent experiments and project notes
projects/14_*/       deployable exact and neural TypeScript poker arena
projects/15_*/       optional PyTorch game-solving experiment
projects/16_*/       standalone C++20 order book, replay, tests, and benchmarks
projects/17_*/       causal execution stress experiments and paired results
projects/18_*/       UDP transport, bounded queues, recovery, risk and profiling
projects/19_*/       checksum-pinned historical data and held-out volatility study
tests/               unit, invariant, and anti-look-ahead tests
.github/workflows/   Python, browser, cross-compiler C++, and sanitizer checks
```

## What to discuss in an interview

Start with a failed assumption, not the prettiest chart. Useful discussion
prompts include the Heston discretization bias, fill-model optimism in the
market maker, instability of a rolling pairs hedge ratio, turnover drag in
momentum, covariance estimation error in risk parity, and over-betting under
parameter uncertainty in Kelly sizing. For game-theory roles, discuss why the
poker bot reports exact exploitability rather than only a head-to-head win rate,
why Leduc results do not imply frontier no-limit Hold'em performance, and why
the larger neural game switches to explicitly approximate cross-play and search.
