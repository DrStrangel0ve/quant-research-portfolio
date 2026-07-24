# Research standards

Every project should survive the following review before it is described as a
backtest or empirical result.

## Data and timing

- State whether data is synthetic, public, licensed, or user-supplied.
- Preserve timestamps and document the decision time and execution time.
- Lag features, parameters, and positions whenever the same observation cannot
  be known and traded at the quoted price.
- Fit scalers, models, hedge ratios, and covariance matrices on the training
  window only.
- Account for delistings, corporate actions, and point-in-time membership before
  using the experiment with real equity data.

## Trading mechanics

- Report gross and net results.
- Define turnover unambiguously as one-way or two-way.
- Include commissions, half-spread, and slippage assumptions.
- Add borrow costs, locate constraints, market impact, and capacity analysis
  before interpreting short or high-turnover strategies commercially.
- Do not treat close-to-close signals as fills at that same close.

## Statistical evidence

- Separate hypothesis formation, parameter selection, and final evaluation.
- Prefer chronological or purged splits to random cross-validation.
- Report uncertainty, drawdowns, tail risk, and the number of independent bets.
- Track every attempted variant; a single selected Sharpe ratio is subject to
  multiple-testing bias.
- Stress parameters and costs instead of reporting one favorable configuration.

## Reproducibility

- Seed pseudo-random generators explicitly.
- Keep core logic in importable modules and thin experiment scripts.
- Test analytic identities, invariants, temporal alignment, and edge cases.
- Record environment and assumptions in version-controlled text.
- Make generated files disposable: a clean checkout must regenerate them.

## Interpretation

- Synthetic experiments validate mechanisms and code, not live profitability.
- A backtest is evidence about a historical simulation under stated assumptions.
- Correlation is not causality, stationarity is not permanent, and liquidity is
  not guaranteed.
