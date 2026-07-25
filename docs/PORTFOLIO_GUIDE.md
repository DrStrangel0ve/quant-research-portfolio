# Portfolio guide for quant recruiting

This repository is deliberately broad, but interview narratives should be
focused. Choose the path closest to the role instead of presenting all projects
with equal weight.

## Quant research

Start with:

1. **Walk-forward trend following** — chronological selection, embargo, honest
   negative OOS evidence, and the difference between model selection and final
   evaluation.
2. **Rolling pairs trading** — parameter instability, spread stationarity,
   execution lag, and why synthetic cointegration is only an implementation
   check.
3. **GARCH forecasting** — likelihood recursion, constrained parameters, and
   why Gaussian innovations understate tail risk.
4. **Risk parity** — covariance estimation error, shrinkage, and component risk.
5. **Poker CFR+ lab** — counterfactual regret, exact exploitability, imperfect
   information, and why a head-to-head win is not an equilibrium certificate.

Strong discussion prompt: *What additional evidence would change this from a
mechanism demonstration into a defensible empirical claim?*

## Quant trading

Start with:

1. **Inventory-aware market making** — reservation-price skew, spread/fill
   tradeoffs, and why independent Poisson fills are optimistic.
2. **Limit order book** — matching priority, partial fills, and marketable
   limits.
3. **Delta hedging** — discrete replication, realized versus implied volatility,
   and transaction-cost convexity.
4. **Kelly sizing** — growth optimality versus drawdown and parameter risk.

Strong discussion prompt: *Which omitted execution mechanism is most likely to
reverse the simulated result?*

## Quant developer

Start with:

1. **Backtest engine** — typed result objects, explicit audit trail, cost model,
   and one-bar anti-look-ahead semantics.
2. **Limit order book** — stateful data structures and FIFO invariants.
3. **Monte Carlo layer** — injected RNG state, vectorization, confidence
   intervals, and analytic regression tests.
4. **CI and tests** — multi-version checks, strict typing, temporal-invariance
   tests, and reproducible commands.
5. **Poker engine and arena** — immutable transitions, Python/RLCard parity,
   checkpoint serialization, and a deployable policy debugger.

Strong discussion prompt: *Which abstractions would need to change for
event-driven, multi-venue, asynchronous production use?*

## Evidence to put on a résumé

Use only claims that remain true after publication and CI verification. A
concise draft:

> Built a typed quantitative-research portfolio spanning derivatives, market
> microstructure, systematic strategies, portfolio risk, probability, and
> imperfect-information games; added a from-scratch CFR+ poker solver with
> exact best-response auditing and an interactive trained-policy arena.

Avoid quoting synthetic Sharpe ratios or small-game poker win rates as real-world
performance achievements. The stronger signal is that the framework exposes
leakage, costs, uncertainty, failed hypotheses, and exploitability.
