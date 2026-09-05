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
6. **Neural poker solver** — external sampling, reservoir replay,
   suit-isomorphic features, Bayesian ranges, and the boundary between a
   blueprint rollout and theoretically safe continual resolving.

Strong discussion prompt: *What additional evidence would change this from a
mechanism demonstration into a defensible empirical claim?*

## Quant trading

Start with:

- **[Execution stress lab](../projects/17_execution_stress_lab/README.md)** —
  begin with event ordering, the distinction between a submitted order and a
  live quote, queue depletion, adverse selection, and paired policy comparisons.
  Explain why a result can reverse when the execution assumptions change and
  why synthetic evidence cannot establish market alpha.

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

- **[C++ order book and replay](../projects/16_cpp_order_book_replay/README.md)** —
  explain price-time priority, integer ticks, indexing and iterator lifetime,
  compare against the reference implementation, and reproduce the benchmark.
  Discuss workload dependence and the distinction between processing latency
  inside one process and end-to-end exchange latency.

1. **Backtest engine** — typed result objects, explicit audit trail, cost model,
   and one-bar anti-look-ahead semantics.
2. **Limit order book** — stateful data structures and FIFO invariants.
3. **Monte Carlo layer** — injected RNG state, vectorization, confidence
   intervals, and analytic regression tests.
4. **CI and tests** — multi-version checks, strict typing, temporal-invariance
   tests, and reproducible commands.
5. **Poker engine and arena** — immutable transitions, Python/RLCard parity,
   checkpoint serialization, cross-runtime neural inference, and a deployable
   policy debugger.

Strong discussion prompt: *Which abstractions would need to change for
event-driven, multi-venue, asynchronous production use?*

## Evidence to put on a résumé

Use only claims that remain true after publication and CI verification. A
concise draft:

> Built a typed quantitative-research portfolio spanning derivatives, market
> microstructure, systematic strategies, portfolio risk, probability, and
> imperfect-information games; built exact CFR+ and neural CFR poker solvers
> with best-response auditing, duplicate evaluation, Bayesian range search,
> and an interactive cross-runtime policy arena.

Avoid quoting synthetic Sharpe ratios or small-game poker win rates as real-world
performance achievements. The stronger signal is that the framework exposes
leakage, costs, uncertainty, failed hypotheses, and exploitability.

## Explaining the new microstructure projects

Use the commands and measured evidence in each project README. Before putting a
claim on a resume, reproduce it on your own machine and be able to explain the
implementation and at least one limitation without relying on the documentation.

- **Engineering example:** Implemented an integer-tick C++20 order book with
  price-time matching and indexed cancellation; validated event replay against
  a reference implementation and measured throughput and processing-latency
  distributions under seeded workloads.
- **Trading example:** Built a synthetic execution simulator with queue-ahead
  depletion, delayed order activation/cancellation, and adverse selection;
  compared symmetric and inventory-aware quoting on paired event tapes with
  confidence intervals and explicit inventory accounting.

Questions to prepare:

1. Which operation dominates the order book as price levels and queue depth grow?
2. What happens to an order lookup when its resting order is partially filled,
   fully filled, or cancelled?
3. Which benchmark costs are inside the timed region? How does timing each event
   perturb the measurement, and why is throughput measured separately?
4. Can an order fill after its cancellation is requested? Walk through the exact
   event sequence and explain how the simulator tests it.
5. Why can spread capture coexist with negative subsequent fill markouts?
6. What is held constant in the paired comparison, and what does its confidence
   interval fail to measure about the model itself?

The C++ project is single-threaded. It does not establish lock-free concurrency,
networking, or production exchange integration experience. The execution lab
does not establish a profitable strategy on historical or live market data.

## Systems and empirical research extensions

For HFT software engineering, lead with
[project 18](../projects/18_market_data_pipeline/README.md). Follow one packet
from the UDP receiver through a bounded queue, generation fence, parser,
snapshot state machine and risk check. Explain the acquire/release edges and
why a full queue needs a loss signal outside the queue. Reproduce both the fast
throughput run and the poor tail-latency case. The matching core is still owned
by one consumer thread; concurrency is across pipeline stages.

For quant research, lead with
[project 19](../projects/19_real_data_volatility/README.md). Explain source
checksums, the timestamp-unit transition, the 2025 model-selection period, and
the separate 2026 evaluation. Defend the realized-variance proxy and bootstrap
assumptions. A result on two crypto pairs at one venue is limited empirical
evidence, not general market alpha.

Possible resume bullets, after reproducing and understanding the work:

- Built a C++20 UDP feed-processing lab with bounded SPSC/mutex queues,
  transactional snapshot recovery, receipt-age checks and a pure pretrade risk
  gate; benchmarked throughput and burst latency and verified concurrency with
  ThreadSanitizer.
- Evaluated causal volatility forecasts on checksum-verified BTC/ETH hourly
  archives using separate chronological model-selection and test periods,
  baseline comparisons and paired block-bootstrap uncertainty estimates.

The next useful evidence should come from a real exchange protocol adapter,
realistic execution data, a reviewed external contribution, or a research
collaboration. Each addition should answer a specific question, expose failure
cases, and produce reproducible evidence. Repository size is not a hiring metric.
