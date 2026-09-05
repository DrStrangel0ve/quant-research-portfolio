# Execution stress lab

An event-driven **synthetic** experiment comparing symmetric market making with
inventory-aware quoting. It tests whether reducing inventory exposure also improves
net returns once queues, delayed messages, fees, and adverse selection are present.
The measured answer is mixed: inventory falls substantially, while the simple skew
policy does **not** improve mean net P&L in these fixed scenarios. See
[RESULTS.md](RESULTS.md) for the actual run, including negative findings.

This project demonstrates execution modeling, causal simulation, invariant testing,
paired experimental design, and interpretation of a risk/return tradeoff. It does
not estimate real-market alpha or reproduce an exchange matching engine.

## Run

From the repository root, using Python 3.11+:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/test_execution_stress.py --no-cov -q
python projects/17_execution_stress_lab/run.py
```

The fixed default is 48 seeds, 1,500 events each, two policies, and six scenarios.
The recorded run completed in about seven seconds after imports. A smoke run is:

```bash
python projects/17_execution_stress_lab/run.py --seeds 2 --events 50
```

The smoke run overwrites local result files and is only an execution check; it is
too small for reliable inference. Restore the default run to reproduce the table.
`--first-seed`, `--seeds`, `--events`, and `--output` are configurable.

Generated CSV/JSON/PNG files live in `results/` and are ignored by Git. The runner
writes every seed's metrics, paired confidence intervals, the first baseline seed's
full decision/fill/equity traces, configuration and tape SHA-256 hashes, and a plot.
No generated market data or access credentials are required.

## Event chronology and causality

Time is an integer **event index**, not milliseconds. Every event has a reference
mid followed by one initially hidden aggressive order (or no order). At event `t`:

1. Observe the current reference mid. Its change can depend on earlier flow.
2. Deliver all previously scheduled cancellations, then bid and ask activations.
3. On a scheduled decision event, calculate both target quotes using **only the
   current mid and current inventory**; submit cancellations/replacements.
4. Deliver any zero-latency messages using the same global cancel-first order.
5. Reveal the aggressive order, deplete eligible queues, and process unit fills.
6. Record cash, inventory, and mark-to-mid equity.

Orders arriving at `t` are eligible for event `t`'s subsequent flow. A cancel due at
`t` removes the order before that flow. Between submission and cancellation, the
old order remains executable. A replacement activates only after cancellation
latency **plus** activation latency; this can create a quoting gap. Each side has
at most one live quote and one serialized replacement request. A pending request
is not revised. If its old quote fills, the scheduled cancellation becomes a no-op
and the replacement still arrives at its scheduled time.

Activation uses the arrival-time mid and current inventory. Crossing the external
touch, crossing a live opposite own quote, violating the hard inventory cap, or a
nonpositive price rejects the activation. A rejected request may be retried on
the next scheduled decision. Same-time cancels on both sides occur before either
activation, avoiding a spurious own-cross rejection against an already cancelled
quote. Activations are processed bid then ask for deterministic ordering.

All exogenous events are generated before either policy runs, using one NumPy
generator per seed. Both policies receive the **same immutable tape**; they never
draw policy-dependent fills. Generation creates latent informed-flow indicators,
but those indicators are not exposed in `MarketEvent`. The simulator reads future
mids only in a separate post-execution markout pass. A regression test changes the
entire future suffix and verifies unchanged past decisions, executions, and equity.

## Queue and adverse-selection model

Prices are integer ticks worth $0.01 per share by default. Initial mid is 10,000
ticks ($100), with an external touch one tick away on each side. Our quotes each
have size one share. New quotes at the touch begin behind three shares; each level
further away adds two shares ahead. An inside-spread quote has no queue ahead.

A buy aggressor can reach prices up to `mid + half_spread + sweep`; a sell
aggressor can reach prices down to `mid - half_spread - sweep`. If our quote is
reachable, event volume consumes its queue ahead first. Partial queue depletion
persists across events. Filling one unit requires volume **strictly beyond** the
remaining ahead queue. Replacing the quote resets its queue position; keeping the
same price preserves it.

By default, flow arrives on 80% of events. Direction is independent and symmetric;
volume is geometric with parameter 0.3, capped at 12. Sweep is 0/1/2 ticks with
probabilities 0.80/0.15/0.05. There is independent one-tick signed reference-price
noise on 15% of events. A latent 30% of active flow induces a two-tick future move
in the aggressor's direction, two events later. This makes passive fills potentially
informative about an unfavorable subsequent price move. Events scheduled beyond
the final event do not enter terminal valuation or mature markouts.

The baseline policies are fixed before evaluation:

- **Symmetric:** bid/ask one tick around the observed mid, subject to the shared cap.
- **Inventory-aware:** shift the reservation price by `round(0.5 * inventory)` ticks
  against inventory (NumPy round-to-even). Back off the inventory-adding side and
  improve the reducing side, bounded at mid to avoid crossing the external touch.

Both use a decision interval of three events, one-event activation and cancellation
latencies, and the same absolute inventory limit of eight shares. A cap blocks
further inventory-adding activations. Unit sizing and at most one live order per
side make this enforceable even with delayed cancels; it is tested on late-fill
and replacement sequences. This assumption would need explicit exposure
reservation if extended to multiple simultaneous orders or sizes above one.

## Accounting and evaluation

For fill side `s` (+1 buy, -1 sell), price `p` ticks, tick value `v`, and fee `f`:

```text
inventory += s
cash      -= s * p * v + f
equity     = cash + inventory * current_mid * v
net_pnl    = terminal_equity - abs(terminal_inventory) * (liquidation_half_spread * v + f)
```

Default fee is $0.001 per fill. Terminal liquidation charges one tick of spread
and one fee per remaining share. It is an explicit terminal cost deduction, not a
simulated terminal order; reported terminal inventory is the pre-liquidation
position. No fill or fee is counted twice. Tests verify both an open position and
a completed round trip. Drawdown includes the starting zero equity and final net
liquidation value.

Diagnostics include mean/max absolute inventory, RMS inventory, max equity
drawdown, fill counts, rejected activations, and terminal MTM/cash/cost. For fills
with a complete ten-event future horizon:

```text
net_markout         = s * (future_mid - fill_price) * tick_value - fill_fee
directional_markout = s * (future_mid - mid_at_fill) * tick_value
```

Negative directional markout indicates adverse post-fill movement. Net markout
also includes spread capture and fees. End-of-tape censored fills are omitted and
the mature count is reported. Markouts are diagnostics, not additional P&L.

Each seed contributes one value per policy and metric. The comparison reports
`inventory_aware - symmetric`, the average paired difference, and a Student-t 95%
confidence interval using **independent seeds as the sampling unit**. Fills within
a seed are not treated as independent samples. Markout means weight eligible
seeds equally, not fills equally. Missing/nonfinite diagnostics are excluded
pairwise; `n_pairs` records the denominator. Fewer than two pairs produces an
unavailable interval rather than a fabricated value.

The six fixed scenarios alter one baseline assumption at a time: no adverse
selection, stronger adverse selection (70%), slower messages (four events each),
longer initial queue (ten shares), and higher fill fees ($0.005). These are model
sensitivity comparisons; scenario intervals are unadjusted for multiple testing.
They quantify Monte Carlo variability conditional on this generator, not model
uncertainty. Seeds and policy parameters were not selected for favorable results.

## Limitations and useful extensions

This is a reduced execution model. The reference price is exogenous, not formed
by a full visible order book. Queue ahead is an aggregate initialized when our
quote arrives; it omits cancellations ahead, arrivals ahead, hidden/iceberg size,
exchange auctions, and multi-venue routing. A reaching aggressive order's volume
is the modeled volume available at our quote's queue; it is not conserved through
all competing external levels. Reference-price moves do not themselves execute
resting orders, and no instantaneous arbitrage agent clears stale quotes. Only
the next reachable aggressive event can fill them. These simplifications can
overstate both spread capture and the persistence of stale liquidity.

Flow, volatility, and latency are stationary; no realistic session length, message
timestamps, financing, borrow, capital constraints, rebates, or market impact is
modeled. Dollar values reflect unit sizing and the chosen tick convention. There
is no Sharpe annualization and no conversion of event delays into wall-clock
latency. The equity paths are not historical backtests.

A stronger next experiment would calibrate queue survival and conditional
markouts against time-stamped public market data, then evaluate a frozen policy
on a held-out period. Any data-derived policy tuning would require a separate
training/evaluation split and disclosure of the search process. The current
negative finding is useful without adding a tuned strategy.

## Source and tests

- Model: [`execution_stress.py`](../../src/quantlab/simulation/execution_stress.py)
- Runner: [`run.py`](run.py)
- Invariants: [`test_execution_stress.py`](../../tests/test_execution_stress.py)
- Recorded experiment: [`RESULTS.md`](RESULTS.md)
