# Recorded synthetic execution experiment

Executed on September 5, 2026 with Python 3.13.3. Reproduce from repository root:

```bash
python projects/17_execution_stress_lab/run.py --seeds 48 --events 1500 --first-seed 202600
```

This run uses seeds 202600–202647, six fixed scenarios, and identical exogenous
tapes for both policies within each scenario/seed. No policy or seed tuning was
performed after seeing these results. Simulation, output generation, and plotting
took 6.66 seconds on the development machine after imports; this is not a trading
latency benchmark. All 48 seed pairs had mature markouts in every scenario.

## Mean net P&L per seed

Dollar amounts use one-share fills and $0.01 ticks. Net P&L includes every fill
fee and a terminal spread/fee deduction for remaining inventory. Difference and
95% Student-t interval are **inventory-aware minus symmetric**, paired by seed.

| Scenario | Symmetric ($) | Inventory-aware ($) | Paired difference ($) | 95% CI ($) |
|---|---:|---:|---:|---:|
| Baseline | -0.1293 | -0.4683 | -0.3390 | [-0.7649, 0.0870] |
| No adverse selection | 2.6197 | 2.1319 | -0.4878 | [-0.6438, -0.3318] |
| Strong adverse selection | -1.0020 | -2.0520 | -1.0500 | [-1.8239, -0.2761] |
| Slow messages | -0.3111 | -0.3525 | -0.0414 | [-0.4911, 0.4084] |
| Long queue | -0.2850 | -0.6553 | -0.3703 | [-0.8861, 0.1455] |
| High fees | -1.1231 | -1.4771 | -0.3540 | [-0.7761, 0.0682] |

The simple inventory skew produced a lower mean P&L in all six scenarios. Baseline
uncertainty includes zero, so this run does not resolve the sign of its expected
P&L difference. The no-adverse-selection and strong-adverse-selection intervals
exclude zero below it. These intervals are not adjusted for multiple comparisons
and describe only this synthetic generator.

## Inventory and drawdown tradeoff

| Scenario | Mean absolute inventory, symmetric | Mean absolute inventory, aware | Mean max drawdown, symmetric ($) | Mean max drawdown, aware ($) |
|---|---:|---:|---:|---:|
| Baseline | 4.006 | 1.206 | 2.385 | 1.019 |
| No adverse selection | 3.700 | 0.981 | 0.429 | 0.082 |
| Strong adverse selection | 3.766 | 1.748 | 3.619 | 2.584 |
| Slow messages | 3.856 | 1.783 | 2.272 | 1.213 |
| Long queue | 3.837 | 0.970 | 2.381 | 0.967 |
| High fees | 4.006 | 1.206 | 2.750 | 1.693 |

Inventory awareness reduced average inventory and average maximum drawdown in
all six scenarios. The mechanism is consistent with backing off inventory-adding
quotes and accepting less spread on reducing quotes. It does not establish that
the policy is preferable without specifying a risk budget or risk-adjusted
objective in advance.

For baseline, equal-seed mean ten-event **net fill markout** was -$0.000855 for
symmetric and -$0.001847 for inventory-aware. Directional markouts were -$0.005661
and -$0.005153, respectively: both experienced adverse post-fill price movement.
In the no-adverse-selection scenario, net markouts were positive ($0.007267 and
$0.005997). This illustrates how removing informative order flow can make an
otherwise weak quoting rule appear profitable in simulation.

## What this result supports

The implementation can reproduce a controlled execution experiment, detect
inventory/P&L tradeoffs, and preserve unfavorable strategy findings. It does not
demonstrate a profitable strategy, real-world execution skill, or a calibrated
estimate of adverse-selection losses. The full assumptions and omissions are
listed in [README.md](README.md).

Re-running writes `results/seed_metrics.csv`, `paired_summary.csv`,
`run_metadata.json` (all configurations and tape hashes), and
`paired_comparison.png`. Raw outputs are generated locally and ignored by Git;
this compact recorded report is the checked-in evidence.
