# 07 — Cross-sectional momentum backtest

**Question:** After volatility scaling and execution costs, how does a
dollar-neutral winner-minus-loser portfolio behave in a regime-changing market?

Ranks use trailing returns only. Selected long and short books are independently
inverse-volatility weighted, each receives half of gross exposure, and the
backtest delays execution by one period.

```bash
python projects/07_cross_sectional_momentum/run.py
```

**Limitations:** Synthetic assets have no survivorship bias, corporate actions,
borrow constraints, or capacity limits. The experiment demonstrates portfolio
formation and accounting, not an empirical momentum premium.
