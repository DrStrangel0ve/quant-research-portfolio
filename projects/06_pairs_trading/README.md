# 06 — Rolling pairs-trading backtest

**Question:** Does a stationary synthetic spread remain tradable after estimating
its hedge ratio only from trailing data and charging turnover costs?

Rolling OLS estimates the log-price hedge ratio, a rolling z-score drives
stateful entry/exit rules, gross exposure is normalized, and targets earn
returns only after a one-bar lag.

```bash
python projects/06_pairs_trading/run.py
```

**Limitations:** The data is deliberately cointegrated and therefore validates
implementation, not pair discovery. A real study needs point-in-time universe
selection, stability tests, borrow constraints, and a held-out period.
