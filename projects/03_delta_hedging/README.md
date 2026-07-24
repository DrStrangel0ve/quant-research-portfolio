# 03 — Discrete delta hedging

**Question:** How do hedge frequency, volatility misspecification, and trading
costs change the distribution of a short option's replication error?

The short-call portfolio is self-financing: the premium funds delta shares,
cash accrues at the risk-free rate, and every rebalance plus final liquidation
pays proportional costs.

```bash
python projects/03_delta_hedging/run.py
```

**Limitations:** The simulated underlying is GBM and has no jumps, stochastic
volatility, discrete ticks, funding spread, or market impact.
