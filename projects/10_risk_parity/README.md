# 10 — Rolling risk parity portfolio

**Question:** Does equal risk contribution diversify volatility more effectively
than equal capital weights when asset volatilities differ?

Every monthly rebalance estimates a shrunk covariance matrix from the previous
60 observations only. A multiplicative solver targets equal component
contributions to portfolio volatility. Both portfolios pay identical costs.

```bash
python projects/10_risk_parity/run.py
```

**Limitations:** Covariance is unstable, linear shrinkage is deliberately simple,
and long-only risk parity can concentrate in low-volatility assets with hidden
tail exposure.
