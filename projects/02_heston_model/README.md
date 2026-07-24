# 02 — Heston stochastic-volatility simulation

**Question:** Can correlated spot and variance shocks generate non-Gaussian
returns and strike-dependent implied volatility?

The experiment uses a full-truncation Euler scheme, reports whether the chosen
parameters satisfy the Feller condition, prices a strip of calls from shared
terminal paths, and numerically inverts Black-Scholes prices.

```bash
python projects/02_heston_model/run.py
```

**Limitations:** Euler discretization is biased at coarse time steps, simulated
variance is truncated at zero, and this is not a production calibration engine.
