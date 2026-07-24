# 09 — GARCH volatility forecasting

**Question:** Can constrained Gaussian maximum likelihood recover volatility
clustering and produce a mean-reverting variance forecast?

The implementation exposes the likelihood recursion, constrains
`omega > 0`, `alpha >= 0`, `beta >= 0`, and `alpha + beta < 1`, then compares
estimated persistence with the known parameters of seeded synthetic data.

```bash
python projects/09_garch_forecasting/run.py
```

**Limitations:** Gaussian innovations understate heavy tails, the mean is fixed
at zero, and optimizer uncertainty is not quantified.
