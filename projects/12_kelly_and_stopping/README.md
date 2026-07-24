# 12 — Kelly sizing and optimal stopping

**Question:** Why does maximizing expected log wealth imply fractional sizing,
and why does sampling roughly the first `1/e` of candidates work in the
secretary problem?

The betting experiment compares half-Kelly, full-Kelly, and over-betting across
terminal wealth distributions. The stopping experiment sweeps rejection
fractions and estimates the probability of selecting the best candidate.

```bash
python projects/12_kelly_and_stopping/run.py
```

**Limitations:** Kelly is extremely sensitive to payoff and probability error.
Real applications require parameter uncertainty, correlated opportunities,
drawdown constraints, and fractional sizing.
