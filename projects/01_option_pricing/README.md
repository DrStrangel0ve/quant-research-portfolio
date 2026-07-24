# 01 — Monte Carlo option pricing

**Question:** How quickly does a Monte Carlo European call estimate converge to
the Black-Scholes benchmark, and how much do antithetic variates help?

The experiment simulates risk-neutral GBM terminal prices, prices the discounted
payoff, reports a 95% confidence interval, and compares ordinary sampling with
antithetic sampling across path counts. Black-Scholes is used as an analytic
unit test, not as data fitted after the fact.

```bash
python projects/01_option_pricing/run.py
```

Outputs: `results/summary.json`, `results/convergence.csv`, and
`results/convergence.png`.

**Limitations:** GBM assumes constant volatility, continuous trading, lognormal
prices, and no jumps. Project 02 relaxes constant volatility.
