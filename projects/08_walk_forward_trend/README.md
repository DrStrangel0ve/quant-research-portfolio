# 08 — Walk-forward trend following

**Question:** Can a signal horizon be selected chronologically rather than by
choosing the best full-sample backtest?

For each expanding training window, the experiment evaluates three trailing
return horizons, selects the training Sharpe winner, freezes that choice for the
next test block, and stitches only out-of-sample targets together. Positions are
volatility targeted, leverage capped, lagged, and charged costs.

```bash
python projects/08_walk_forward_trend/run.py
```

**Limitations:** Testing several horizons still consumes a research degree of
freedom. A final untouched dataset and a richer cost/capacity model are needed
before drawing investment conclusions.
