# 04 — Inventory-aware market making

**Question:** How does risk aversion trade spread capture against inventory
risk in an Avellaneda-Stoikov-inspired quoting policy?

Reservation prices move against inventory, quote width grows with remaining
risk, fills arrive with distance-sensitive Poisson intensities, and P&L is
marked to the final midprice.

```bash
python projects/04_market_making/run.py
```

**Limitations:** Independent Poisson fills omit queue position, informed flow,
latency, maker fees, price impact, and fill/price-shock dependence. Results are
a mechanism study, not a realistic profitability forecast.
