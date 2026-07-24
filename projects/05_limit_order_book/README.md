# 05 — Price-time-priority limit order book

**Question:** Can a small exchange core preserve FIFO priority, partial fills,
cancellation semantics, and a non-crossed book under stochastic order flow?

The reusable book has explicit orders and trades. The experiment seeds both
sides and mixes passive limit orders with aggressive market orders while
recording spread and trade-price distributions.

```bash
python projects/05_limit_order_book/run.py
```

**Limitations:** The order-flow generator is illustrative; it is not calibrated
to message-level exchange data and has no latency or hidden liquidity.
