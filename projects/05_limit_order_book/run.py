from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quantlab.microstructure.order_book import LimitOrderBook, Order, Side
from quantlab.reporting.artifacts import write_summary


def main() -> None:
    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)
    rng = np.random.default_rng(5005)
    book = LimitOrderBook()
    next_order_id = 1
    for level in range(1, 11):
        book.add_limit_order(Order(next_order_id, Side.BUY, 100.0 - 0.01 * level, 20))
        next_order_id += 1
        book.add_limit_order(Order(next_order_id, Side.SELL, 100.0 + 0.01 * level, 20))
        next_order_id += 1

    spreads: list[float] = []
    trade_prices: list[float] = []
    trade_count = 0
    for _ in range(5_000):
        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        if rng.random() < 0.35:
            trades = book.execute_market_order(side, int(rng.integers(1, 8)))
            trade_count += len(trades)
            trade_prices.extend(trade.price for trade in trades)
        else:
            reference = 100.0 + rng.normal(0.0, 0.025)
            offset = abs(rng.normal(0.02, 0.015))
            price = reference - offset if side == Side.BUY else reference + offset
            trades = book.add_limit_order(
                Order(next_order_id, side, round(max(price, 0.01), 2), int(rng.integers(1, 10)))
            )
            next_order_id += 1
            trade_count += len(trades)
            trade_prices.extend(trade.price for trade in trades)
        if book.spread is not None:
            spreads.append(book.spread)

    plt.hist(spreads, bins=30)
    plt.xlabel("Quoted spread")
    plt.ylabel("Observations")
    plt.title("Simulated limit-order-book spreads")
    plt.tight_layout()
    plt.savefig(output / "spread_distribution.png", dpi=160)
    plt.close()
    write_summary(
        output / "summary.json",
        {
            "events": 5_000,
            "trade_messages": trade_count,
            "mean_spread": float(np.mean(spreads)),
            "median_spread": float(np.median(spreads)),
            "mean_trade_price": float(np.mean(trade_prices)),
            "final_best_bid": float(book.best_bid or 0.0),
            "final_best_ask": float(book.best_ask or 0.0),
        },
    )


if __name__ == "__main__":
    main()
