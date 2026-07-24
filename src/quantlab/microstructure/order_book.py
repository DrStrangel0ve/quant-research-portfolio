"""Minimal price-time-priority limit order book for simulation experiments."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import StrEnum


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    order_id: int
    side: Side
    price: float
    quantity: int

    def __post_init__(self) -> None:
        if self.order_id < 0:
            raise ValueError("order_id cannot be negative")
        if self.price <= 0.0:
            raise ValueError("price must be positive")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")


@dataclass(frozen=True)
class Trade:
    resting_order_id: int
    aggressor_side: Side
    price: float
    quantity: int


class LimitOrderBook:
    """Single-instrument order book with FIFO queues at each price level."""

    def __init__(self) -> None:
        self._bids: dict[float, deque[Order]] = defaultdict(deque)
        self._asks: dict[float, deque[Order]] = defaultdict(deque)
        self._orders: dict[int, Order] = {}

    @property
    def best_bid(self) -> float | None:
        return max(self._bids, default=None)

    @property
    def best_ask(self) -> float | None:
        return min(self._asks, default=None)

    @property
    def spread(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    def depth(self, side: Side, price: float) -> int:
        book = self._bids if side == Side.BUY else self._asks
        return sum(order.quantity for order in book.get(price, ()))

    def add_limit_order(self, order: Order) -> list[Trade]:
        """Match a marketable limit order, then rest any remainder."""
        if order.order_id in self._orders:
            raise ValueError("order_id must be unique")
        trades = self._match(order)
        if order.quantity > 0:
            book = self._bids if order.side == Side.BUY else self._asks
            book[order.price].append(order)
            self._orders[order.order_id] = order
        return trades

    def execute_market_order(self, side: Side, quantity: int) -> list[Trade]:
        """Execute against available depth; unfilled market quantity is cancelled."""
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        synthetic = Order(order_id=0, side=side, price=float("inf"), quantity=quantity)
        if side == Side.SELL:
            synthetic.price = 1e-12
        return self._match(synthetic)

    def cancel(self, order_id: int) -> bool:
        """Cancel a resting order, returning whether it existed."""
        order = self._orders.pop(order_id, None)
        if order is None:
            return False
        book = self._bids if order.side == Side.BUY else self._asks
        queue = book[order.price]
        queue.remove(order)
        if not queue:
            del book[order.price]
        return True

    def _match(self, aggressor: Order) -> list[Trade]:
        opposite = self._asks if aggressor.side == Side.BUY else self._bids
        trades: list[Trade] = []
        while aggressor.quantity > 0 and opposite:
            best_price = min(opposite) if aggressor.side == Side.BUY else max(opposite)
            crosses = (
                aggressor.price >= best_price
                if aggressor.side == Side.BUY
                else aggressor.price <= best_price
            )
            if not crosses:
                break
            queue = opposite[best_price]
            resting = queue[0]
            executed = min(aggressor.quantity, resting.quantity)
            trades.append(
                Trade(
                    resting_order_id=resting.order_id,
                    aggressor_side=aggressor.side,
                    price=best_price,
                    quantity=executed,
                )
            )
            aggressor.quantity -= executed
            resting.quantity -= executed
            if resting.quantity == 0:
                queue.popleft()
                self._orders.pop(resting.order_id)
            if not queue:
                del opposite[best_price]
        return trades
