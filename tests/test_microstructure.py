import numpy as np
import pytest

from quantlab.microstructure.order_book import LimitOrderBook, Order, Side
from quantlab.simulation.market_making import MarketMakerParameters, simulate_market_maker


def test_limit_order_book_respects_price_time_priority() -> None:
    book = LimitOrderBook()
    book.add_limit_order(Order(1, Side.SELL, 100.0, 5))
    book.add_limit_order(Order(2, Side.SELL, 100.0, 5))
    trades = book.execute_market_order(Side.BUY, 7)
    assert [(trade.resting_order_id, trade.quantity) for trade in trades] == [(1, 5), (2, 2)]
    assert book.depth(Side.SELL, 100.0) == 3


def test_marketable_limit_order_rests_its_remainder() -> None:
    book = LimitOrderBook()
    book.add_limit_order(Order(1, Side.SELL, 100.0, 2))
    trades = book.add_limit_order(Order(2, Side.BUY, 101.0, 5))
    assert len(trades) == 1
    assert trades[0].price == 100.0
    assert book.best_bid == 101.0
    assert book.depth(Side.BUY, 101.0) == 3


def test_cancellation_removes_order() -> None:
    book = LimitOrderBook()
    book.add_limit_order(Order(1, Side.BUY, 99.0, 4))
    assert book.cancel(1)
    assert not book.cancel(1)
    assert book.best_bid is None


def test_market_maker_is_reproducible_and_inventory_bounded() -> None:
    parameters = MarketMakerParameters(max_inventory=3)
    arguments = {
        "parameters": parameters,
        "horizon": 1.0,
        "n_steps": 100,
        "n_paths": 200,
    }
    first = simulate_market_maker(**arguments, rng=np.random.default_rng(11))
    second = simulate_market_maker(**arguments, rng=np.random.default_rng(11))
    assert first.paths.equals(second.paths)
    assert np.abs(first.terminal_inventory).max() <= 3
    assert first.pnl_std >= 0.0


def test_invalid_order_is_rejected() -> None:
    with pytest.raises(ValueError):
        Order(1, Side.BUY, 0.0, 1)
