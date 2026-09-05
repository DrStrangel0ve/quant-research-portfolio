"""Execution invariants tested on hand-constructed tapes with known outcomes."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quantlab.simulation.execution_stress import (
    MarketEvent,
    RestingOrder,
    StressConfig,
    generate_tape,
    paired_summary,
    quote_prices,
    simulate_execution,
)


def test_fifo_requires_volume_beyond_ahead_queue() -> None:
    order = RestingOrder(side=1, price=99, queue_ahead=3)
    assert not order.consume(2)
    assert order.queue_ahead == 1
    assert not order.consume(1)
    assert order.queue_ahead == 0
    assert order.consume(1)
    with pytest.raises(ValueError):
        order.consume(-1)


def test_queue_depletion_persists_across_events() -> None:
    config = StressConfig(activation_latency=0, quote_interval=10, queue_ahead=3)
    tape = (MarketEvent(100, -1, 2), MarketEvent(100, -1, 1), MarketEvent(100, -1, 1))
    result = simulate_execution(tape, policy="symmetric", config=config)
    assert result.fills["event"].tolist() == [2]
    assert result.fills["price_ticks"].tolist() == [99]


def test_order_cannot_fill_before_activation_and_arrival_precedes_flow() -> None:
    config = StressConfig(activation_latency=2, quote_interval=10, queue_ahead=0)
    result = simulate_execution(
        tuple(MarketEvent(100, 1, 1) for _ in range(4)),
        policy="symmetric",
        config=config,
    )
    assert result.fills["event"].tolist() == [2]


def test_cancellation_latency_exposes_stale_order() -> None:
    config = StressConfig(activation_latency=0, cancellation_latency=2, quote_interval=1)
    tape = (MarketEvent(100), MarketEvent(103, 1, 10))
    delayed = simulate_execution(tape, policy="symmetric", config=config)
    instant = simulate_execution(
        tape,
        policy="symmetric",
        config=replace(config, cancellation_latency=0),
    )
    assert delayed.fills["price_ticks"].tolist() == [101]
    assert instant.fills["price_ticks"].tolist() == [104]


def test_replacement_waits_for_cancel_then_activation() -> None:
    config = StressConfig(activation_latency=1, cancellation_latency=2, quote_interval=1)
    tape = tuple(MarketEvent(mid) for mid in [100, 100, 101, 101, 101, 101])
    result = simulate_execution(tape, policy="symmetric", config=config)
    # Initial bid arrives t=1. Reprice submitted t=2; cancel t=4; replacement t=5.
    assert result.trace.loc[3, "bid_price"] == 99
    assert np.isnan(result.trace.loc[4, "bid_price"])
    assert result.trace.loc[5, "bid_price"] == 100


def test_crossing_quote_is_rejected_using_arrival_time_mid() -> None:
    config = StressConfig(activation_latency=2, quote_interval=10)
    tape = (MarketEvent(100), MarketEvent(100), MarketEvent(97, -1, 10))
    result = simulate_execution(tape, policy="symmetric", config=config)
    assert result.metrics["fills"] == 0
    assert result.metrics["rejected_activations"] == 1


def test_activation_cannot_cross_own_order_awaiting_cancel() -> None:
    config = StressConfig(
        activation_latency=0,
        cancellation_latency=2,
        quote_interval=1,
        queue_ahead=0,
    )
    tape = (MarketEvent(100, -1, 1), MarketEvent(103))
    result = simulate_execution(tape, policy="symmetric", config=config)
    assert result.trace.loc[1, "ask_price"] == 101
    assert np.isnan(result.trace.loc[1, "bid_price"])
    assert result.metrics["rejected_activations"] == 1


def test_nonpositive_quote_is_rejected() -> None:
    config = StressConfig(activation_latency=0, initial_mid=1)
    result = simulate_execution((MarketEvent(1, -1, 10),), policy="symmetric", config=config)
    assert result.metrics["fills"] == 0
    assert result.metrics["rejected_activations"] == 1


def test_future_events_cannot_change_past_decisions_fills_or_equity() -> None:
    config = StressConfig(n_events=80)
    tape = generate_tape(config, seed=19)
    prefix = 40
    changed = tape[:prefix] + tuple(replace(event, mid=event.mid + 30) for event in tape[prefix:])
    first = simulate_execution(tape, policy="inventory_aware", config=config)
    second = simulate_execution(changed, policy="inventory_aware", config=config)
    pd.testing.assert_frame_equal(first.trace.iloc[:prefix], second.trace.iloc[:prefix])
    pd.testing.assert_frame_equal(
        first.decisions.query("event < @prefix"),
        second.decisions.query("event < @prefix"),
    )
    execution_columns = ["event", "side", "price_ticks", "mid_ticks", "inventory_after", "fee"]
    pd.testing.assert_frame_equal(
        first.fills.query("event < @prefix")[execution_columns],
        second.fills.query("event < @prefix")[execution_columns],
    )


def test_current_aggressive_order_is_not_visible_to_decision() -> None:
    config = StressConfig(activation_latency=0)
    buy = simulate_execution((MarketEvent(100, 1, 10),), policy="symmetric", config=config)
    sell = simulate_execution((MarketEvent(100, -1, 10),), policy="symmetric", config=config)
    pd.testing.assert_frame_equal(buy.decisions, sell.decisions)
    assert buy.metrics["terminal_inventory"] == -1
    assert sell.metrics["terminal_inventory"] == 1


def test_accounting_marks_inventory_then_charges_liquidation_once() -> None:
    config = StressConfig(
        activation_latency=0,
        quote_interval=10,
        queue_ahead=0,
        tick_value=0.5,
        fee_per_fill=0.1,
        liquidation_half_spread=2,
        markout_horizon=1,
    )
    result = simulate_execution(
        (MarketEvent(100, -1, 1), MarketEvent(102)),
        policy="symmetric",
        config=config,
    )
    assert result.metrics["terminal_cash"] == pytest.approx(-49.6)
    assert result.metrics["terminal_inventory"] == 1
    assert result.metrics["terminal_mtm"] == pytest.approx(1.4)
    assert result.metrics["liquidation_cost"] == pytest.approx(1.1)
    assert result.metrics["net_pnl"] == pytest.approx(0.3)
    assert result.fills.loc[0, "net_markout"] == pytest.approx(1.4)
    assert result.fills.loc[0, "directional_markout"] == pytest.approx(1.0)


def test_round_trip_cash_and_no_terminal_liquidation_charge() -> None:
    config = StressConfig(activation_latency=0, quote_interval=10, queue_ahead=0)
    result = simulate_execution(
        (MarketEvent(100, -1, 1), MarketEvent(100, 1, 1)),
        policy="symmetric",
        config=config,
    )
    assert result.metrics["terminal_inventory"] == 0
    assert result.metrics["liquidation_cost"] == 0
    assert result.metrics["net_pnl"] == pytest.approx(0.02 - 2 * config.fee_per_fill)


def test_hard_cap_survives_inflight_replacements_and_late_fills() -> None:
    config = StressConfig(
        activation_latency=1,
        cancellation_latency=3,
        quote_interval=1,
        queue_ahead=0,
        inventory_limit=2,
    )
    tape = tuple(MarketEvent(100 + index % 2, -1, 20) for index in range(40))
    result = simulate_execution(tape, policy="symmetric", config=config)
    assert result.metrics["terminal_inventory"] == 2
    assert result.metrics["max_abs_inventory"] == 2
    assert result.metrics["fills"] == 2


def test_inventory_skew_backs_off_adding_side_and_improves_reducing_side() -> None:
    config = StressConfig(inventory_skew=1.0, inventory_limit=3)
    assert quote_prices(mid=100, inventory=2, policy="symmetric", config=config) == (99, 101)
    assert quote_prices(mid=100, inventory=2, policy="inventory_aware", config=config) == (97, 100)
    assert quote_prices(mid=100, inventory=-2, policy="inventory_aware", config=config) == (
        100,
        103,
    )
    assert quote_prices(mid=100, inventory=3, policy="inventory_aware", config=config)[0] is None
    assert quote_prices(mid=100, inventory=-3, policy="inventory_aware", config=config)[1] is None


def test_adverse_flow_moves_only_future_mid_at_configured_lag() -> None:
    config = StressConfig(
        n_events=20,
        noise_probability=0,
        flow_probability=1,
        adverse_probability=1,
        adverse_lag=3,
        adverse_jump=2,
    )
    tape = generate_tape(config, seed=7)
    assert [event.mid for event in tape[:3]] == [config.initial_mid] * 3
    for index in range(3, len(tape)):
        assert tape[index].mid - tape[index - 1].mid == tape[index - 3].aggressor * 2


def test_seed_and_reused_tape_are_reproducible_and_not_mutated() -> None:
    config = StressConfig(n_events=100)
    tape = generate_tape(config, seed=42)
    assert tape == generate_tape(config, seed=42)
    assert tape != generate_tape(config, seed=43)
    first = simulate_execution(tape, policy="symmetric", config=config)
    simulate_execution(tape, policy="inventory_aware", config=config)
    second = simulate_execution(tape, policy="symmetric", config=config)
    pd.testing.assert_frame_equal(first.trace, second.trace)
    pd.testing.assert_frame_equal(first.fills, second.fills)


def test_no_fills_has_zero_pnl_and_missing_markouts() -> None:
    result = simulate_execution((MarketEvent(100),), policy="symmetric", config=StressConfig())
    assert result.metrics["net_pnl"] == 0
    assert result.metrics["fills"] == 0
    assert np.isnan(result.metrics["net_markout"])


def test_paired_ci_uses_seed_differences_and_preserves_negative_result() -> None:
    rows = []
    for seed, pnl in enumerate([100.0, 200.0, 300.0]):
        for policy, value in (("symmetric", pnl), ("inventory_aware", pnl - 2)):
            rows.append(
                {
                    "scenario": "unit",
                    "seed": seed,
                    "policy": policy,
                    **dict.fromkeys(
                        [
                            "net_pnl",
                            "mean_abs_inventory",
                            "max_drawdown",
                            "fills",
                            "net_markout",
                            "directional_markout",
                        ],
                        value,
                    ),
                }
            )
    result = paired_summary(pd.DataFrame(rows))
    assert (result["paired_difference"] == -2).all()
    assert (result["ci_95_low"] == -2).all()
    assert (result["ci_95_high"] == -2).all()
    assert (result["n_pairs"] == 3).all()
    with pytest.raises(ValueError, match="duplicate"):
        paired_summary(pd.DataFrame([*rows, rows[0]]))


def test_paired_summary_reports_sparse_markouts_without_dropping_pnl() -> None:
    config = StressConfig()
    rows = []
    for seed in range(2):
        for policy in ("symmetric", "inventory_aware"):
            result = simulate_execution((MarketEvent(100),), policy=policy, config=config)
            rows.append({"scenario": "empty", "seed": seed, "policy": policy, **result.metrics})
    # Infinite diagnostics are unavailable too, not valid paired observations.
    rows[0]["net_markout"] = float("inf")
    summary = paired_summary(pd.DataFrame(rows)).set_index("metric")
    assert summary.loc["net_pnl", "n_pairs"] == 2
    assert summary.loc["net_pnl", "ci_95_low"] == 0
    assert summary.loc["net_markout", "n_pairs"] == 0
    assert np.isnan(summary.loc["net_markout", "ci_95_low"])


def test_paired_summary_rejects_unmatched_seed_instead_of_silent_selection() -> None:
    rows = pd.DataFrame(
        [
            {"scenario": "unit", "seed": 0, "policy": "symmetric"},
            {"scenario": "unit", "seed": 0, "policy": "inventory_aware"},
            {"scenario": "unit", "seed": 1, "policy": "symmetric"},
        ]
    )
    with pytest.raises(ValueError, match="every scenario and seed"):
        paired_summary(rows)


@pytest.mark.parametrize(
    "updates",
    [
        {"activation_latency": -1},
        {"inventory_limit": 0},
        {"adverse_probability": 1.1},
        {"tick_value": float("nan")},
        {"fee_per_fill": -1},
    ],
)
def test_invalid_config_rejected(updates: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        StressConfig(**updates)  # type: ignore[arg-type]
