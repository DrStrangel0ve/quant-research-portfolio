"""Causal, synthetic execution stress tests; this is not an exchange backtester."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

Policy = Literal["symmetric", "inventory_aware"]


@dataclass(frozen=True)
class StressConfig:
    n_events: int = 1_500
    initial_mid: int = 10_000
    tick_value: float = 0.01
    half_spread: int = 1
    queue_ahead: int = 3
    depth_queue: int = 2
    activation_latency: int = 1
    cancellation_latency: int = 1
    quote_interval: int = 3
    inventory_limit: int = 8
    inventory_skew: float = 0.5
    fee_per_fill: float = 0.001
    liquidation_half_spread: int = 1
    markout_horizon: int = 10
    flow_probability: float = 0.8
    noise_probability: float = 0.15
    adverse_probability: float = 0.3
    adverse_lag: int = 2
    adverse_jump: int = 2

    def __post_init__(self) -> None:
        positive = (
            self.n_events,
            self.initial_mid,
            self.half_spread,
            self.quote_interval,
            self.inventory_limit,
            self.markout_horizon,
            self.adverse_lag,
        )
        nonnegative = (
            self.queue_ahead,
            self.depth_queue,
            self.activation_latency,
            self.cancellation_latency,
            self.liquidation_half_spread,
            self.adverse_jump,
        )
        if any(value <= 0 for value in positive) or any(value < 0 for value in nonnegative):
            raise ValueError("invalid count, price, latency, or inventory bound")
        if not np.isfinite(self.tick_value) or self.tick_value <= 0:
            raise ValueError("tick_value must be finite and positive")
        if any(
            not np.isfinite(value) or value < 0
            for value in (
                self.inventory_skew,
                self.fee_per_fill,
            )
        ):
            raise ValueError("skew and fees must be finite and nonnegative")
        if any(
            not 0 <= value <= 1
            for value in (
                self.flow_probability,
                self.noise_probability,
                self.adverse_probability,
            )
        ):
            raise ValueError("probabilities must be in [0, 1]")


@dataclass(frozen=True)
class MarketEvent:
    """Public reference mid followed by one initially unobserved aggressive order.

    aggressor=+1 buys from our ask; -1 sells to our bid. sweep is the number
    of levels beyond the external touch the aggressive order can reach.
    """

    mid: int
    aggressor: int = 0
    volume: int = 0
    sweep: int = 0

    def __post_init__(self) -> None:
        if self.mid <= 0 or self.aggressor not in (-1, 0, 1):
            raise ValueError("mid must be positive and aggressor must be -1, 0, or 1")
        if self.volume < 0 or self.sweep < 0 or (self.aggressor == 0 and self.volume != 0):
            raise ValueError("invalid aggressive order volume or sweep")


@dataclass
class RestingOrder:
    side: int  # +1 bid / inventory purchase, -1 ask / inventory sale
    price: int
    queue_ahead: int

    def consume(self, volume: int) -> bool:
        """Consume FIFO queue; a unit quote fills only after all ahead volume."""
        if volume < 0:
            raise ValueError("volume cannot be negative")
        consumed = min(self.queue_ahead, volume)
        self.queue_ahead -= consumed
        return volume - consumed >= 1


@dataclass
class _SideState:
    order: RestingOrder | None = None
    cancel_due: int | None = None
    activate_due: int | None = None
    target: int | None = None


@dataclass(frozen=True)
class StressResult:
    trace: pd.DataFrame
    fills: pd.DataFrame
    decisions: pd.DataFrame
    metrics: dict[str, float]


def generate_tape(config: StressConfig, seed: int) -> tuple[MarketEvent, ...]:
    """Generate all exogenous randomness independently of any quoting policy.

    A latent subset of aggressive orders induces a same-direction mid jump
    adverse_lag events later. The policy sees neither the subset nor future mids.
    """
    rng = np.random.default_rng(seed)
    n = config.n_events
    sides = rng.choice([-1, 1], size=n)
    active = rng.random(n) < config.flow_probability
    informed = rng.random(n) < config.adverse_probability
    volumes = np.minimum(rng.geometric(0.3, size=n), 12)
    sweeps = rng.choice([0, 1, 2], size=n, p=[0.8, 0.15, 0.05])
    noise = rng.choice([-1, 1], size=n) * (rng.random(n) < config.noise_probability)
    mid = config.initial_mid
    tape: list[MarketEvent] = []
    for index in range(n):
        if index:
            mid += int(noise[index])
        prior = index - config.adverse_lag
        if prior >= 0 and active[prior] and informed[prior]:
            mid += int(sides[prior]) * config.adverse_jump
        if mid <= 0:
            raise ValueError("synthetic price reached zero; increase initial_mid")
        side = int(sides[index]) if active[index] else 0
        tape.append(
            MarketEvent(
                mid=mid,
                aggressor=side,
                volume=int(volumes[index]) if side else 0,
                sweep=int(sweeps[index]),
            )
        )
    return tuple(tape)


def quote_prices(
    *,
    mid: int,
    inventory: int,
    policy: Policy,
    config: StressConfig,
) -> tuple[int | None, int | None]:
    """Use only the current observed mid and inventory, with no tape access."""
    if policy not in ("symmetric", "inventory_aware"):
        raise ValueError("unknown policy")
    shift = int(np.rint(config.inventory_skew * inventory)) if policy == "inventory_aware" else 0
    # A reducing quote may improve the touch as far as mid, but cannot cross it.
    bid = min(mid, mid - config.half_spread - shift)
    ask = max(mid, mid + config.half_spread - shift)
    return (
        bid if inventory < config.inventory_limit else None,
        ask if inventory > -config.inventory_limit else None,
    )


def _deliver(
    state: _SideState,
    *,
    opposite: RestingOrder | None,
    side: int,
    now: int,
    mid: int,
    inventory: int,
    config: StressConfig,
) -> bool:
    """Deliver one activation after cancels; return rejected activation."""
    rejected = False
    if state.activate_due is not None and state.activate_due <= now:
        target = state.target
        if target is None:
            raise RuntimeError("activation without target")
        # Exchange/gateway checks at arrival, not against the submission-time mid.
        crosses = (
            target >= mid + config.half_spread
            if side == 1
            else (target <= mid - config.half_spread)
        )
        crosses_own = opposite is not None and (
            target >= opposite.price if side == 1 else target <= opposite.price
        )
        exceeds_limit = side * inventory >= config.inventory_limit
        if crosses or crosses_own or exceeds_limit or target <= 0:
            rejected = True
        else:
            distance = side * (mid - target)
            ahead = (
                0
                if distance < config.half_spread
                else (config.queue_ahead + config.depth_queue * (distance - config.half_spread))
            )
            state.order = RestingOrder(side, target, ahead)
        state.activate_due = None
        state.target = None
    return rejected


def _deliver_messages(
    states: dict[int, _SideState],
    *,
    now: int,
    mid: int,
    inventory: int,
    config: StressConfig,
) -> int:
    # Global same-time ordering: all cancels first, then bid activation, then ask.
    for state in states.values():
        if state.cancel_due is not None and state.cancel_due <= now:
            state.order = None
            state.cancel_due = None
    return sum(
        int(
            _deliver(
                state,
                opposite=states[-side].order,
                side=side,
                now=now,
                mid=mid,
                inventory=inventory,
                config=config,
            )
        )
        for side, state in states.items()
    )


def _request(state: _SideState, *, target: int | None, now: int, config: StressConfig) -> None:
    # Serialize requests per side; never overlap a resting and replacement order.
    if state.cancel_due is not None or state.activate_due is not None:
        return
    if state.order is not None and state.order.price == target:
        return
    if state.order is None and target is None:
        return
    cancel_delay = config.cancellation_latency if state.order is not None else 0
    if state.order is not None:
        state.cancel_due = now + cancel_delay
    if target is not None:
        state.target = target
        state.activate_due = now + cancel_delay + config.activation_latency


def simulate_execution(
    tape: tuple[MarketEvent, ...],
    *,
    policy: Policy,
    config: StressConfig,
) -> StressResult:
    """Run a unit-size quote pair against a common immutable event tape.

    Event order: observe mid; deliver older messages; make scheduled decision;
    deliver zero-latency messages; process hidden aggressive flow; record equity.
    Future tape elements are read only in a separate post-run markout pass.
    """
    if not tape:
        raise ValueError("tape cannot be empty")
    if policy not in ("symmetric", "inventory_aware"):
        raise ValueError("unknown policy")
    states = {1: _SideState(), -1: _SideState()}
    inventory = 0
    cash = 0.0
    rejected = 0
    trace: list[dict[str, float | int]] = []
    fills: list[dict[str, float | int]] = []
    decisions: list[dict[str, float | int | None]] = []
    for now, event in enumerate(tape):
        rejected += _deliver_messages(
            states,
            now=now,
            mid=event.mid,
            inventory=inventory,
            config=config,
        )
        if now % config.quote_interval == 0:
            bid, ask = quote_prices(
                mid=event.mid, inventory=inventory, policy=policy, config=config
            )
            decisions.append(
                {
                    "event": now,
                    "mid": event.mid,
                    "inventory": inventory,
                    "desired_bid": bid,
                    "desired_ask": ask,
                }
            )
            for side, target in ((1, bid), (-1, ask)):
                _request(states[side], target=target, now=now, config=config)
            rejected += _deliver_messages(
                states,
                now=now,
                mid=event.mid,
                inventory=inventory,
                config=config,
            )
        if event.aggressor:
            side = -event.aggressor
            state = states[side]
            order = state.order
            limit = event.mid + event.aggressor * (config.half_spread + event.sweep)
            reachable = order is not None and (
                order.price >= limit if side == 1 else order.price <= limit
            )
            if order is not None and reachable and order.consume(event.volume):
                inventory += side
                cash -= side * order.price * config.tick_value + config.fee_per_fill
                fills.append(
                    {
                        "event": now,
                        "side": side,
                        "price_ticks": order.price,
                        "mid_ticks": event.mid,
                        "inventory_after": inventory,
                        "fee": config.fee_per_fill,
                    }
                )
                state.order = None
                if abs(inventory) > config.inventory_limit:
                    raise RuntimeError("inventory reservation invariant violated")
        trace.append(
            {
                "event": now,
                "mid_ticks": event.mid,
                "inventory": inventory,
                "cash": cash,
                "equity": cash + inventory * event.mid * config.tick_value,
                "bid_price": states[1].order.price if states[1].order else np.nan,
                "ask_price": states[-1].order.price if states[-1].order else np.nan,
            }
        )

    # Diagnostics are deliberately separated from all execution and decision code.
    for fill in fills:
        event_index = int(fill["event"])
        later = event_index + config.markout_horizon
        fill["net_markout"] = (
            int(fill["side"]) * (tape[later].mid - fill["price_ticks"]) * config.tick_value
            - config.fee_per_fill
            if later < len(tape)
            else np.nan
        )
        fill["directional_markout"] = (
            int(fill["side"]) * (tape[later].mid - fill["mid_ticks"]) * config.tick_value
            if later < len(tape)
            else np.nan
        )
    trace_frame = pd.DataFrame(trace)
    fill_frame = pd.DataFrame(
        fills,
        columns=[
            "event",
            "side",
            "price_ticks",
            "mid_ticks",
            "inventory_after",
            "fee",
            "net_markout",
            "directional_markout",
        ],
    )
    mtm = cash + inventory * tape[-1].mid * config.tick_value
    liquidation_cost = abs(inventory) * (
        config.liquidation_half_spread * config.tick_value + config.fee_per_fill
    )
    pnl = mtm - liquidation_cost
    equity = np.array([0.0, *trace_frame["equity"].tolist(), pnl], dtype=float)
    net_markout = fill_frame["net_markout"].dropna()
    directional = fill_frame["directional_markout"].dropna()
    metrics = {
        "net_pnl": pnl,
        "terminal_mtm": mtm,
        "liquidation_cost": liquidation_cost,
        "terminal_cash": cash,
        "terminal_inventory": float(inventory),
        "fills": float(len(fills)),
        "fees": len(fills) * config.fee_per_fill,
        "mean_abs_inventory": float(trace_frame["inventory"].abs().mean()),
        "max_abs_inventory": float(trace_frame["inventory"].abs().max()),
        "rms_inventory": float(np.sqrt(np.mean(trace_frame["inventory"].to_numpy() ** 2))),
        "max_drawdown": float(np.max(np.maximum.accumulate(equity) - equity)),
        "net_markout": float(net_markout.mean()) if len(net_markout) else float("nan"),
        "directional_markout": float(directional.mean()) if len(directional) else float("nan"),
        "mature_markouts": float(len(net_markout)),
        "rejected_activations": float(rejected),
    }
    return StressResult(trace_frame, fill_frame, pd.DataFrame(decisions), metrics)


def paired_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Student-t CIs on seed-level paired differences, not individual fills."""
    required = {"scenario", "seed", "policy"}
    if not required.issubset(runs.columns):
        raise ValueError("runs require scenario, seed, and policy columns")
    if runs.duplicated(["scenario", "seed", "policy"]).any():
        raise ValueError("duplicate scenario/seed/policy")
    metrics = [
        "net_pnl",
        "mean_abs_inventory",
        "max_drawdown",
        "fills",
        "net_markout",
        "directional_markout",
    ]
    rows: list[dict[str, str | float | int]] = []
    for scenario, group in runs.groupby("scenario", sort=True):
        presence = group.pivot(index="seed", columns="policy", values="scenario")
        if set(presence.columns) != {"symmetric", "inventory_aware"} or presence.isna().any().any():
            raise ValueError("both policies required for every scenario and seed")
        for metric in metrics:
            wide = group.pivot(index="seed", columns="policy", values=metric)
            # Missing finite diagnostics are omitted pairwise and reported in n_pairs.
            wide = (
                wide[["symmetric", "inventory_aware"]].replace([np.inf, -np.inf], np.nan).dropna()
            )
            difference = wide["inventory_aware"] - wide["symmetric"]
            n = len(difference)
            mean = float(difference.mean()) if n else float("nan")
            half_width = (
                float(student_t.ppf(0.975, df=n - 1)) * float(difference.std(ddof=1)) / np.sqrt(n)
                if n >= 2
                else float("nan")
            )
            rows.append(
                {
                    "scenario": str(scenario),
                    "metric": metric,
                    "n_pairs": n,
                    "symmetric_mean": float(wide["symmetric"].mean()),
                    "inventory_aware_mean": float(wide["inventory_aware"].mean()),
                    "paired_difference": mean,
                    "ci_95_low": mean - half_width,
                    "ci_95_high": mean + half_width,
                }
            )
    return pd.DataFrame(rows)
