"""Run fixed synthetic execution scenarios and seed-paired comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quantlab.simulation.execution_stress import (  # noqa: E402
    Policy,
    StressConfig,
    generate_tape,
    paired_summary,
    simulate_execution,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=48)
    parser.add_argument("--events", type=int, default=1500)
    parser.add_argument("--first-seed", type=int, default=202600)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if args.seeds < 2 or args.events < 50:
        parser.error("use at least 2 seeds and 50 events")
    start = time.perf_counter()
    base = StressConfig(n_events=args.events)
    scenarios = {
        "baseline": base,
        "no_adverse_selection": replace(base, adverse_probability=0),
        "strong_adverse_selection": replace(base, adverse_probability=0.7),
        "slow_messages": replace(base, activation_latency=4, cancellation_latency=4),
        "long_queue": replace(base, queue_ahead=10),
        "high_fees": replace(base, fee_per_fill=0.005),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float | int]] = []
    policies: tuple[Policy, ...] = ("symmetric", "inventory_aware")
    tape_hashes: dict[str, list[str]] = {}
    for scenario, config in scenarios.items():
        tape_hashes[scenario] = []
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            tape = generate_tape(config, seed)
            tape_hashes[scenario].append(
                hashlib.sha256(
                    json.dumps([asdict(event) for event in tape], separators=(",", ":")).encode(),
                ).hexdigest()
            )
            for policy in policies:
                result = simulate_execution(tape, policy=policy, config=config)
                rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "policy": policy,
                        **result.metrics,
                    }
                )
                if scenario == "baseline" and seed == args.first_seed:
                    result.trace.to_csv(args.output / f"example_{policy}_trace.csv", index=False)
                    result.fills.to_csv(args.output / f"example_{policy}_fills.csv", index=False)
                    result.decisions.to_csv(
                        args.output / f"example_{policy}_decisions.csv",
                        index=False,
                    )
    runs = pd.DataFrame(rows)
    summary = paired_summary(runs)
    runs.to_csv(args.output / "seed_metrics.csv", index=False)
    summary.to_csv(args.output / "paired_summary.csv", index=False)
    metadata = {
        "description": "Synthetic model sensitivity; no real-market alpha claim.",
        "first_seed": args.first_seed,
        "seed_count": args.seeds,
        "events_per_seed": args.events,
        "configurations": {name: asdict(config) for name, config in scenarios.items()},
        "tape_sha256": tape_hashes,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "elapsed_seconds": time.perf_counter() - start,
        "ci_method": "Student-t on independent seed-level paired policy differences",
        "difference_direction": "inventory_aware minus symmetric",
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    for axis, metric, title in zip(
        axes,
        ["net_pnl", "mean_abs_inventory"],
        ["Net P&L difference ($ / seed)", "Mean |inventory| difference (shares)"],
        strict=True,
    ):
        values = summary[summary["metric"] == metric].set_index("scenario").loc[list(scenarios)]
        estimate = values["paired_difference"].to_numpy()
        axis.errorbar(
            estimate,
            np.arange(len(values)),
            fmt="o",
            color="#176b87",
            capsize=4,
            xerr=np.vstack(
                [
                    estimate - values["ci_95_low"].to_numpy(),
                    values["ci_95_high"].to_numpy() - estimate,
                ]
            ),
        )
        axis.axvline(0, color="#666666", linewidth=1)
        axis.set_yticks(np.arange(len(values)), values.index)
        axis.invert_yaxis()
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Synthetic execution stress | inventory-aware minus symmetric | paired 95% CIs")
    fig.savefig(args.output / "paired_comparison.png", dpi=160)
    plt.close(fig)
    print(summary[summary["metric"] == "net_pnl"].to_string(index=False))
    print(f"Saved results to {args.output}; runtime {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
