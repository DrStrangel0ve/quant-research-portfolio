"""Render every benchmark repeat; no filtering of slow runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument(
        "--context", default="Synthetic order-event pipeline · supplied measurements"
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = frame.groupby(["queue", "mode", "burst"])[
        ["events_per_second", "scheduled_p99_ns", "attempt_p99_ns", "full_retries"]
    ].agg(["min", "median", "max"])
    summary.to_csv(args.output / "summary.csv")
    print(summary.to_string())
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    colors = {"spsc": "#147d92", "mutex": "#b84d3f"}
    for index, queue in enumerate(("spsc", "mutex")):
        values = frame[(frame["mode"] == "throughput") & (frame["queue"] == queue)]
        x = index + np.linspace(-0.1, 0.1, len(values))
        axes[0].scatter(x, values["events_per_second"] / 1e6, color=colors[queue], s=40)
        axes[0].hlines(values["events_per_second"].median() / 1e6,
                       index - 0.2, index + 0.2, color=colors[queue])
    axes[0].set(xticks=[0, 1], xticklabels=["SPSC", "Mutex"], ylabel="Million events / second",
                title="Uninstrumented throughput")
    labels = []
    for index, (burst, queue) in enumerate(((32, "spsc"), (32, "mutex"),
                                          (256, "spsc"), (256, "mutex"))):
        values = frame[(frame["burst"] == burst) & (frame["queue"] == queue)]
        axes[1].scatter(index + np.linspace(-0.1, 0.1, len(values)),
                        values["scheduled_p99_ns"] / 1e6, color=colors[queue], s=40)
        labels.append(f"{queue.upper()}\n{burst} / 100 µs")
    axes[1].set(xticks=range(4), xticklabels=labels, yscale="log",
                ylabel="Scheduled release to completion p99 (ms; log scale)",
                title="Burst latency: all five repeats")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    figure.suptitle(args.context)
    figure.savefig(args.output / "benchmark.png", dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
