"""Run the frozen BTC/ETH variance forecasting study on pinned public archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from data import load_hourly  # noqa: E402

from quantlab.research import volatility_study  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).parent
    parser.add_argument("--download", action="store_true", help="fetch missing pinned archives")
    parser.add_argument("--cache-dir", type=Path, default=project / "data_cache")
    parser.add_argument("--output", type=Path, default=project / "results")
    args = parser.parse_args()
    started = time.perf_counter()
    manifest = project / "manifest.json"
    hourly = load_hourly(manifest, args.cache_dir, download=args.download)
    result = volatility_study.run_study(hourly)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in {
        "predictions": result.predictions,
        "validation_scores": result.validation,
        "heldout_metrics": result.metrics,
        "paired_comparisons": result.comparisons,
    }.items():
        frame.to_csv(args.output / f"{name}.csv", index=False)
    metadata = {
        "description": "Retrospective fixed holdout; not preregistered or prospective.",
        "target": "Daily sum of 24 hourly squared log returns, grouped by UTC bar opening day",
        "split": {
            "warmup": "2023-12",
            "train": "2024",
            "validation": "2025",
            "test": "2026-01-01 through 2026-06-30 inclusive",
        },
        "selection": "minimum 2025 mean QLIKE per asset; fixed candidates; first minimum tie break",
        "candidates": list(volatility_study.LAMBDAS),
        "selected_lambdas": result.selected_lambdas,
        "daily_counts": result.daily_counts,
        "variance_floor": volatility_study.VARIANCE_FLOOR,
        "floor_usage": "positive forecast floor; QLIKE target floor only; raw target for MSE",
        "bootstrap": {
            "method": "paired non-circular moving-block percentile interval",
            "block_calendar_days": 7,
            "resamples": 2000,
            "seed": 20260905,
            "confidence": 0.95,
            "direction": "selected EWMA minus baseline",
            "scope": "conditional on fixed selected model; no multiplicity adjustment",
        },
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "study_source_sha256": hashlib.sha256(
            Path(volatility_study.__file__).read_bytes()
        ).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(len(hourly), 1, figsize=(11, 7), layout="constrained", squeeze=False)
    for axis, asset in zip(axes[:, 0], sorted(hourly), strict=True):
        frame = result.predictions.loc[
            (result.predictions["asset"] == asset) & (result.predictions["split"] == "test")
        ]
        axis.plot(
            frame["date"],
            np.sqrt(frame["realized_variance"]) * 100,
            label="Observed hourly-return proxy",
            color="#808080",
            alpha=0.6,
            linewidth=1,
        )
        axis.plot(
            frame["date"],
            np.sqrt(frame["rolling_30"]) * 100,
            label="Prior 30-day mean",
            color="#c87b25",
            linewidth=1.4,
        )
        axis.plot(
            frame["date"],
            np.sqrt(frame["selected_ewma"]) * 100,
            label=f"Validation-selected EWMA ({result.selected_lambdas[asset]:.2f})",
            color="#176b87",
            linewidth=1.4,
        )
        axis.set_title(asset)
        axis.set_ylabel("Daily volatility proxy (%)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="upper right")
    fig.suptitle("Held-out daily variance forecasting | January–June 2026 | one venue")
    fig.savefig(args.output / "heldout_volatility.png", dpi=160)
    plt.close(fig)
    print(result.validation.to_string(index=False))
    print(result.metrics.to_string(index=False))
    print(result.comparisons.to_string(index=False))
    print(json.dumps(result.daily_counts, indent=2))
    print(f"Saved results to {args.output}; runtime {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
