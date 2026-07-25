"""Validate a pressure-adaptive v1/v3 meta-policy on duplicate poker."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from quantlab.poker.adaptive import PressureAdaptivePolicy
from quantlab.poker.deep_cfr import (
    CallingStationMicroPolicy,
    MicroPolicy,
    NeuralPolicy,
    PressureMicroPolicy,
    RandomMicroPolicy,
)
from quantlab.poker.micro_evaluation import (
    duplicate_micro_match,
    paired_policy_improvement,
)


def parse_args() -> argparse.Namespace:
    project = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=project / "artifacts" / "deep_cfr_checkpoint.pt",
    )
    parser.add_argument(
        "--specialist",
        type=Path,
        default=project / "artifacts" / "v3" / "deep_cfr_checkpoint.pt",
    )
    parser.add_argument("--thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--pairs", type=int, default=2_000)
    parser.add_argument("--direct-pairs", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=15_600)
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "artifacts" / "adaptive_validation.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = NeuralPolicy.from_checkpoint(args.baseline)
    specialist = NeuralPolicy.from_checkpoint(args.specialist)
    opponents: dict[str, MicroPolicy] = {
        "uniform_random": RandomMicroPolicy(),
        "calling_station": CallingStationMicroPolicy(),
        "pot_pressure": PressureMicroPolicy(),
    }
    results: list[dict[str, Any]] = []
    for threshold_index, threshold in enumerate(args.thresholds):
        adaptive = PressureAdaptivePolicy(
            baseline,
            specialist,
            threshold=threshold,
        )
        improvements = {
            name: asdict(
                paired_policy_improvement(
                    adaptive,
                    baseline,
                    opponent,
                    duplicate_pairs=args.pairs,
                    rng=np.random.default_rng(
                        args.seed + threshold_index * 100 + opponent_index
                    ),
                )
            )
            for opponent_index, (name, opponent) in enumerate(
                opponents.items(),
                start=1,
            )
        }
        deltas = [
            float(result["mean_improvement_big_blinds"])
            for result in improvements.values()
        ]
        errors = [float(result["standard_error"]) for result in improvements.values()]
        suite_mean = float(np.mean(deltas))
        suite_error = math.sqrt(sum(error**2 for error in errors)) / len(errors)
        direct = duplicate_micro_match(
            adaptive,
            baseline,
            duplicate_pairs=args.direct_pairs,
            rng=np.random.default_rng(args.seed + threshold_index * 100 + 50),
        )
        results.append(
            {
                "threshold": threshold,
                "paired_improvement_vs_v1": improvements,
                "equal_weight_opponent_suite": {
                    "mean_improvement_big_blinds": suite_mean,
                    "standard_error": suite_error,
                    "ci95_low": suite_mean - 1.96 * suite_error,
                    "ci95_high": suite_mean + 1.96 * suite_error,
                },
                "direct_adaptive_vs_v1": asdict(direct),
            }
        )
    payload = {
        "experiment": {
            "baseline": str(args.baseline),
            "pressure_specialist": str(args.specialist),
            "pairs_per_opponent": args.pairs,
            "direct_pairs": args.direct_pairs,
            "seed": args.seed,
            "information_boundary": (
                "adaptation uses only public opponent actions and legal-action sets"
            ),
            "selection_warning": (
                "Select a threshold here, then evaluate it once on a fresh seed."
            ),
        },
        "adaptive_policies": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
