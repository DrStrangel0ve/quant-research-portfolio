"""Select a conservative value-search weight using paired duplicate poker."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

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
from quantlab.poker.value_search import NeuralStateValue, ValueGuidedResolver


def parse_args() -> argparse.Namespace:
    project = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=project / "artifacts" / "deep_cfr_checkpoint.pt",
    )
    parser.add_argument(
        "--value-checkpoint",
        type=Path,
        default=project / "artifacts" / "value" / "state_value_checkpoint.pt",
    )
    parser.add_argument("--weights", type=float, nargs="+", required=True)
    parser.add_argument("--belief-samples", type=int, default=128)
    parser.add_argument("--search-streets", type=int, nargs="+", default=[1])
    parser.add_argument("--pairs", type=int, default=500)
    parser.add_argument("--direct-pairs", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=15_900)
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "artifacts" / "value_search_validation.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = NeuralPolicy.from_checkpoint(args.baseline)
    value_function = NeuralStateValue.from_checkpoint(args.value_checkpoint)
    opponents: dict[str, MicroPolicy] = {
        "uniform_random": RandomMicroPolicy(),
        "calling_station": CallingStationMicroPolicy(),
        "pot_pressure": PressureMicroPolicy(),
    }
    results: list[dict[str, Any]] = []
    for weight_index, weight in enumerate(args.weights):
        resolver = ValueGuidedResolver(
            baseline,
            value_function,
            belief_samples=args.belief_samples,
            improvement_weight=weight,
            search_streets=tuple(args.search_streets),
            seed=args.seed + weight_index,
        )
        improvements = {
            name: asdict(
                paired_policy_improvement(
                    resolver,
                    baseline,
                    opponent,
                    duplicate_pairs=args.pairs,
                    rng=np.random.default_rng(
                        args.seed + weight_index * 100 + opponent_index
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
            resolver,
            baseline,
            duplicate_pairs=args.direct_pairs,
            rng=np.random.default_rng(args.seed + weight_index * 100 + 50),
        )
        results.append(
            {
                "improvement_weight": weight,
                "paired_improvement_vs_v1": improvements,
                "equal_weight_opponent_suite": {
                    "mean_improvement_big_blinds": suite_mean,
                    "standard_error": suite_error,
                    "ci95_low": suite_mean - 1.96 * suite_error,
                    "ci95_high": suite_mean + 1.96 * suite_error,
                },
                "direct_value_search_vs_v1": asdict(direct),
            }
        )
    payload = {
        "experiment": {
            "baseline": str(args.baseline.resolve()),
            "value_checkpoint": str(args.value_checkpoint.resolve()),
            "belief_samples": args.belief_samples,
            "search_streets": args.search_streets,
            "pairs_per_opponent": args.pairs,
            "direct_pairs": args.direct_pairs,
            "seed": args.seed,
            "selection_warning": (
                "Select one search weight here, then evaluate it once on a fresh seed."
            ),
        },
        "value_search_policies": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
