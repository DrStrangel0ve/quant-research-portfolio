"""Evaluate v1/v3 behavioral mixtures on a fixed duplicate-poker suite."""

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
    MixturePolicy,
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
        "--candidate",
        type=Path,
        default=project / "artifacts" / "v3" / "deep_cfr_checkpoint.pt",
    )
    parser.add_argument("--candidate-weights", type=float, nargs="+", required=True)
    parser.add_argument("--pairs", type=int, default=1_000)
    parser.add_argument("--direct-pairs", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=15_500)
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "artifacts" / "mixture_validation.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = NeuralPolicy.from_checkpoint(args.baseline)
    candidate = NeuralPolicy.from_checkpoint(args.candidate)
    opponents: dict[str, MicroPolicy] = {
        "uniform_random": RandomMicroPolicy(),
        "calling_station": CallingStationMicroPolicy(),
        "pot_pressure": PressureMicroPolicy(),
    }
    results: list[dict[str, Any]] = []
    for weight_index, candidate_weight in enumerate(args.candidate_weights):
        if not 0.0 <= candidate_weight <= 1.0:
            raise ValueError("candidate weights must lie in [0, 1]")
        mixture = MixturePolicy(
            (baseline, candidate),
            (1.0 - candidate_weight, candidate_weight),
        )
        improvements = {
            name: asdict(
                paired_policy_improvement(
                    mixture,
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
            mixture,
            baseline,
            duplicate_pairs=args.direct_pairs,
            rng=np.random.default_rng(args.seed + weight_index * 100 + 50),
        )
        results.append(
            {
                "candidate_weight": candidate_weight,
                "baseline_weight": 1.0 - candidate_weight,
                "paired_improvement_vs_v1": improvements,
                "equal_weight_opponent_suite": {
                    "mean_improvement_big_blinds": suite_mean,
                    "standard_error": suite_error,
                    "ci95_low": suite_mean - 1.96 * suite_error,
                    "ci95_high": suite_mean + 1.96 * suite_error,
                },
                "direct_mixture_vs_v1": asdict(direct),
            }
        )
    payload = {
        "experiment": {
            "baseline": str(args.baseline),
            "candidate": str(args.candidate),
            "pairs_per_opponent": args.pairs,
            "direct_pairs": args.direct_pairs,
            "seed": args.seed,
            "selection_warning": (
                "Use this grid only for model selection; rerun the chosen weight "
                "on a fresh seed and larger sample."
            ),
        },
        "mixtures": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
