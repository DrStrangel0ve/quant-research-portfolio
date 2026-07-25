"""Continue a Deep CFR checkpoint and gate promotion with paired evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from quantlab.poker.deep_cfr import (
    CallingStationMicroPolicy,
    DeepCFRTrainer,
    MicroPolicy,
    NeuralPolicy,
    PressureMicroPolicy,
    RandomMicroPolicy,
)
from quantlab.poker.micro_evaluation import (
    duplicate_micro_match,
    paired_policy_improvement,
)
from quantlab.poker.resolver import BeliefRolloutResolver


def parse_args() -> argparse.Namespace:
    project = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project / "artifacts" / "deep_cfr_checkpoint.pt",
    )
    parser.add_argument("--additional-iterations", type=int, default=2_400)
    parser.add_argument("--traversals", type=int, default=8)
    parser.add_argument("--advantage-steps", type=int, default=60)
    parser.add_argument("--strategy-steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--evaluation-pairs", type=int, default=3_000)
    parser.add_argument("--direct-pairs", type=int, default=5_000)
    parser.add_argument("--resolver-pairs", type=int, default=100)
    parser.add_argument("--resolver-rollouts", type=int, default=64)
    parser.add_argument("--anchor-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=15_002)
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "artifacts" / "v3",
    )
    parser.add_argument(
        "--promote-browser",
        type=Path,
        default=None,
        help="Write the candidate to this browser path only if the gate passes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = NeuralPolicy.from_checkpoint(args.checkpoint)
    trainer = DeepCFRTrainer.from_checkpoint(
        args.checkpoint,
        iterations=args.additional_iterations,
        traversals_per_player=args.traversals,
        advantage_steps=args.advantage_steps,
        strategy_steps=args.strategy_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    starting_iteration = trainer.iteration
    starting_snapshot_count = len(trainer.snapshots)
    anchor_started = time.perf_counter()
    trainer.seed_strategy_memory(baseline, samples=args.anchor_samples)
    anchor_seconds = time.perf_counter() - anchor_started
    started = time.perf_counter()
    candidate = trainer.train()
    training_seconds = time.perf_counter() - started
    trainer.save_checkpoint(args.output / "deep_cfr_checkpoint.pt")
    trainer.export_strategy_json(args.output / "micro_strategy.json")

    new_diagnostics = [
        asdict(snapshot) for snapshot in trainer.snapshots[starting_snapshot_count:]
    ]
    _write_csv(args.output / "continuation_diagnostics.csv", new_diagnostics)

    opponents: dict[str, MicroPolicy] = {
        "uniform_random": RandomMicroPolicy(),
        "calling_station": CallingStationMicroPolicy(),
        "pot_pressure": PressureMicroPolicy(),
    }
    improvements = {
        name: asdict(
            paired_policy_improvement(
                candidate,
                baseline,
                opponent,
                duplicate_pairs=args.evaluation_pairs,
                rng=np.random.default_rng(args.seed + offset),
            )
        )
        for offset, (name, opponent) in enumerate(opponents.items(), start=1)
    }
    direct = duplicate_micro_match(
        candidate,
        baseline,
        duplicate_pairs=args.direct_pairs,
        rng=np.random.default_rng(args.seed + 10),
    )
    resolver = BeliefRolloutResolver(
        candidate,
        rollouts_per_action=args.resolver_rollouts,
        seed=args.seed + 20,
    )
    resolver_match = duplicate_micro_match(
        resolver,
        candidate,
        duplicate_pairs=args.resolver_pairs,
        rng=np.random.default_rng(args.seed + 21),
    )

    deltas = [float(result["mean_improvement_big_blinds"]) for result in improvements.values()]
    errors = [float(result["standard_error"]) for result in improvements.values()]
    suite_mean = float(np.mean(deltas))
    suite_standard_error = math.sqrt(sum(error**2 for error in errors)) / len(errors)
    suite_half_width = 1.96 * suite_standard_error
    suite_ci = (suite_mean - suite_half_width, suite_mean + suite_half_width)
    no_significant_regression = all(
        float(result["ci95_high"]) >= 0.0 for result in improvements.values()
    )
    direct_not_materially_worse = direct.ci95_low >= -0.10
    promotion_passed = (
        suite_ci[0] > 0.0
        and no_significant_regression
        and direct_not_materially_worse
    )
    if promotion_passed and args.promote_browser is not None:
        trainer.export_strategy_json(args.promote_browser)

    summary: dict[str, Any] = {
        "experiment": {
            "algorithm": "policy-replay-anchored resumed external-sampling neural CFR",
            "source_checkpoint": str(args.checkpoint),
            "starting_iteration": starting_iteration,
            "completed_iteration": trainer.iteration,
            "additional_iterations": args.additional_iterations,
            "training_seconds": training_seconds,
            "anchor_samples": args.anchor_samples,
            "anchor_seconds": anchor_seconds,
            "device": str(trainer.device),
            "seed": args.seed,
            "config": asdict(trainer.config),
        },
        "paired_improvement_vs_v1": improvements,
        "equal_weight_opponent_suite": {
            "mean_improvement_big_blinds": suite_mean,
            "standard_error": suite_standard_error,
            "ci95_low": suite_ci[0],
            "ci95_high": suite_ci[1],
        },
        "direct_candidate_vs_v1": asdict(direct),
        "belief_rollout_resolver_vs_candidate": {
            "rollouts_per_action": args.resolver_rollouts,
            "common_random_numbers": True,
            "stratified_range_sampling": True,
            "match": asdict(resolver_match),
        },
        "promotion_gate": {
            "passed": promotion_passed,
            "requirements": {
                "opponent_suite_ci95_low_above_zero": suite_ci[0] > 0.0,
                "no_fixed_opponent_significant_regression": no_significant_regression,
                "direct_v1_ci95_low_at_least_minus_0_10": direct_not_materially_worse,
            },
            "browser_exported": promotion_passed and args.promote_browser is not None,
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("continuation produced no diagnostic snapshots")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
