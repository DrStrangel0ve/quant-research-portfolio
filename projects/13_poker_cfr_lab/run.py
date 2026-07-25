"""Train and audit a CFR+ agent on heads-up Leduc Hold'em."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from quantlab.poker.agents import AggressiveBot, CallingStationBot, RandomBot
from quantlab.poker.cfr import CFRPlusTrainer
from quantlab.poker.evaluation import duplicate_match, expected_value, exploitability
from quantlab.poker.rlcard_adapter import load_rlcard_reference_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3_000)
    parser.add_argument("--duplicate-pairs", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=13_001)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations < 100:
        raise ValueError("iterations must be at least 100 for the convergence audit")
    args.output.mkdir(parents=True, exist_ok=True)

    checkpoints = sorted(
        {
            100,
            min(500, args.iterations),
            min(1_000, args.iterations),
            args.iterations,
        }
    )
    trainer = CFRPlusTrainer(seed=args.seed, information_mode="perfect_recall")
    convergence: list[dict[str, float | int]] = []
    completed = 0
    for checkpoint in checkpoints:
        policy = trainer.train(checkpoint - completed)
        completed = checkpoint
        convergence.append(
            {
                "iterations": checkpoint,
                "information_sets": len(policy.table),
                "exact_exploitability_bb_per_hand": exploitability(policy),
                "exact_value_vs_random_bb_per_hand": expected_value(policy, RandomBot()),
            }
        )

    policy = trainer.average_policy()
    trainer.save(args.output / "cfr_plus_checkpoint.json")
    with (args.output / "convergence.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(convergence[0]))
        writer.writeheader()
        writer.writerows(convergence)

    exact_matchups = {
        "uniform_random": expected_value(policy, RandomBot()),
        "calling_station": expected_value(policy, CallingStationBot()),
        "always_aggressive": expected_value(policy, AggressiveBot()),
    }
    summary: dict[str, Any] = {
        "experiment": {
            "game": "heads-up limit Leduc Hold'em (RLCard rules)",
            "algorithm": "chance-sampled CFR+ with linear averaging",
            "seed": args.seed,
            "iterations": args.iterations,
            "information_mode": "perfect_recall public history",
            "information_sets": len(policy.table),
        },
        "exact": {
            "self_play_value_bb_per_hand": expected_value(policy, policy),
            "exploitability_bb_per_hand": exploitability(policy),
            "matchups_as_player_zero_bb_per_hand": exact_matchups,
        },
        "convergence": convergence,
    }

    try:
        reference = load_rlcard_reference_policy()
    except ImportError:
        summary["rlcard_reference"] = {
            "status": "not installed; run `pip install -e .[poker]`",
        }
    else:
        paired = duplicate_match(
            policy,
            reference,
            duplicate_pairs=args.duplicate_pairs,
            rng=np.random.default_rng(args.seed + 1),
        )
        summary["rlcard_reference"] = {
            "status": "evaluated",
            "checkpoint": "RLCard 1.2.0 bundled leduc-holdem-cfr",
            "reference_information_sets": len(reference.table),
            "reference_exact_exploitability_bb_per_hand": exploitability(reference),
            "our_exact_value_as_player_zero_bb_per_hand": expected_value(
                policy,
                reference,
            ),
            "our_exact_value_as_player_one_bb_per_hand": -expected_value(
                reference,
                policy,
            ),
            "paired_duplicate_match": asdict(paired),
        }

    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
