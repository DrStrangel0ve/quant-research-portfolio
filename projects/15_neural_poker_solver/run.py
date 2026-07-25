"""Train and audit the Royal Micro Hold'em neural CFR agent."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from quantlab.poker.deep_cfr import (
    CallingStationMicroPolicy,
    DeepCFRTrainer,
    PressureMicroPolicy,
    RandomMicroPolicy,
    TrainingConfig,
)
from quantlab.poker.micro_evaluation import duplicate_micro_match
from quantlab.poker.resolver import BeliefRolloutResolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1_600)
    parser.add_argument("--traversals", type=int, default=6)
    parser.add_argument("--train-every", type=int, default=20)
    parser.add_argument("--advantage-steps", type=int, default=50)
    parser.add_argument("--strategy-steps", type=int, default=500)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--duplicate-pairs", type=int, default=2_000)
    parser.add_argument("--resolver-pairs", type=int, default=50)
    parser.add_argument("--resolver-rollouts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=15_001)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results",
    )
    parser.add_argument(
        "--browser-output",
        type=Path,
        default=None,
        help="Optional second JSON export consumed by project 14.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = TrainingConfig(
        iterations=args.iterations,
        traversals_per_player=args.traversals,
        train_every=args.train_every,
        advantage_steps=args.advantage_steps,
        strategy_steps=args.strategy_steps,
        hidden_size=args.hidden_size,
        seed=args.seed,
    )
    trainer = DeepCFRTrainer(config)
    started = time.perf_counter()
    policy = trainer.train()
    training_seconds = time.perf_counter() - started

    checkpoint = args.output / "deep_cfr_checkpoint.pt"
    browser_model = args.output / "micro_strategy.json"
    trainer.save_checkpoint(checkpoint)
    trainer.export_strategy_json(browser_model)
    if args.browser_output is not None:
        trainer.export_strategy_json(args.browser_output)

    diagnostics = [asdict(snapshot) for snapshot in trainer.snapshots]
    _write_csv(args.output / "training_diagnostics.csv", diagnostics)
    _plot_training(args.output / "training_diagnostics.png", diagnostics)

    opponents = {
        "uniform_random": RandomMicroPolicy(),
        "calling_station": CallingStationMicroPolicy(),
        "pot_pressure": PressureMicroPolicy(),
    }
    matchups = {
        name: asdict(
            duplicate_micro_match(
                policy,
                opponent,
                duplicate_pairs=args.duplicate_pairs,
                rng=np.random.default_rng(args.seed + offset),
            )
        )
        for offset, (name, opponent) in enumerate(opponents.items(), start=1)
    }
    self_play = duplicate_micro_match(
        policy,
        policy,
        duplicate_pairs=args.duplicate_pairs,
        rng=np.random.default_rng(args.seed + 10),
    )

    resolver = BeliefRolloutResolver(
        policy,
        rollouts_per_action=args.resolver_rollouts,
        seed=args.seed + 20,
    )
    resolver_match = duplicate_micro_match(
        resolver,
        policy,
        duplicate_pairs=args.resolver_pairs,
        rng=np.random.default_rng(args.seed + 21),
    )
    summary: dict[str, Any] = {
        "experiment": {
            "game": "Royal Micro Hold'em: 24 cards, 20-chip stacks, two streets",
            "algorithm": "external-sampling neural CFR with reservoir replay",
            "seed": args.seed,
            "training_seconds": training_seconds,
            "device": str(trainer.device),
            "config": asdict(config),
        },
        "training": {
            "advantage_samples": [
                len(trainer.advantage_memories[0]),
                len(trainer.advantage_memories[1]),
            ],
            "strategy_samples": len(trainer.strategy_memory),
            "final_diagnostic": diagnostics[-1],
        },
        "duplicate_evaluation": {
            "pairs_per_baseline": args.duplicate_pairs,
            "neural_blueprint_matchups": matchups,
            "self_play_control": asdict(self_play),
        },
        "belief_rollout_resolver": {
            "description": (
                "Bayes-filtered opponent range plus one-step action search "
                "with blueprint rollouts; not a safe subgame solver"
            ),
            "rollouts_per_action": args.resolver_rollouts,
            "match_vs_blueprint": asdict(resolver_match),
        },
        "audit_scope": {
            "exact_exploitability": (
                "not tractable for this larger game; Project 13 remains the exact oracle"
            ),
            "reported_units": "big blinds per hand",
            "confidence_intervals": "paired normal 95% intervals over duplicate deals",
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("training produced no diagnostic snapshots")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_training(path: Path, rows: list[dict[str, Any]]) -> None:
    iterations = [int(row["iteration"]) for row in rows]
    player_zero = [float(row["advantage_loss_player_zero"]) for row in rows]
    player_one = [float(row["advantage_loss_player_one"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(iterations, player_zero, label="Player 0 advantage MSE")
    axis.plot(iterations, player_one, label="Player 1 advantage MSE")
    axis.set(
        xlabel="External-sampling iteration",
        ylabel="Replay-weighted loss",
        title="Deep CFR training diagnostics",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
