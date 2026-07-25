"""Train and audit an information-set value model for Royal Micro Hold'em."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from quantlab.poker.deep_cfr import NeuralPolicy
from quantlab.poker.value_search import (
    ValueTrainingConfig,
    save_state_value_checkpoint,
    train_state_value,
)


def parse_args() -> argparse.Namespace:
    project = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=project / "artifacts" / "deep_cfr_checkpoint.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "artifacts" / "value",
    )
    parser.add_argument("--examples", type=int, default=60_000)
    parser.add_argument("--rollout-repeats", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=15_800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ValueTrainingConfig(
        examples=args.examples,
        rollout_repeats=args.rollout_repeats,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        seed=args.seed,
    )
    blueprint = NeuralPolicy.from_checkpoint(args.blueprint)
    started = time.perf_counter()
    result = train_state_value(blueprint, config)
    elapsed = time.perf_counter() - started
    result.metrics["training_seconds"] = elapsed
    result.metrics["blueprint_checkpoint"] = str(args.blueprint.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    save_state_value_checkpoint(
        args.output / "state_value_checkpoint.pt",
        result,
        config,
    )
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(
        json.dumps(result.metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result.metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
