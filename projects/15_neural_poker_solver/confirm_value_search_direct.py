"""Run a high-power direct safety check for the selected value-search policy."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from quantlab.poker.deep_cfr import NeuralPolicy
from quantlab.poker.micro_evaluation import duplicate_micro_match
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
        default=(
            project
            / "artifacts"
            / "value_v3_equity"
            / "state_value_checkpoint.pt"
        ),
    )
    parser.add_argument("--weight", type=float, default=0.15)
    parser.add_argument("--belief-samples", type=int, default=32)
    parser.add_argument("--direct-pairs", type=int, default=8_000)
    parser.add_argument("--seed", type=int, default=15_930)
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "artifacts" / "value_search_direct_final.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = NeuralPolicy.from_checkpoint(args.baseline)
    value_function = NeuralStateValue.from_checkpoint(args.value_checkpoint)
    candidate = ValueGuidedResolver(
        baseline,
        value_function,
        belief_samples=args.belief_samples,
        improvement_weight=args.weight,
        search_streets=(1,),
        seed=args.seed,
    )
    direct = duplicate_micro_match(
        candidate,
        baseline,
        duplicate_pairs=args.direct_pairs,
        rng=np.random.default_rng(args.seed),
    )
    payload = {
        "experiment": {
            "baseline": str(args.baseline.resolve()),
            "value_checkpoint": str(args.value_checkpoint.resolve()),
            "improvement_weight": args.weight,
            "belief_samples": args.belief_samples,
            "search_streets": [1],
            "direct_pairs": args.direct_pairs,
            "seed": args.seed,
            "predeclared_safety_gate": "ci95_low >= -0.10 BB/hand",
        },
        "direct_value_search_vs_v1": asdict(direct),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
