"""Export trained and reference decisions for the browser poker arena."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantlab.poker.cfr import CFRPlusTrainer
from quantlab.poker.evaluation import best_response
from quantlab.poker.rlcard_adapter import load_rlcard_reference_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).parent / "artifacts" / "cfr_plus_checkpoint.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "artifacts" / "arena_policies.json",
    )
    args = parser.parse_args()

    trained = CFRPlusTrainer.load(args.checkpoint).average_policy()
    reference = load_rlcard_reference_policy()
    response_zero = best_response(trained, player=0)
    response_one = best_response(trained, player=1)
    payload = {
        "format": "quantlab-leduc-arena-v1",
        "trained": {
            "label": "CFR+ 20K",
            "information_mode": trained.information_mode,
            "policy": trained.table,
        },
        "rlcard_reference": {
            "label": "RLCard CFR",
            "information_mode": reference.information_mode,
            "policy": reference.table,
        },
        "exploit_bot": {
            "label": "Exact Punisher",
            "target": "CFR+ 20K average policy",
            "actions_by_seat": {
                "0": {key: int(action) for key, action in response_zero.actions.items()},
                "1": {key: int(action) for key, action in response_one.actions.items()},
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"Wrote {args.output} with {len(trained.table)} trained and "
        f"{len(reference.table)} reference information states"
    )


if __name__ == "__main__":
    main()
