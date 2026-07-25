"""Run every portfolio experiment and fail fast on integration errors."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PROJECTS = (
    "01_option_pricing",
    "02_heston_model",
    "03_delta_hedging",
    "04_market_making",
    "05_limit_order_book",
    "06_pairs_trading",
    "07_cross_sectional_momentum",
    "08_walk_forward_trend",
    "09_garch_forecasting",
    "10_risk_parity",
    "11_probability_games",
    "12_kelly_and_stopping",
    "13_poker_cfr_lab",
)


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    started = time.perf_counter()
    for project in PROJECTS:
        script = repository / "projects" / project / "run.py"
        project_started = time.perf_counter()
        subprocess.run(
            [sys.executable, str(script)],
            cwd=repository,
            check=True,
        )
        elapsed = time.perf_counter() - project_started
        print(f"PASS {project} ({elapsed:.2f}s)")
    total = time.perf_counter() - started
    print(f"All {len(PROJECTS)} projects passed in {total:.2f}s")


if __name__ == "__main__":
    main()
