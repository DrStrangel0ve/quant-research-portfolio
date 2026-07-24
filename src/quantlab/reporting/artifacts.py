"""Small helpers for consistent experiment outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def write_summary(path: Path, values: Mapping[str, float | int | str | bool]) -> None:
    """Write a stable, human-readable JSON scorecard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned: dict[str, float | int | str | bool] = {}
    for key, value in values.items():
        if isinstance(value, np.integer):
            cleaned[key] = int(value)
        elif isinstance(value, np.floating):
            cleaned[key] = float(value)
        else:
            cleaned[key] = value
    path.write_text(json.dumps(cleaned, indent=2, sort_keys=True) + "\n", encoding="utf-8")
