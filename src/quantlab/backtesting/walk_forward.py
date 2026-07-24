"""Purged chronological train/test splits for time-series research."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class WalkForwardSplit:
    train: NDArray[np.int64]
    test: NDArray[np.int64]


def expanding_window_splits(
    n_observations: int,
    *,
    min_train_size: int,
    test_size: int,
    step_size: int | None = None,
    embargo: int = 0,
) -> Iterator[WalkForwardSplit]:
    """Yield expanding-window chronological splits with an optional embargo."""
    if n_observations <= 0:
        raise ValueError("n_observations must be positive")
    if min_train_size <= 0 or test_size <= 0:
        raise ValueError("window sizes must be positive")
    if embargo < 0:
        raise ValueError("embargo cannot be negative")
    step = test_size if step_size is None else step_size
    if step <= 0:
        raise ValueError("step_size must be positive")

    test_start = min_train_size + embargo
    while test_start + test_size <= n_observations:
        train_end = test_start - embargo
        yield WalkForwardSplit(
            train=np.arange(0, train_end),
            test=np.arange(test_start, test_start + test_size),
        )
        test_start += step
