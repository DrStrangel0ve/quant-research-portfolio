"""Input validation shared by research modules."""

from __future__ import annotations

import numpy as np
import pandas as pd


def require_datetime_index(frame: pd.Series | pd.DataFrame, name: str) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must use a DatetimeIndex")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be sorted")
    if frame.index.has_duplicates:
        raise ValueError(f"{name} index must not contain duplicates")


def require_finite(frame: pd.Series | pd.DataFrame, name: str) -> None:
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")


def require_same_shape_and_labels(left: pd.DataFrame, right: pd.DataFrame) -> None:
    if not left.index.equals(right.index):
        raise ValueError("returns and positions must have identical indices")
    if not left.columns.equals(right.columns):
        raise ValueError("returns and positions must have identical columns")
