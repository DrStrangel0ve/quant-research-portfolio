"""Offline archive boundaries that could silently corrupt a time-series study."""

from __future__ import annotations

import calendar
import csv
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest


def _load_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "projects/19_real_data_volatility/data.py"
    spec = importlib.util.spec_from_file_location("_portfolio_volatility_data_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not locate volatility data module")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve postponed annotations through the module registry.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


data = _load_module()


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_download(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Ingestion tests must never access the network")

    monkeypatch.setattr(data.urllib.request, "urlopen", unexpected_download)


def _period(month: str) -> tuple[datetime, datetime, int]:
    start = datetime.fromisoformat(month + "-01").replace(tzinfo=UTC)
    hours = calendar.monthrange(start.year, start.month)[1] * 24
    return start, start + timedelta(hours=hours), hours


def _rows(month: str = "2024-12") -> list[list[str]]:
    start, _, hours = _period(month)
    scale = 1_000 if month < "2025-01" else 1_000_000
    first = int(start.timestamp()) * scale
    return [
        [
            str(first + hour * 3_600 * scale),
            "100", "102", "99", "101", "10",
            str(first + (hour + 1) * 3_600 * scale - 1),
            "1000", "2", "5", "500", "0",
        ]
        for hour in range(hours)
    ]


def _case(
    rows: list[list[str]] | None = None,
    *,
    month: str = "2024-12",
    missing: Sequence[str] = (),
    zip_member: str | None = None,
    extra_member: bool = False,
) -> tuple[bytes, dict[str, Any]]:
    if rows is None:
        rows = _rows(month)
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    raw = stream.getvalue().encode("utf-8")
    filename = f"BTCUSDT-1h-{month}.csv"
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(zip_member if zip_member is not None else filename, raw)
        if extra_member:
            archive.writestr("unexpected.csv", b"extra")
    content = zipped.getvalue()
    start, end, hours = _period(month)
    url = f"https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-{month}.zip"
    record = {
        "symbol": "BTCUSDT",
        "month": month,
        "url": url,
        "checksum_url": url + ".CHECKSUM",
        "sha256": hashlib.sha256(content).hexdigest(),
        "archive_bytes": len(content),
        "csv_filename": filename,
        "uncompressed_bytes": len(raw),
        "rows": hours - len(missing),
        "expected_hours": hours,
        "period_start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "period_end_exclusive_utc": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timestamp_unit": "ms" if month < "2025-01" else "us",
        "missing_open_times_utc": list(missing),
    }
    return content, record


def _parse(rows: list[list[str]], **kwargs: Any) -> pd.DataFrame:
    content, record = _case(rows, **kwargs)
    return data.parse_archive(content, data.Archive.from_record(record))


def _write_dataset(
    directory: Path, cases: Sequence[tuple[bytes, dict[str, Any]]],
) -> tuple[Path, Path]:
    cache = directory / "cache"
    cache.mkdir()
    records = []
    for content, record in cases:
        filename = data.Archive.from_record(record).filename
        (cache / filename).write_bytes(content)
        records.append(record)
    manifest = directory / "manifest.json"
    manifest.write_text(
        json.dumps({
            "schema_version": 1, "dataset": "spot klines", "interval": "1h", "files": records,
        }),
        encoding="utf-8",
    )
    return manifest, cache


def test_timestamp_unit_transition_joins_two_months_without_rescaling_error(tmp_path: Path) -> None:
    december = _rows("2024-12")
    january = _rows("2025-01")
    # These anchors are independent literal milliseconds/microseconds at UTC midnight.
    assert december[0][0] == "1733011200000"
    assert january[0][0] == "1735689600000000"
    manifest, cache = _write_dataset(
        tmp_path, [_case(january, month="2025-01"), _case(december, month="2024-12")],
    )
    frames = data.load_hourly(manifest, cache)
    assert set(frames) == {"BTCUSDT"}
    frame = frames["BTCUSDT"]
    assert len(frame) == 1_488
    assert frame.index[743] == pd.Timestamp("2024-12-31T23:00:00Z")
    assert frame.index[744] == pd.Timestamp("2025-01-01T00:00:00Z")
    assert frame.index[-1] == pd.Timestamp("2025-01-31T23:00:00Z")
    assert frame.index.is_monotonic_increasing and frame.index.is_unique
    assert (frame.index[1:] - frame.index[:-1] == pd.Timedelta(hours=1)).all()
    assert frame.iloc[0][["open", "high", "low", "close"]].tolist() == [100, 102, 99, 101]


def test_explicit_missing_hour_is_preserved_without_filling_or_inventing_returns() -> None:
    rows = _rows()
    del rows[17]
    gap = "2024-12-01T17:00:00Z"
    frame = _parse(rows, missing=[gap])
    assert len(frame) == 743
    assert pd.Timestamp(gap) not in frame.index
    assert frame.index[17] - frame.index[16] == pd.Timedelta(hours=2)


@pytest.mark.parametrize("alteration", ["same_size_tamper", "shortened"])
def test_pinned_archive_rejects_corruption_before_unzipping(alteration: str) -> None:
    content, record = _case()
    corrupted = (
        bytes([content[0] ^ 1]) + content[1:]
        if alteration == "same_size_tamper" else content[:-1]
    )
    expected = "SHA-256 mismatch" if alteration == "same_size_tamper" else "size mismatch"
    with pytest.raises(data.DataValidationError, match=expected):
        data.parse_archive(corrupted, data.Archive.from_record(record))


@pytest.mark.parametrize(
    "member", ["../BTCUSDT-1h-2024-12.csv", "nested/BTCUSDT-1h-2024-12.csv", "wrong.csv"],
)
def test_zip_member_path_must_be_exact_even_with_matching_hash(member: str) -> None:
    content, record = _case(zip_member=member)
    with pytest.raises(data.DataValidationError, match="ZIP members"):
        data.parse_archive(content, data.Archive.from_record(record))


def test_zip_cannot_hide_additional_members_or_lie_about_uncompressed_size() -> None:
    content, record = _case(extra_member=True)
    with pytest.raises(data.DataValidationError, match="ZIP members"):
        data.parse_archive(content, data.Archive.from_record(record))
    content, record = _case()
    record["uncompressed_bytes"] += 1
    with pytest.raises(data.DataValidationError, match="member size"):
        data.parse_archive(content, data.Archive.from_record(record))


@pytest.mark.parametrize("columns", [11, 13])
def test_noncanonical_column_count_is_rejected(columns: int) -> None:
    rows = _rows()
    rows[0] = rows[0][:11] if columns == 11 else rows[0] + ["extra"]
    with pytest.raises(data.DataValidationError, match="12 columns"):
        _parse(rows)


@pytest.mark.parametrize(
    ("column", "value"),
    [(1, "nan"), (2, "inf"), (3, "-1"), (4, "0"), (4, "bad"), (5, "-1"), (7, "nan")],
)
def test_invalid_numeric_values_cannot_enter_forecasts(column: int, value: str) -> None:
    rows = _rows()
    rows[0][column] = value
    with pytest.raises(data.DataValidationError, match="invalid"):
        _parse(rows)


@pytest.mark.parametrize(("column", "value"), [(2, "100"), (3, "101"), (4, "103")])
def test_impossible_ohlc_bounds_are_rejected(column: int, value: str) -> None:
    rows = _rows()
    rows[0][column] = value
    with pytest.raises(data.DataValidationError, match="OHLC bounds"):
        _parse(rows)


@pytest.mark.parametrize("month", ["2024-12", "2025-01"])
def test_close_time_must_end_the_bar_in_its_original_timestamp_unit(month: str) -> None:
    rows = _rows(month)
    rows[0][6] = str(int(rows[0][6]) + 1)
    with pytest.raises(data.DataValidationError, match="close_time"):
        _parse(rows, month=month)


@pytest.mark.parametrize(
    "alteration", ["duplicate", "reordered", "missing", "off_grid", "wrong_month"],
)
def test_time_grid_rejects_silent_return_alignment_errors(alteration: str) -> None:
    rows = _rows()
    if alteration == "duplicate":
        rows[1] = rows[0].copy()
    elif alteration == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    elif alteration == "missing":
        del rows[17]
    else:
        offset = 1 if alteration == "off_grid" else 31 * 24 * 3_600_000
        for row in rows:
            row[0] = str(int(row[0]) + offset)
            row[6] = str(int(row[6]) + offset)
    with pytest.raises(data.DataValidationError, match="open time|row count|Hourly grid"):
        _parse(rows)


def test_declared_missing_exception_cannot_hide_a_different_missing_hour() -> None:
    rows = _rows()
    del rows[18]
    with pytest.raises(data.DataValidationError, match="Hourly grid"):
        _parse(rows, missing=["2024-12-01T17:00:00Z"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("symbol", "../BTCUSDT"),
        ("month", "2024-13"),
        ("url", "https://example.com/BTCUSDT-1h-2024-12.zip"),
        ("checksum_url", "https://example.com/checksum"),
        ("csv_filename", "../BTCUSDT-1h-2024-12.csv"),
        ("timestamp_unit", "us"),
        ("period_start_utc", "2024-12-01T01:00:00Z"),
        ("period_end_exclusive_utc", "2024-12-31T23:00:00Z"),
        ("rows", 743),
        ("expected_hours", 743),
        ("archive_bytes", 1_000_001),
        ("uncompressed_bytes", 2_000_001),
    ],
)
def test_manifest_prevents_path_unit_and_calendar_drift(field: str, value: Any) -> None:
    _, record = _case()
    record[field] = value
    with pytest.raises(data.DataValidationError):
        data.Archive.from_record(record)


def test_2025_manifest_cannot_claim_millisecond_timestamps() -> None:
    _, record = _case(month="2025-01")
    record["timestamp_unit"] = "ms"
    with pytest.raises(data.DataValidationError, match="timestamp unit"):
        data.Archive.from_record(record)


@pytest.mark.parametrize("second_month", ["2024-12", "2025-02"])
def test_manifest_rejects_duplicate_or_unlisted_months(tmp_path: Path, second_month: str) -> None:
    manifest, cache = _write_dataset(tmp_path, [_case(), _case(month=second_month)])
    with pytest.raises(data.DataValidationError, match="Duplicate, overlapping, or missing month"):
        data.load_hourly(manifest, cache)


def test_missing_cache_never_downloads_without_explicit_flag(tmp_path: Path) -> None:
    manifest, cache = _write_dataset(tmp_path, [_case()])
    (cache / "BTCUSDT-1h-2024-12.zip").unlink()
    with pytest.raises(FileNotFoundError, match="--download"):
        data.load_hourly(manifest, cache)


def test_corrupt_cache_is_not_replaced_even_when_download_is_enabled(tmp_path: Path) -> None:
    manifest, cache = _write_dataset(tmp_path, [_case()])
    path = cache / "BTCUSDT-1h-2024-12.zip"
    original = path.read_bytes()
    corrupted = bytes([original[0] ^ 1]) + original[1:]
    path.write_bytes(corrupted)
    with pytest.raises(data.DataValidationError, match="SHA-256 mismatch"):
        data.load_hourly(manifest, cache, download=True)
    assert path.read_bytes() == corrupted
