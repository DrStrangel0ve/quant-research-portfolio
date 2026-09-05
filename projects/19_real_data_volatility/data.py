"""Download pinned public archives and validate their hourly spot candles.

No download occurs unless ``download=True``. Existing corrupt or changed cache
entries fail closed; neither the manifest nor its hashes are updated here.
"""

from __future__ import annotations

import calendar
import csv
import hashlib
import io
import json
import math
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pandas as pd

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
MAX_ARCHIVE_BYTES = 1_000_000
MAX_CSV_BYTES = 2_000_000


class DataValidationError(ValueError):
    """The pinned manifest, archive, or candle data violated its contract."""


def _text(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise DataValidationError(f"Manifest field {field!r} must be a string")
    return value


def _count(record: dict[str, Any], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise DataValidationError(f"Manifest field {field!r} must be a nonnegative integer")
    return value


def _utc(value: str) -> datetime:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value):
        raise DataValidationError(f"Expected an explicit UTC timestamp, got {value!r}")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass(frozen=True)
class Archive:
    symbol: str
    month: str
    url: str
    sha256: str
    archive_bytes: int
    csv_filename: str
    uncompressed_bytes: int
    rows: int
    start: datetime
    end: datetime
    unit: Literal["ms", "us"]
    missing: tuple[datetime, ...]

    @property
    def filename(self) -> str:
        return f"{self.symbol}-1h-{self.month}.zip"

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Archive:
        symbol = _text(record, "symbol")
        month = _text(record, "month")
        if not re.fullmatch(r"[A-Z0-9]{2,20}", symbol):
            raise DataValidationError("Invalid symbol in manifest")
        if not re.fullmatch(r"\d{4}-\d{2}", month):
            raise DataValidationError("Invalid month in manifest")
        try:
            start = datetime(int(month[:4]), int(month[5:]), 1, tzinfo=UTC)
        except ValueError as error:
            raise DataValidationError("Invalid calendar month in manifest") from error
        hours = calendar.monthrange(start.year, start.month)[1] * 24
        end = start + timedelta(hours=hours)
        filename = f"{symbol}-1h-{month}.zip"
        expected_url = f"{BASE_URL}/{symbol}/1h/{filename}"
        if _text(record, "url") != expected_url:
            raise DataValidationError("Archive URL must match its canonical provider path")
        if _text(record, "checksum_url") != expected_url + ".CHECKSUM":
            raise DataValidationError("Checksum URL must match its archive")
        sha256 = _text(record, "sha256")
        if not re.fullmatch(r"[a-f0-9]{64}", sha256):
            raise DataValidationError("Expected a lowercase SHA-256 digest")
        if _text(record, "csv_filename") != filename.removesuffix(".zip") + ".csv":
            raise DataValidationError("Unexpected CSV filename in manifest")
        if _utc(_text(record, "period_start_utc")) != start:
            raise DataValidationError("Archive start does not match calendar month")
        if _utc(_text(record, "period_end_exclusive_utc")) != end:
            raise DataValidationError("Archive end does not match calendar month")
        if _count(record, "expected_hours") != hours:
            raise DataValidationError("Expected hour count does not match calendar month")
        unit: Literal["ms", "us"] = "ms" if start.year < 2025 else "us"
        if _text(record, "timestamp_unit") != unit:
            raise DataValidationError("Unexpected provider timestamp unit for this period")
        raw_missing = record.get("missing_open_times_utc")
        if not isinstance(raw_missing, list) or not all(
            isinstance(value, str) for value in raw_missing
        ):
            raise DataValidationError("Missing-hour exceptions must be an explicit timestamp list")
        missing = tuple(_utc(value) for value in raw_missing)
        if len(set(missing)) != len(missing) or tuple(sorted(missing)) != missing:
            raise DataValidationError("Missing-hour exceptions must be unique and sorted")
        if any(not start <= value < end or value.minute or value.second for value in missing):
            raise DataValidationError("Missing-hour exception is not a grid point in its month")
        rows = _count(record, "rows")
        if rows != hours - len(missing):
            raise DataValidationError("Manifest row count does not match declared hourly grid")
        archive_bytes = _count(record, "archive_bytes")
        uncompressed_bytes = _count(record, "uncompressed_bytes")
        if not 0 < archive_bytes <= MAX_ARCHIVE_BYTES:
            raise DataValidationError("Archive size is outside the hourly dataset bound")
        if not 0 < uncompressed_bytes <= MAX_CSV_BYTES:
            raise DataValidationError("CSV size is outside the hourly dataset bound")
        return cls(
            symbol,
            month,
            expected_url,
            sha256,
            archive_bytes,
            filename.removesuffix(".zip") + ".csv",
            uncompressed_bytes,
            rows,
            start,
            end,
            unit,
            missing,
        )


def _manifest(path: Path) -> list[Archive]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise DataValidationError("Expected manifest schema_version 1")
    if raw.get("interval") != "1h" or raw.get("dataset") != "spot klines":
        raise DataValidationError("Manifest must describe hourly spot klines")
    records = raw.get("files")
    if not isinstance(records, list) or not records:
        raise DataValidationError("Manifest files must be a nonempty list")
    if not all(isinstance(record, dict) for record in records):
        raise DataValidationError("Manifest archive records must be objects")
    archives = sorted(
        (Archive.from_record(record) for record in records),
        key=lambda archive: (archive.symbol, archive.month),
    )
    previous: Archive | None = None
    for archive in archives:
        if (
            previous is not None
            and previous.symbol == archive.symbol
            and previous.end != archive.start
        ):
            raise DataValidationError("Duplicate, overlapping, or missing month in manifest")
        previous = archive
    return archives


def _checked_bytes(content: bytes, archive: Archive) -> bytes:
    if len(content) != archive.archive_bytes:
        raise DataValidationError(f"Archive size mismatch: {archive.filename}")
    actual = hashlib.sha256(content).hexdigest()
    if actual != archive.sha256:
        raise DataValidationError(
            f"SHA-256 mismatch: {archive.filename}; expected {archive.sha256}, got {actual}. "
            "The pinned manifest has not been changed."
        )
    return content


def _archive_bytes(archive: Archive, cache_dir: Path, download: bool) -> bytes:
    path = cache_dir / archive.filename
    if path.exists():
        if path.stat().st_size != archive.archive_bytes:
            raise DataValidationError(f"Cached archive size mismatch: {path}")
        return _checked_bytes(path.read_bytes(), archive)
    if not download:
        raise FileNotFoundError(
            f"Missing pinned archive: {path}. Run this project's run.py with --download "
            "to fetch the public source files."
        )
    request = urllib.request.Request(archive.url, headers={"User-Agent": "quant-research-study/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        content = bytes(response.read(MAX_ARCHIVE_BYTES + 1))
    _checked_bytes(content, archive)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # A failed/partial write cannot later pass the length/hash checks. Do not
    # overwrite an existing archive or replace a publisher revision silently.
    with path.open("xb") as handle:
        handle.write(content)
    return content


def _integer(value: str, field: str, row: int) -> int:
    if not re.fullmatch(r"\d+", value):
        raise DataValidationError(f"Row {row}: {field} must be a nonnegative integer")
    return int(value)


def _number(value: str, field: str, row: int, *, positive: bool) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise DataValidationError(f"Row {row}: invalid {field}") from error
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise DataValidationError(f"Row {row}: invalid finite/range requirement for {field}")
    return result


def parse_archive(content: bytes, archive: Archive) -> pd.DataFrame:
    """Verify an exact pinned ZIP, then parse its sole expected CSV in memory."""
    _checked_bytes(content, archive)
    with zipfile.ZipFile(io.BytesIO(content)) as zipped:
        infos = zipped.infolist()
        if len(infos) != 1 or infos[0].filename != archive.csv_filename:
            raise DataValidationError(f"Unexpected ZIP members: {archive.filename}")
        if infos[0].is_dir() or infos[0].file_size != archive.uncompressed_bytes:
            raise DataValidationError(f"Unexpected ZIP member size/type: {archive.filename}")
        raw = zipped.read(infos[0])
    if len(raw) != archive.uncompressed_bytes:
        raise DataValidationError(f"Uncompressed size mismatch: {archive.filename}")

    scale = 1_000 if archive.unit == "ms" else 1_000_000
    timestamps: list[int] = []
    values: list[dict[str, float | int]] = []
    for row_number, row in enumerate(csv.reader(io.StringIO(raw.decode("utf-8"))), start=1):
        if len(row) != 12:
            raise DataValidationError(f"Row {row_number}: expected exactly 12 columns")
        timestamp = _integer(row[0], "open_time", row_number)
        close_time = _integer(row[6], "close_time", row_number)
        if close_time != timestamp + 3_600 * scale - 1:
            raise DataValidationError(f"Row {row_number}: close_time does not end its hourly bar")
        if timestamps and timestamp <= timestamps[-1]:
            raise DataValidationError(f"Row {row_number}: duplicate or nonincreasing open time")
        timestamps.append(timestamp)
        candle: dict[str, float | int] = {}
        for column, field in ((1, "open"), (2, "high"), (3, "low"), (4, "close")):
            candle[field] = _number(row[column], field, row_number, positive=True)
        if candle["high"] < max(candle["open"], candle["close"], candle["low"]) or (
            candle["low"] > min(candle["open"], candle["close"], candle["high"])
        ):
            raise DataValidationError(f"Row {row_number}: inconsistent OHLC bounds")
        for column, field in (
            (5, "base_volume"),
            (7, "quote_volume"),
            (9, "taker_buy_base_volume"),
            (10, "taker_buy_quote_volume"),
        ):
            candle[field] = _number(row[column], field, row_number, positive=False)
        candle["trade_count"] = _integer(row[8], "trade_count", row_number)
        if _integer(row[11], "ignore", row_number) != 0:
            raise DataValidationError(f"Row {row_number}: noncanonical unused field")
        values.append(candle)
    if len(values) != archive.rows:
        raise DataValidationError(f"Unexpected CSV row count: {archive.filename}")
    index = pd.DatetimeIndex(pd.to_datetime(timestamps, unit=archive.unit, utc=True))
    expected = pd.date_range(archive.start, archive.end, freq="h", inclusive="left")
    if archive.missing:
        expected = expected.difference(pd.DatetimeIndex(archive.missing))
    if not index.equals(expected):
        raise DataValidationError(f"Hourly grid differs from pinned exceptions: {archive.filename}")
    frame = pd.DataFrame(values, index=index)
    frame.index.name = "open_time"
    return frame


def load_hourly(
    manifest_path: str | Path,
    cache_dir: str | Path,
    download: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load validated UTC hourly bars, preserving any explicitly pinned gaps.

    The index is the bar's *open* time; its closing price becomes available only
    at the end of that hour. This function does not resample, forward-fill, or
    fabricate missing observations. Callers must handle gaps before returns.
    """
    grouped: dict[str, list[pd.DataFrame]] = {}
    for archive in _manifest(Path(manifest_path)):
        try:
            content = _archive_bytes(archive, Path(cache_dir), download)
            frame = parse_archive(content, archive)
        except (ValueError, zipfile.BadZipFile) as error:
            raise DataValidationError(f"{archive.filename}: {error}") from error
        grouped.setdefault(archive.symbol, []).append(frame)
    return {symbol: pd.concat(frames) for symbol, frames in grouped.items()}
