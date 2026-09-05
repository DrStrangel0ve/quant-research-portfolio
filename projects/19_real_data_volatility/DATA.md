# Data provenance and validation

This study uses historical **Binance spot BTCUSDT and ETHUSDT hourly candles**. The input is public exchange data, not simulated prices. All 62 archives were downloaded without credentials or payment and checked against the provider's companion SHA-256 files before their hashes were pinned in [manifest.json](manifest.json).

The primary upstream documentation describes the monthly archives, candle fields, checksum files, timestamp units, and archive revisions: [Binance Public Data](https://github.com/binance/binance-public-data). The underlying candle endpoint documents its UTC time convention and field meanings: [Binance Spot API: klines](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data).

## Coverage verified on 2026-09-05

| Property | Verified input |
|---|---|
| Instruments | BTCUSDT and ETHUSDT, spot, Binance |
| Interval | One hour, UTC |
| Months per instrument | December 2023 through June 2026 inclusive: 31 |
| Archives | 62 ZIP files, one CSV each |
| Compressed bytes | 2,607,526 |
| Uncompressed CSV bytes | 7,339,000 |
| Candle rows | 22,632 per instrument; 45,264 total |
| First bar open | 2023-12-01 00:00:00 UTC |
| Last bar open | 2026-06-30 23:00:00 UTC |
| Last bar ends | 2026-07-01 00:00:00 UTC, exclusive |
| Missing, duplicate, or off-grid hours | None in either instrument |
| Publisher checksum matches | 62 of 62 |

December 2023 supplies historical context for the first training forecasts. The specified study partitions are 2024 training, 2025 validation, and January–June 2026 evaluation. Data retrieval does not select parameters or examine forecast scores. These are retrospective partitions; this is not a prospective live experiment.

Each manifest record preserves its exact download URL, checksum URL, SHA-256, compressed/uncompressed byte counts, CSV name, row count, UTC monthly bounds, timestamp unit, HTTP `Last-Modified`, and retrieval time. The recorded acquisition finished at **2026-09-05T17:54:54.924516Z**. The complete manifest is suitable for a run fingerprint.

For example, the BTC June 2026 archive contains 720 rows in 42,265 compressed bytes. Its pinned SHA-256 is:

```text
7c446aee297f382ee92b8d9b3300a1d7c21bed8166118f3bb261275bed5e308e
```

[BTC June 2026 archive](https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2026-06.zip) · [Publisher checksum](https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2026-06.zip.CHECKSUM)

## Loading contract

```python
from pathlib import Path
from data import load_hourly

project = Path(__file__).resolve().parent
hourly = load_hourly(project / "manifest.json", project / "data_cache", download=True)
btc = hourly["BTCUSDT"]
```

`download=False` is the default. Missing files produce an error describing the runner's `--download` option. Downloads are limited to the canonical provider URL in the manifest; raw archives are cached under `data_cache/`. Existing files are verified again on every load. A length or hash mismatch is an error, including with downloads enabled. The loader never updates the manifest, fetches replacement hashes, or silently substitutes a revised archive. The provider documents that archived files can change; any intentional data revision must therefore be a separate reviewed change.

Each DataFrame has a UTC `DatetimeIndex` named `open_time`, with columns `open`, `high`, `low`, `close`, `base_volume`, `quote_volume`, `taker_buy_base_volume`, `taker_buy_quote_volume`, and `trade_count`. The raw schema also includes close time and an unused field; both are validated rather than retained as model inputs.

The strict parser checks:

- Exact ZIP bytes against the pinned SHA-256 and length; exactly one member with the expected CSV name and uncompressed length. It reads that member in memory and does not extract ZIP paths.
- Exactly 12 fields per row, positive finite OHLC prices, valid high/low bounds, finite nonnegative volumes, and integer trade counts.
- The provider's timestamp transition: milliseconds through December 2024 and microseconds beginning January 2025. Both are normalized to UTC without float timestamp conversion.
- Strictly increasing open times, close time equal to the next hour minus one source time unit, and the complete expected monthly hour grid.
- Consecutive months without duplicate, overlapping, or omitted periods in the manifest.

The manifest supports an explicit list of historically verified missing-hour exceptions for an archive. This study's lists are all empty. If a future pinned dataset has exceptions, the loader preserves the gap and verifies that the observed grid equals the expected grid minus exactly those exceptions. It never fills prices or creates rows. Return construction must independently reject any return that crosses a missing hour.

## Availability and interpretation

An index value denotes the **opening** of a bar. Its close becomes an available historical observation at the end of that hour; a model must not use that close to make a forecast at the bar's opening. A daily variance proxy formed from hourly close-to-close returns also needs the previous day's last hourly close. The first day of the downloaded warmup lacks the preceding close and should not be treated as a complete daily proxy.

Hourly candles do not reveal within-hour price paths, bid/ask quotes, queue position, executable fills, or order-book depth. Their squared-return aggregate is a coarse realized-variance proxy, not observed latent volatility. This dataset supports a historical forecasting comparison; it cannot establish execution performance, market-making profitability, cross-venue generalization, or live publication latency. Matching a publisher checksum establishes archive identity, not the economic accuracy of every observation.

The upstream README labels its project **MIT**, but the repository does not provide a separate raw-data license file. Provider attribution is retained here; raw archives are excluded from version control. This project distributes its ingestion code, pinned provenance, and derived study outputs rather than republishing the downloaded CSV/ZIP files.
