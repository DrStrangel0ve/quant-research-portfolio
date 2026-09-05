# Integer-tick order book and deterministic replay

A C++20 matching engine with price/time priority, partial fills, cancellation by order ID, a strict CSV replay tool, and an independently implemented scan reference. The experiment asks: **when do indexed price levels justify their extra storage and allocation costs?** The supplied benchmark includes tiny and populated books; the scan reference wins in the tiny-book sample.

This is a single-threaded in-memory matching exercise. It does not claim exchange compatibility or live trading performance.

## Build and run

From the repository root, with CMake and a C++20 compiler:

```sh
cmake -S projects/16_cpp_order_book_replay -B projects/16_cpp_order_book_replay/build -DCMAKE_BUILD_TYPE=Release
cmake --build projects/16_cpp_order_book_replay/build --config Release --parallel
ctest --test-dir projects/16_cpp_order_book_replay/build -C Release --output-on-failure
```

For a single-configuration generator, executables are directly in `build/`:

```sh
projects/16_cpp_order_book_replay/build/book_replay projects/16_cpp_order_book_replay/data/example.csv
projects/16_cpp_order_book_replay/build/book_benchmark 12000 20270901 2000 3
```

Visual Studio places executables in `build/Release/`; use `.exe`, for example:

```powershell
& projects/16_cpp_order_book_replay/build/Release/book_benchmark.exe 12000 20270901 2000 3
```

GCC/Clang sanitizer build, in a separate directory:

```sh
cmake -S projects/16_cpp_order_book_replay -B projects/16_cpp_order_book_replay/build/sanitized -DCMAKE_BUILD_TYPE=Debug -DBOOK_SANITIZERS=ON
cmake --build projects/16_cpp_order_book_replay/build/sanitized --parallel
ctest --test-dir projects/16_cpp_order_book_replay/build/sanitized --output-on-failure
```

## Semantics and data structures

| Decision | Contract / tradeoff |
|---|---|
| `int64_t` prices | Positive integer ticks. No floating-point equality or price/quantity multiplication. Tick size is external to this engine. |
| `uint64_t` quantities and IDs | Positive input quantity and ID. A duplicate **active** ID is rejected before matching. An ID can be reused after complete execution or cancellation. |
| `std::map<Price, std::list<Order>>` per side | Ordered price discovery and FIFO within each price level. Incoming orders execute against the best opposite price, at the resting maker's price. |
| `std::unordered_map<Id, Location>` | Stores stable map/list iterators. Cancellation does not search price levels or scan the queue. Map/list iterators survive unrelated insertions and hash-index rehashes. |
| Residual limit orders | An unfilled incoming quantity rests at its original limit and joins the end of its FIFO queue. Only limit-add and cancel operations are supported. |
| Unknown cancellation | Returns `false`; book state is unchanged. Repeated cancellation is harmless. |
| Reproducible priority | Input sequence defines time priority. There is no wall-clock dependency. Snapshots list bids descending, asks ascending, preserving FIFO. |

For `L` price levels, `N` resting orders, and `F` executed maker orders: a noncrossing add needs `O(log L)` level lookup and expected/amortized constant hash/list work. Cancellation uses expected constant hash lookup and amortized constant erasure through stored iterators. Matching visits executed makers directly, plus `O(log L)` when a residual creates/joins its level. Hash-table rehashes and adversarial collisions prevent per-operation worst-case latency guarantees. Snapshots and invariant checks are deliberately outside benchmark timing.

`reference.hpp` stores all active orders in one arrival-ordered list and scans for IDs and the next best maker. Equal-price ties retain arrival order. This is a readable correctness and complexity baseline, **not another optimized matching engine**. The implementations share the data types, input validation, and snapshot ordering; they use different storage and matching algorithms. Handwritten expected-trade tests reduce dependence on the reference.

The indexed engine is noncopyable and nonmovable because its lookup entries contain iterators. Input validation fails without mutation. Allocation failures preserve structural consistency, but a multi-fill event is not transactional: fills already applied before a later allocation failure are not rolled back. There is no persistence, crash recovery, or exception fault-injection suite.

## Correctness evidence

`book_tests` uses throwing checks, which remain active under `NDEBUG` in Release builds. It covers:

- Exact expected trades for buy/sell sweeps, price improvement, equal-price FIFO, partial fills, and residual orders.
- Cancellation of middle and final orders, repeated cancellation, ID reuse, and hash-index rehash followed by cancellation and full-book draining.
- Zero/negative prices, zero quantity/ID, invalid side, duplicate active IDs, and maximum representable price/quantity.
- **40,000 randomized events across eight fixed seeds**, comparing every trade sequence, cancellation result, and canonical resting state with the reference after each event.
- Independent maker/taker contracts and quantity-conservation checks; internal level/index consistency and uncrossed-book invariants after each randomized event.

The verified Windows Release run completed **199,286 checks**. CTest also runs the valid replay fixture and rejects five malformed/invalid replay inputs. Seeds and failing event indices are reported for reproduction.

## Replay format

```text
# Add: operation,id,side,price_ticks,quantity
A,1,S,101,5
A,2,B,102,3
# Cancel: operation,id
C,1
```

Only exact `A`/`C`, uppercase `B`/`S`, and integer fields are accepted; whitespace normalization and quoted CSV are intentionally unsupported. Blank lines and lines beginning with `#` are ignored. Malformed fields, integer overflow, and rejected adds stop the replay with a line-numbered error. Earlier successfully processed events and their output remain; this is not an atomic file import.

The example fixture yields three maker-priced trades, one successful cancellation, and the single remaining sell order `6` at `105` ticks for quantity `9`.

## Benchmark interpretation

CLI arguments are `events seed initial_depth repeats`. The generator preloads passive orders across up to 100 price levels per side, then mixes passive adds, aggressive limit adds, and successful cancellations of randomly selected active orders. Passive quantities are 20–100; aggressive quantities are 1–12. Nominal operation probabilities are 45% passive / 15% aggressive / 40% cancel. Boundary overrides favor adds/cancels near 90%/110% of initial depth to keep the book populated at a comparable scale. This is a synthetic operation mix, not a model of exchange arrivals or market participants. The generator uses a fixed `mt19937_64` stream and modulo mapping; the financial significance of its seed is zero.

Every benchmark first compares **each operation's complete result** and the final FIFO state against the reference. Each timed replay also checks its result digest and final state after timing. Setup, parsing, input generation, reference verification, and invariant checks are excluded. Every pass starts from the same preloaded book; execution order alternates between engines across repeats.

- **Throughput** comes from a whole replay with only a start/end clock read. It includes result digestion and returned trade-vector destruction. It excludes book construction/destruction and preloading.
- **p50/p99** come from a separate replay, with clock reads surrounding `apply()`. They include its result allocation and timer effects, but exclude result digestion and returned trade-vector destruction. They summarize the mixed operation workload, not individual operation types.
- Percentiles use the nearest-rank method. Back-to-back clock readings are reported separately; no overhead is subtracted. On the measured Windows machine, readings were quantized in roughly **100 ns** steps, and back-to-back p50 was zero. An indexed p50 of 100 ns therefore cannot establish precision below 100 ns or a precise 100 ns underlying execution cost.
- No core affinity, frequency locking, isolated CPU, or allocation pool is used. Short throughput passes can be noisy. These are local API processing observations, not wire-to-wire, network, exchange, concurrent, or production latency measurements.

See [RESULTS.md](RESULTS.md) for the machine, compiler, exact workload counts, all repetitions, and limits. The result demonstrates a storage tradeoff: tiny books favor a simple scan; the indexed implementation avoids linear search as resident depth grows. Production alternatives such as dense tick arrays, intrusive queues, memory pools, flat containers, and other optimized engines have not been compared.

## Useful extensions

Add per-operation latency distributions and allocator counts, profile cache misses, and compare pooled/intrusive or dense tick-range storage on the same verified trace. Extend the event protocol with carefully specified replace semantics before adding concurrency. A real exchange feed replay would require exchange-specific event semantics and data provenance rather than treating this fixture as market data.
