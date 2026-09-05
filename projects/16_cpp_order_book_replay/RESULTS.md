# Recorded local benchmark and verification

Measured 2026-09-05 on Windows 11 (`10.0.26200`), Intel Core i5-14400F, 10 cores / 16 logical processors, x64. Compiler: MSVC **19.50.35719** (`_MSC_FULL_VER=195035719`), CMake 4.2.1, Visual Studio 18 2026 generator, `Release` with `NDEBUG` defined and default CMake Release optimization flags. Ordinary desktop session, no affinity pinning or CPU-frequency controls. No other benchmark was intentionally run concurrently; the desktop was not isolated from background work.

These measurements are specific to this machine, allocator, compiler, synthetic trace, and timing method. The reference is a deliberately simple global-list implementation with linear scans for IDs/prices. Large differences do not represent comparisons against production or other optimized engines.

## Commands and correctness

```powershell
cmake -S projects/16_cpp_order_book_replay -B projects/16_cpp_order_book_replay/build -G "Visual Studio 18 2026" -A x64
cmake --build projects/16_cpp_order_book_replay/build --config Release --parallel
ctest --test-dir projects/16_cpp_order_book_replay/build -C Release --output-on-failure
& projects/16_cpp_order_book_replay/build/Release/book_tests.exe
foreach ($depth in @(16, 2000, 8192)) {
  & projects/16_cpp_order_book_replay/build/Release/book_benchmark.exe 12000 20270901 $depth 3
}
```

The core tests passed **199,286 checks**, including **40,000 randomized events / eight seeds**. The tests remain active in Release. The replay fixture output was:

```text
TRADE,1,4,101,5
TRADE,2,4,101,3
CANCEL,2,1
TRADE,3,5,102,4
REST,6,S,105,9
events=7 trades=3 resting=1
```

All benchmark runs verified complete operation results against the independent scan engine before timing, and checked a deterministic result digest plus the final resting state after every measured pass. A digest is a replay guard, not a substitute for the exact comparison.

## Workload counts

Each run uses **12,000 measured events**, seed **20270901**, and three repeats of the same generated trace. Different starting depths generate different traces because cancellation selects from the current active book and depth boundaries alter operation choices. This is a scaling illustration, not a paired identical-event experiment across depths.

| Initial orders | Min–max resident orders | Final orders | Passive adds | Aggressive adds | Successful cancels | Trades | Result digest |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 14–17 | 16 | 5,419 | 1,179 | 5,402 | 1,193 | 11981076279869920838 |
| 2,000 | 1,989–2,200 | 2,183 | 5,321 | 1,729 | 4,950 | 1,891 | 13074359819520561592 |
| 8,192 | 8,189–8,696 | 8,692 | 5,463 | 1,779 | 4,758 | 1,950 | 12347485608375999948 |

Small aggressive quantities mainly produce partial fills. This workload does not characterize large sweeps, pathological hashing, peak bursts, or cold-start behavior. The tests include sweeps, but there is no performance claim for them.

## All measured repetitions

Throughput is from a replay with no per-event clocks and includes result digestion/destruction. Percentiles are from a separate instrumented replay around `apply()`, excluding result digestion/destruction. See the README for exact boundaries. Numbers below retain the tool's reported units; the display does not imply that precision is achievable.

| Initial orders | Engine | Repeat | Events/s | p50 (ns) | p99 (ns) |
|---:|---|---:|---:|---:|---:|
| 16 | indexed | 1 | 9,681,323.1 | 100 | 300 |
| 16 | scan reference | 1 | 11,788,977.3 | 100 | 200 |
| 16 | indexed | 2 | 7,879,703.2 | 100 | 300 |
| 16 | scan reference | 2 | 12,133,468.1 | 100 | 200 |
| 16 | indexed | 3 | 9,331,259.7 | 100 | 300 |
| 16 | scan reference | 3 | 12,390,294.3 | 100 | 200 |
| 2,000 | indexed | 1 | 7,759,456.8 | 100 | 300 |
| 2,000 | scan reference | 1 | 73,102.5 | 16,800 | 37,800 |
| 2,000 | indexed | 2 | 11,291,992.1 | 100 | 300 |
| 2,000 | scan reference | 2 | 71,802.9 | 16,800 | 42,500 |
| 2,000 | indexed | 3 | 9,445,100.4 | 100 | 300 |
| 2,000 | scan reference | 3 | 71,406.9 | 17,100 | 38,100 |
| 8,192 | indexed | 1 | 8,720,296.5 | 200 | 400 |
| 8,192 | scan reference | 1 | 15,092.3 | 81,300 | 224,400 |
| 8,192 | indexed | 2 | 8,752,096.9 | 100 | 300 |
| 8,192 | scan reference | 2 | 14,722.4 | 83,500 | 222,400 |
| 8,192 | indexed | 3 | 9,321,112.3 | 100 | 500 |
| 8,192 | scan reference | 3 | 15,039.2 | 83,000 | 208,900 |

Back-to-back clock readings for every depth: **p50 = 0 ns, p99 = 100 ns**. The clock observations show roughly 100 ns quantization. The indexed latency results are close to that resolution; no sub-100 ns interpretation or overhead-corrected latency is justified. Throughput repetitions also vary materially: the 2,000-order indexed sample spans about 7.76–11.29 million events/s. More repeated/longer experiments on an isolated machine would be needed for precise performance estimates or uncertainty intervals.

Median throughput over the three repetitions is about 9.33M / 12.13M events/s for indexed / scan at 16 initial orders, 9.45M / 71.8k at 2,000, and 8.75M / 15.0k at 8,192. The relevant engineering result is the crossover: indexing costs more for the tiny book, while global linear scans dominate for populated books. Performance results should be reproduced before inclusion in a resume or interview claim.
