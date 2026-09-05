#include "book.hpp"
#include "reference.hpp"
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <thread>

using namespace lob;
using Clock = std::chrono::steady_clock;
namespace {
struct Event { bool cancellation; Order order; };
struct Result {
  bool cancelled = false;
  std::vector<Trade> trades;
  bool operator==(const Result&) const = default;
};
struct Workload {
  std::vector<Order> initial;
  std::vector<Event> events;
  std::vector<Order> final_state;
  std::size_t passive = 0, aggressive = 0, cancels = 0, fills = 0, minimum = 0, maximum = 0;
  std::uint64_t digest = 0;
};
template<class Engine> Result apply(Engine& book, const Event& event) {
  if (event.cancellation) return {book.cancel(event.order.id), {}};
  return {false, book.add(event.order)};
}
void digest(std::uint64_t& hash, const Result& result) {
  const auto mix = [&](std::uint64_t value) { hash = (hash ^ value) * 1099511628211ULL; };
  mix(result.cancelled);
  mix(result.trades.size());
  for (const auto& trade : result.trades) {
    mix(trade.maker); mix(trade.taker); mix(static_cast<std::uint64_t>(trade.price)); mix(trade.quantity);
  }
}
Workload make_workload(std::uint64_t seed, std::size_t count, std::size_t depth) {
  std::mt19937_64 random(seed);
  Workload workload;
  workload.minimum = depth;
  workload.maximum = depth;
  Book state;
  Id next_id = 1;
  const auto passive = [&] {
    const auto side = random() % 2 ? Side::buy : Side::sell;
    return Order{next_id++, side, (side == Side::buy ? 9800 : 10100) + static_cast<Price>(random() % 100), 20 + random() % 81};
  };
  for (std::size_t i = 0; i < depth; ++i) {
    const auto order = passive();
    workload.initial.push_back(order);
    state.add(order);
  }
  workload.events.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    auto kind = random() % 100;
    // Keep resident depth comparable across the run instead of quietly draining
    // the book or letting a small-depth workload grow into a large-depth one.
    if (state.size() <= std::max<std::size_t>(1, depth * 9 / 10)) kind = 99;
    if (state.size() >= depth + std::max<std::size_t>(1, depth / 10)) kind = 0;
    Event event{};
    if (kind < 40 && state.size()) {
      const auto active = state.snapshot();
      event = {true, active[random() % active.size()]};
      ++workload.cancels;
    } else if (kind < 55) {
      const auto side = random() % 2 ? Side::buy : Side::sell;
      event = {false, {next_id++, side, side == Side::buy ? 10200 : 9700, 1 + random() % 12}};
      ++workload.aggressive;
    } else {
      event = {false, passive()};
      ++workload.passive;
    }
    const auto result = apply(state, event);
    workload.fills += result.trades.size();
    digest(workload.digest, result);
    workload.events.push_back(event);
    workload.minimum = std::min(workload.minimum, state.size());
    workload.maximum = std::max(workload.maximum, state.size());
  }
  state.check_invariants();
  workload.final_state = state.snapshot();
  return workload;
}
void verify(const Workload& workload) {
  Book engine;
  ReferenceBook reference;
  for (const auto& order : workload.initial) { engine.add(order); reference.add(order); }
  for (const auto& event : workload.events) {
    if (apply(engine, event) != apply(reference, event)) throw std::runtime_error("benchmark differential result mismatch");
  }
  engine.check_invariants();
  if (engine.snapshot() != reference.snapshot() || engine.snapshot() != workload.final_state)
    throw std::runtime_error("benchmark differential final-state mismatch");
}
double percentile(std::vector<double> samples, double fraction) {
  std::sort(samples.begin(), samples.end());
  return samples[static_cast<std::size_t>(std::ceil(fraction * static_cast<double>(samples.size()))) - 1];
}
struct Measurement { double events_per_second, p50_ns, p99_ns; };
template<class Engine> Measurement measure(const Workload& workload, bool instrument) {
  Engine book;
  for (const auto& order : workload.initial) book.add(order);
  std::vector<double> samples;
  if (instrument) samples.reserve(workload.events.size());
  std::uint64_t hash = 0;
  const auto begin = Clock::now();
  for (const auto& event : workload.events) {
    if (instrument) {
      const auto start = Clock::now();
      const auto result = apply(book, event);
      const auto stop = Clock::now();
      samples.push_back(std::chrono::duration<double, std::nano>(stop - start).count());
      digest(hash, result);
    } else {
      digest(hash, apply(book, event));
    }
  }
  const auto elapsed = std::chrono::duration<double>(Clock::now() - begin).count();
  if (hash != workload.digest || book.snapshot() != workload.final_state)
    throw std::runtime_error("measured replay diverged from verified results");
  return {static_cast<double>(workload.events.size()) / elapsed,
          instrument ? percentile(samples, 0.50) : 0,
          instrument ? percentile(samples, 0.99) : 0};
}
template<class Engine> void report(const Workload& workload, std::string_view engine, std::size_t repeat) {
  const auto throughput = measure<Engine>(workload, false);
  const auto latency = measure<Engine>(workload, true);
  std::cout << engine << ',' << repeat << ',' << std::fixed << std::setprecision(1)
            << throughput.events_per_second << ',' << latency.p50_ns << ',' << latency.p99_ns << '\n';
}
std::uint64_t positive(std::string_view value) {
  std::uint64_t result{};
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc{} || end != value.data() + value.size() || result == 0)
    throw std::invalid_argument("arguments must be positive integers");
  return result;
}
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc > 5) throw std::invalid_argument("usage: book_benchmark [events=12000] [seed=20270901] [initial_depth=2000] [repeats=3]");
    const auto count = argc > 1 ? positive(argv[1]) : 12000;
    const auto seed = argc > 2 ? positive(argv[2]) : 20270901;
    const auto depth = argc > 3 ? positive(argv[3]) : 2000;
    const auto repeats = argc > 4 ? positive(argv[4]) : 3;
    if (count > 1000000 || depth > 100000 || repeats > 100) throw std::invalid_argument("limits: 1000000 events, 100000 initial depth, 100 repeats");
#ifdef _MSC_VER
    std::cout << "# compiler=MSVC " << _MSC_FULL_VER << '\n';
#elif defined(__clang__)
    std::cout << "# compiler=Clang " << __clang_version__ << '\n';
#elif defined(__GNUC__)
    std::cout << "# compiler=GCC " << __VERSION__ << '\n';
#endif
#ifdef NDEBUG
    std::cout << "# build=Release (NDEBUG defined)\n";
#else
    std::cout << "# build=checks enabled (use CMake Release for performance comparisons)\n";
#endif
    std::cout << "# hardware_threads=" << std::thread::hardware_concurrency() << " seed=" << seed << " events=" << count << " initial_depth=" << depth << '\n';
    const auto workload = make_workload(seed, static_cast<std::size_t>(count), static_cast<std::size_t>(depth));
    verify(workload);  // Also warms code paths. Input generation and verification are not timed.
    std::vector<double> timer_cost;
    for (std::size_t i = 0; i < 10000; ++i) {
      const auto start = Clock::now();
      const auto stop = Clock::now();
      timer_cost.push_back(std::chrono::duration<double, std::nano>(stop - start).count());
    }
    std::cout << "# passive=" << workload.passive << " aggressive=" << workload.aggressive << " cancels=" << workload.cancels
              << " fills=" << workload.fills << " depth_min=" << workload.minimum << " depth_max=" << workload.maximum
              << " depth_final=" << workload.final_state.size() << " digest=" << workload.digest << '\n';
    std::cout << "# back_to_back_clock_p50_ns=" << percentile(timer_cost, 0.5) << " back_to_back_clock_p99_ns=" << percentile(timer_cost, 0.99) << '\n';
    std::cout << "# exact event results and final states verified against scan reference\n";
    std::cout << "engine,repeat,events_per_second,p50_ns,p99_ns\n";
    for (std::size_t repeat = 1; repeat <= repeats; ++repeat) {
      if (repeat % 2) { report<Book>(workload, "indexed", repeat); report<ReferenceBook>(workload, "scan_reference", repeat); }
      else { report<ReferenceBook>(workload, "scan_reference", repeat); report<Book>(workload, "indexed", repeat); }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "benchmark error: " << error.what() << '\n';
    return 1;
  }
}
