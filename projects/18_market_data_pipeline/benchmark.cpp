#include "pipeline.hpp"
#include "queues.hpp"
#include "workload.hpp"
#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <latch>
#include <string>
#include <thread>
#ifdef __linux__
#include <cerrno>
#include <cstring>
#include <sched.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;
using Nanoseconds = std::chrono::nanoseconds;
struct Envelope {
  mdp::Packet packet{};
  Clock::time_point scheduled{};
  Clock::time_point attempted{};
};
struct Options { std::size_t events{100000}; int repeats{5}; int producer_cpu{-1}; int consumer_cpu{-1}; };
struct Measurements {
  double seconds{};
  std::uint64_t retries{};
  std::vector<std::int64_t> scheduled;
  std::vector<std::int64_t> handoff;
  std::vector<std::int64_t> service;
};
void require(bool value, const char* reason) { if (!value) throw std::runtime_error(reason); }
void pin_cpu(int cpu) {
  if (cpu < 0) return;
#ifdef __linux__
  if (cpu >= CPU_SETSIZE) throw std::invalid_argument("CPU outside CPU_SETSIZE");
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(static_cast<std::size_t>(cpu), &set);
  if (sched_setaffinity(0, sizeof(set), &set) != 0)
    throw std::runtime_error(std::string("CPU affinity failed: ") + std::strerror(errno));
#else
  throw std::invalid_argument("CPU affinity options require Linux");
#endif
}
std::int64_t elapsed(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration_cast<Nanoseconds>(end - start).count();
}
std::int64_t percentile(const std::vector<std::int64_t>& sorted, std::size_t thousandths) {
  if (sorted.empty()) return 0;
  return sorted[(sorted.size() - 1) * thousandths / 1000];
}
void print_distribution(std::vector<std::int64_t>& values) {
  std::sort(values.begin(), values.end());
  std::cout << ',' << percentile(values, 500) << ',' << percentile(values, 990)
            << ',' << percentile(values, 999) << ',' << (values.empty() ? 0 : values.back());
}
template<template<class, std::size_t> class Queue>
Measurements measure(const std::vector<mdp::Packet>& packets,
                     const std::vector<mdp::Packet>& snapshot,
                     const std::vector<lob::Order>& expected,
                     std::size_t burst, const Options& options) {
  constexpr std::size_t capacity = 256;
  Queue<Envelope, capacity> queue;
  mdp::Pipeline pipeline;
  for (const auto& packet : snapshot) pipeline.on_packet(packet);
  require(pipeline.live(), "benchmark preload failed");
  Measurements result;
  const bool timed = burst != 0;
  if (timed) {
    result.scheduled.resize(packets.size());
    result.handoff.resize(packets.size());
    result.service.resize(packets.size());
  }
  pin_cpu(options.consumer_cpu);
  std::latch ready(1);
  std::atomic<bool> go{false};
  std::atomic<bool> failed{false};
  std::exception_ptr producer_error;
  Clock::time_point started;
  std::uint64_t retries = 0;
  std::jthread producer([&](std::stop_token stop) {
    try {
      pin_cpu(options.producer_cpu);
    } catch (...) {
      producer_error = std::current_exception();
      failed.store(true, std::memory_order_release);
      ready.count_down();
      return;
    }
    ready.count_down();
    while (!go.load(std::memory_order_acquire)) {
      if (stop.stop_requested()) return;
      std::this_thread::yield();
    }
    try {
      for (std::size_t i = 0; i < packets.size(); ++i) {
        Envelope envelope;
        envelope.packet = packets[i];
        if (timed) {
          // Keep target release time when backpressure delays actual publication.
          envelope.scheduled = started + Nanoseconds(static_cast<std::int64_t>(i / burst) * 100000);
          while (Clock::now() < envelope.scheduled) {
            if (stop.stop_requested()) return;
            std::this_thread::yield();
          }
        }
        while (true) {
          if (stop.stop_requested()) return;
          // This is BEFORE the accepted enqueue attempt, not exact queue residence.
          if (timed) envelope.attempted = Clock::now();
          if (queue.try_push(envelope)) break;
          ++retries;
          std::this_thread::yield();
        }
      }
    } catch (...) {
      producer_error = std::current_exception();
      failed.store(true, std::memory_order_release);
    }
  });
  ready.wait();
  if (failed.load(std::memory_order_acquire)) {
    producer.join();
    std::rethrow_exception(producer_error);
  }
  started = Clock::now();
  go.store(true, std::memory_order_release);
  std::uint64_t allowed = 0;
  std::uint64_t empty_polls = 0;
  for (std::size_t i = 0; i < packets.size(); ++i) {
    Envelope envelope;
    while (!queue.try_pop(envelope)) {
      require(!failed.load(std::memory_order_acquire), "producer failed");
      if (++empty_polls % 65536 == 0)
        require(Clock::now() - started < std::chrono::seconds(30), "benchmark deadline");
      std::this_thread::yield();
    }
    const auto service_started = timed ? Clock::now() : Clock::time_point{};
    pipeline.on_packet(envelope.packet);
    if (mdp::evaluate_risk(pipeline, {}, {lob::Side::buy, 10001, 1}, {}) == mdp::RiskDecision::accepted)
      ++allowed;
    if (timed) {
      const auto completed = Clock::now();
      result.scheduled[i] = elapsed(envelope.scheduled, completed);
      result.handoff[i] = elapsed(envelope.attempted, completed);
      result.service[i] = elapsed(service_started, completed);
    }
  }
  const auto finished = Clock::now();
  producer.join();
  if (producer_error) std::rethrow_exception(producer_error);
  // Final state and outcomes are verified outside every timed pass.
  require(pipeline.live() && pipeline.counters().rejected == 0, "unhealthy benchmark path");
  require(pipeline.counters().accepted == snapshot.size() + packets.size(), "lost/duplicate packets");
  require(pipeline.expected_sequence() == snapshot.size() + packets.size() + 1, "sequence mismatch");
  require(pipeline.book().snapshot() == expected, "sequential final-state mismatch");
  require(allowed == packets.size(), "risk check count mismatch");
  require(queue.empty(), "undrained benchmark queue");
  pipeline.book().check_invariants();
  result.seconds = std::chrono::duration<double>(finished - started).count();
  result.retries = retries;
  return result;
}
Options parse(int argc, char** argv) {
  Options result;
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 >= argc) throw std::invalid_argument("missing argument value");
    const std::string name = argv[i];
    const std::string value = argv[i + 1];
    int number{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
      throw std::invalid_argument("invalid integer argument");
    if (name == "--events" && number >= 5 && number <= 2000000) result.events = static_cast<std::size_t>(number);
    else if (name == "--repeats" && number >= 1 && number <= 20) result.repeats = number;
    else if (name == "--producer-cpu" && number >= 0) result.producer_cpu = number;
    else if (name == "--consumer-cpu" && number >= 0) result.consumer_cpu = number;
    else throw std::invalid_argument("unknown argument or value outside bounds");
  }
  if (result.producer_cpu >= 0 && result.producer_cpu == result.consumer_cpu)
    throw std::invalid_argument("choose distinct producer/consumer CPUs");
  if ((result.producer_cpu >= 0) != (result.consumer_cpu >= 0))
    throw std::invalid_argument("supply both CPU affinity options together");
  return result;
}
}
int main(int argc, char** argv) {
  try {
    const auto options = parse(argc, argv);
    const auto snapshot = mdp::snapshot_packets(1, 1024);
    const auto packets = mdp::workload_packets(options.events);
    mdp::Pipeline reference;
    for (const auto& packet : snapshot) reference.on_packet(packet);
    for (const auto& packet : packets) reference.on_packet(packet);
    require(reference.live() && reference.counters().rejected == 0, "invalid generated workload");
    const auto expected = reference.book().snapshot();
    std::vector<std::int64_t> clock_cost(10000);
    for (auto& sample : clock_cost) {
      const auto begin = Clock::now();
      sample = elapsed(begin, Clock::now());
    }
    std::sort(clock_cost.begin(), clock_cost.end());
    std::cerr << "clock_pair_ns,p50=" << percentile(clock_cost, 500)
              << ",p99=" << percentile(clock_cost, 990)
              << ",max=" << clock_cost.back()
              << "; steady=" << Clock::is_steady << "; producer_cpu=" << options.producer_cpu
              << "; consumer_cpu=" << options.consumer_cpu << '\n';
    // Unreported warm-up per variant; generated workload, threads, preload, results
    // allocation, reference verification, sorting and formatting are excluded.
    const auto warm_packets = mdp::workload_packets(1000);
    mdp::Pipeline warm_reference;
    for (const auto& packet : snapshot) warm_reference.on_packet(packet);
    for (const auto& packet : warm_packets) warm_reference.on_packet(packet);
    const auto warm_expected = warm_reference.book().snapshot();
    (void)measure<mdp::SpscQueue>(warm_packets, snapshot, warm_expected, 0, options);
    (void)measure<mdp::MutexQueue>(warm_packets, snapshot, warm_expected, 0, options);
    std::cout << "queue,mode,repeat,events,capacity,burst,period_ns,seconds,events_per_second,full_retries"
                 ",scheduled_p50_ns,scheduled_p99_ns,scheduled_p999_ns,scheduled_max_ns"
                 ",attempt_p50_ns,attempt_p99_ns,attempt_p999_ns,attempt_max_ns"
                 ",service_p50_ns,service_p99_ns,service_p999_ns,service_max_ns\n";
    for (const std::size_t burst : {std::size_t{0}, std::size_t{32}, std::size_t{256}}) {
      for (int repetition = 1; repetition <= options.repeats; ++repetition) {
        for (int turn = 0; turn < 2; ++turn) {
          const bool spsc = (repetition + turn) % 2 == 1;
          auto result = spsc ? measure<mdp::SpscQueue>(packets, snapshot, expected, burst, options)
                             : measure<mdp::MutexQueue>(packets, snapshot, expected, burst, options);
          std::cout << (spsc ? "spsc" : "mutex") << ',' << (burst == 0 ? "throughput" : "latency")
                    << ',' << repetition << ',' << packets.size() << ",256," << burst << ','
                    << (burst == 0 ? 0 : 100000) << ',' << std::setprecision(9) << result.seconds
                    << ',' << static_cast<double>(packets.size()) / result.seconds << ',' << result.retries;
          print_distribution(result.scheduled);
          print_distribution(result.handoff);
          print_distribution(result.service);
          std::cout << '\n';
        }
      }
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
