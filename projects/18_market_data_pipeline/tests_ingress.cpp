#include "ingress.hpp"
#include "watchdog.hpp"
#include "workload.hpp"
#include <atomic>
#include <iostream>
#include <string>
#include <thread>

namespace {
void require(bool condition, const char* reason) {
  if (!condition) throw std::runtime_error(reason);
}

template<template<class, std::size_t> class Queue>
void check_freshness() {
  using Ingress = mdp::Ingress<Queue, 4>;
  using Clock = typename Ingress::Clock;
  using namespace std::chrono_literals;
  const typename Clock::time_point zero{};
  bool rejected_zero = false;
  bool rejected_negative = false;
  try { Ingress invalid(0ms); } catch (const std::invalid_argument&) { rejected_zero = true; }
  try { Ingress invalid(-1ms); } catch (const std::invalid_argument&) { rejected_negative = true; }
  require(rejected_zero && rejected_negative, "freshness age must be positive");
  Ingress ingress(250ms);
  mdp::Pipeline pipeline;
  const auto snapshot = [&](std::uint64_t epoch, typename Clock::time_point received_at) {
    for (const auto& packet : mdp::snapshot_packets(epoch)) {
      require(ingress.push(packet, received_at), "fresh snapshot enqueue");
      require(ingress.consume_one(pipeline, received_at), "fresh snapshot dequeue");
    }
  };
  snapshot(1, zero);
  ingress.synchronize_loss(pipeline, zero + 249ms);
  require(pipeline.live(), "book fresh before consumer deadline");
  ingress.synchronize_loss(pipeline, zero + 250ms);
  require(!pipeline.live(), "empty queue and paused receiver cannot preserve live book");
  require(pipeline.counters().transport_losses == 1, "consumer freshness invalidates once");
  ingress.synchronize_loss(pipeline, zero + 251ms);
  require(pipeline.counters().transport_losses == 1, "freshness loss latched until accepted recovery");
  require(ingress.push(mdp::encode({mdp::Kind::Heartbeat, 5, 1}), zero + 251ms), "fresh old-epoch heartbeat");
  ingress.consume_one(pipeline, zero + 251ms);
  require(!pipeline.live(), "fresh heartbeat cannot heal expired book");

  const auto partial = mdp::snapshot_packets(2);
  require(ingress.push(partial[0], zero + 260ms), "new partial snapshot begin");
  ingress.consume_one(pipeline, zero + 260ms);
  require(ingress.push(partial[1], zero + 260ms), "new partial first order");
  ingress.consume_one(pipeline, zero + 260ms);
  const auto received_before_age = pipeline.counters().received;
  require(ingress.push(partial[2], zero + 260ms), "aged staged order queued");
  ingress.consume_one(pipeline, zero + 510ms);
  require(!pipeline.live() && ingress.aged_frames() == 1, "aged frame invalidates partial recovery");
  require(pipeline.counters().received == received_before_age, "aged frame rejected before parsing or applying");
  require(ingress.push(partial[3], zero + 511ms), "fresh end after aged staged order");
  ingress.consume_one(pipeline, zero + 511ms);
  require(!pipeline.live(), "fresh snapshot end cannot activate expired partial state");
  snapshot(2, zero + 512ms);
  require(!pipeline.live(), "same epoch cannot recover aged staged state");
  snapshot(3, zero + 520ms);
  require(pipeline.live() && pipeline.epoch() == 3, "complete fresh higher-epoch snapshot recovers");

  // Dequeuing at t=765 must not change a t=760 receipt into t=765 freshness.
  require(ingress.push(mdp::encode({mdp::Kind::Heartbeat, 5, 3}), zero + 760ms), "timely heartbeat queued");
  ingress.consume_one(pipeline, zero + 765ms);
  require(pipeline.live(), "fresh accepted heartbeat renews receipt deadline");
  ingress.synchronize_loss(pipeline, zero + 1009ms);
  require(pipeline.live(), "receipt age below exact boundary");
  ingress.synchronize_loss(pipeline, zero + 1010ms);
  require(!pipeline.live(), "consumer processing time does not extend receipt freshness");

  Ingress backlog(250ms);
  mdp::Pipeline untouched;
  for (const auto& packet : mdp::snapshot_packets(1)) require(backlog.push(packet, zero), "old backlog enqueue");
  while (backlog.consume_one(untouched, zero + 250ms)) {}
  require(backlog.aged_frames() == 4 && untouched.counters().received == 0,
          "entire old snapshot discarded without parsing");
  require(!untouched.live() && untouched.epoch() == 0, "old backlog cannot activate a book");
  require(untouched.counters().transport_losses == 1, "old backlog latches one freshness loss");

  // A producer may timestamp after the consumer captured its optional 'now'.
  require(backlog.push(mdp::snapshot_packets(2)[0], zero + 301ms), "later producer clock capture");
  backlog.consume_one(untouched, zero + 300ms);
  require(untouched.epoch() == 2, "negative apparent age from overlapping clock capture is not stale");
}

template<template<class, std::size_t> class Queue>
void check_concurrent_recovery() {
  using Ingress = mdp::Ingress<Queue, 4>;
  using Clock = typename Ingress::Clock;
  const typename Clock::time_point zero{};
  Ingress ingress;
  mdp::Pipeline pipeline;
  for (const auto& packet : mdp::snapshot_packets(1)) {
    require(ingress.push(packet, zero), "concurrent setup enqueue");
    ingress.consume_one(pipeline, zero);
  }
  require(pipeline.live(), "concurrent setup live");
  const auto interrupted_snapshot = mdp::snapshot_packets(2);
  const auto recovery = mdp::snapshot_packets(3);
  std::atomic<unsigned> phase{0};
  std::atomic<bool> probing_loss{false};
  std::atomic<bool> abort{false};
  std::string producer_error;
  std::string consumer_error;
  const auto deadline = Clock::now() + std::chrono::seconds(30);
  const auto running = [&] {
    if (Clock::now() >= deadline) abort.store(true, std::memory_order_relaxed);
    return !abort.load(std::memory_order_relaxed);
  };
  const auto wait_phase = [&](unsigned target) {
    while (phase.load(std::memory_order_acquire) < target) {
      if (!running()) return false;
      std::this_thread::yield();
    }
    return true;
  };
  std::jthread producer([&] {
    try {
      require(ingress.push(interrupted_snapshot[0], zero), "concurrent snapshot begin enqueue");
      phase.store(1, std::memory_order_release);
      if (!wait_phase(2)) return;
      while (!probing_loss.load(std::memory_order_acquire)) {
        if (!running()) return;
        std::this_thread::yield();
      }
      // Consumer keeps checking the out-of-band generation while producer fills
      // the four slots. Deliberately dropping the final packet fences the partial
      // snapshot even when no subsequent datagram arrives.
      for (std::size_t index = 1; index < interrupted_snapshot.size(); ++index)
        require(ingress.push(interrupted_snapshot[index], zero), "concurrent fill remainder");
      require(ingress.push(mdp::encode({mdp::Kind::Heartbeat, 5, 2}), zero), "concurrent fill final slot");
      require(!ingress.push(mdp::encode({mdp::Kind::Heartbeat, 6, 2}), zero), "concurrent terminal overflow");
      phase.store(3, std::memory_order_release);
      if (!wait_phase(4)) return;
      // Actual producer/consumer queue operations run concurrently here. Four
      // packets fit even if the consumer has not started, so recovery adds no
      // scheduler-dependent overflow or retry loss to the experiment.
      for (const auto& packet : recovery)
        require(ingress.push(packet, zero), "concurrent fresh recovery enqueue");
      phase.store(5, std::memory_order_release);
    } catch (const std::exception& error) {
      producer_error = error.what();
      abort.store(true, std::memory_order_relaxed);
    }
  });
  std::jthread consumer([&] {
    try {
      if (!wait_phase(1)) return;
      require(ingress.consume_one(pipeline, zero), "concurrent begin dequeue");
      require(!pipeline.live() && pipeline.epoch() == 2, "consumer staging second epoch");
      phase.store(2, std::memory_order_release);
      probing_loss.store(true, std::memory_order_release);
      while (phase.load(std::memory_order_acquire) < 3) {
        if (!running()) return;
        ingress.synchronize_loss(pipeline, zero);
        std::this_thread::yield();
      }
      while (ingress.consume_one(pipeline, zero)) {}
      require(!pipeline.live(), "concurrent overflow destroys partial recovery");
      require(ingress.stale_frames() == 4 && ingress.drops() == 1, "concurrent generation accounting");
      phase.store(4, std::memory_order_release);
      for (std::size_t index = 0; index < recovery.size(); ++index) {
        while (!ingress.consume_one(pipeline, zero)) {
          if (!running()) return;
          std::this_thread::yield();
        }
      }
      if (!wait_phase(5)) return;
      ingress.synchronize_loss(pipeline, zero);
      require(pipeline.live() && pipeline.epoch() == 3, "concurrent higher epoch activates");
      require(pipeline.counters().snapshots_completed == 2, "only complete initial and final snapshots activate");
      pipeline.book().check_invariants();
    } catch (const std::exception& error) {
      consumer_error = error.what();
      abort.store(true, std::memory_order_relaxed);
    }
  });
  producer.join();
  consumer.join();
  if (!producer_error.empty()) throw std::runtime_error(producer_error);
  if (!consumer_error.empty()) throw std::runtime_error(consumer_error);
  require(!abort.load(), "concurrent phase watchdog did not expire");
  require(phase.load() == 5 && pipeline.live(), "concurrent recovery completed");
}

template<template<class, std::size_t> class Queue>
void check_ingress() {
  mdp::Ingress<Queue, 4> ingress;
  mdp::Pipeline pipeline;
  for (const auto& packet : mdp::snapshot_packets(1)) {
    require(ingress.push(packet), "initial enqueue");
    require(ingress.consume_one(pipeline), "initial dequeue");
  }
  require(pipeline.live(), "initial snapshot live");
  const auto recovery = mdp::snapshot_packets(2);
  for (const auto& packet : recovery) require(ingress.push(packet), "fill four slots");
  require(!ingress.push(mdp::encode({mdp::Kind::Heartbeat, 5, 2})), "final packet dropped");
  while (ingress.consume_one(pipeline)) {}
  require(!pipeline.live(), "pre-loss queued snapshot must not heal terminal loss");
  require(ingress.stale_frames() == 4 && ingress.drops() == 1, "loss accounting");
  require(pipeline.epoch() == 1, "discarded recovery not activated");
  for (const auto& packet : mdp::snapshot_packets(3)) {
    require(ingress.push(packet), "post-loss enqueue");
    ingress.consume_one(pipeline);
  }
  require(pipeline.live() && pipeline.epoch() == 3, "new recovery heals loss");
  ingress.report_loss();
  require(!ingress.consume_one(pipeline), "empty queue stays empty");
  require(!pipeline.live(), "empty queue still delivers loss fence");

  mdp::Watchdog watchdog(std::chrono::milliseconds(10));
  const mdp::Watchdog::Clock::time_point zero{};
  require(!watchdog.expired(zero + std::chrono::hours(1)), "unstarted watchdog");
  watchdog.observe(zero);
  require(!watchdog.expired(zero + std::chrono::milliseconds(9)), "before deadline");
  require(watchdog.expired(zero + std::chrono::milliseconds(10)), "exact deadline");
  require(!watchdog.expired(zero + std::chrono::milliseconds(11)), "latch single loss");
  watchdog.observe(zero + std::chrono::milliseconds(12));
  require(!watchdog.expired(zero + std::chrono::milliseconds(21)), "new observation resets deadline");
  require(watchdog.expired(zero + std::chrono::milliseconds(22)), "second silence detected");
}
}
int main() {
  try {
    check_ingress<mdp::SpscQueue>();
    check_ingress<mdp::MutexQueue>();
    check_freshness<mdp::SpscQueue>();
    check_freshness<mdp::MutexQueue>();
    check_concurrent_recovery<mdp::SpscQueue>();
    check_concurrent_recovery<mdp::MutexQueue>();
    std::cout << "ingress terminal loss, receipt freshness, partial recovery, concurrent fencing and watchdog checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
