#pragma once
#include "pipeline.hpp"
#include "queues.hpp"
#include <atomic>
#include <chrono>
#include <limits>

namespace mdp {
struct Frame {
  Packet packet{};
  std::uint64_t generation{};
  std::chrono::steady_clock::time_point received_at{};
};

// Exactly one receiver and one book-owner consumer. Loss signals bypass the
// bounded queue: dropping its final packet must still invalidate the consumer.
template <template<class, std::size_t> class Queue, std::size_t Capacity>
class Ingress {
public:
  using Clock = std::chrono::steady_clock;
private:
  Queue<Frame, Capacity> queue_;
  std::atomic<std::uint64_t> generation_{0};
  std::atomic<std::uint64_t> drops_{0};
  std::uint64_t consumed_generation_{0};
  std::uint64_t stale_frames_{0};
  std::uint64_t aged_frames_{0};
  Clock::duration maximum_age_;
  Clock::time_point last_applied_{};
  bool has_last_applied_{};
  bool freshness_invalidated_{};

  bool aged(Clock::time_point received_at, Clock::time_point now) const {
    // A producer can timestamp after a consumer captures its 'now' argument.
    // This small negative difference is not age and must not expire the frame.
    return now >= received_at && now - received_at >= maximum_age_;
  }
  void invalidate_freshness(Pipeline& pipeline) {
    if (!freshness_invalidated_) pipeline.mark_transport_loss();
    freshness_invalidated_ = true;
    has_last_applied_ = false;
  }
public:
  explicit Ingress(Clock::duration maximum_age = std::chrono::milliseconds(250))
      : maximum_age_(maximum_age) {
    if (maximum_age <= Clock::duration::zero())
      throw std::invalid_argument("positive consumer freshness age required");
  }
  // Producer only. A datagram is dropped, never silently overwritten.
  // This is host receipt time, not exchange/event time or a network-age bound.
  bool push(const Packet& packet, Clock::time_point now = Clock::now()) {
    if (queue_.try_push({packet, generation_.load(std::memory_order_relaxed), now})) return true;
    drops_.fetch_add(1, std::memory_order_relaxed);
    report_loss();
    return false;
  }
  void report_loss() {
    const auto previous = generation_.load(std::memory_order_relaxed);
    if (previous == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("transport loss generation exhausted");
    generation_.store(previous + 1, std::memory_order_release);
  }
  // Consumer only. Call before decisions, including when the queue is empty:
  // a paused receiver cannot keep an old live book fresh. The age of the last
  // accepted frame is measured from receipt, not from when backlog was drained.
  void synchronize_loss(Pipeline& pipeline, Clock::time_point now = Clock::now()) {
    const auto current = generation_.load(std::memory_order_acquire);
    if (current != consumed_generation_) {
      pipeline.mark_transport_loss();
      consumed_generation_ = current;
      has_last_applied_ = false;
    }
    if (has_last_applied_ && aged(last_applied_, now)) invalidate_freshness(pipeline);
  }
  bool consume_one(Pipeline& pipeline, Clock::time_point now = Clock::now()) {
    synchronize_loss(pipeline, now);
    Frame frame;
    if (!queue_.try_pop(frame)) return false;
    synchronize_loss(pipeline, now);
    if (frame.generation != consumed_generation_) {
      ++stale_frames_;
      return true;
    }
    if (aged(frame.received_at, now)) {
      ++aged_frames_;
      invalidate_freshness(pipeline); // Also clears any partial staged snapshot.
      return true;
    }
    const auto outcome = pipeline.on_packet(frame.packet);
    if (outcome != Outcome::rejected) {
      last_applied_ = frame.received_at;
      has_last_applied_ = true;
      freshness_invalidated_ = false;
    } else {
      has_last_applied_ = false;
    }
    synchronize_loss(pipeline, now);
    return true;
  }
  [[nodiscard]] std::uint64_t drops() const { return drops_.load(std::memory_order_relaxed); }
  [[nodiscard]] std::uint64_t stale_frames() const { return stale_frames_; }
  [[nodiscard]] std::uint64_t aged_frames() const { return aged_frames_; }
};
} // namespace mdp
