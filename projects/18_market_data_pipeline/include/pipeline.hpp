#pragma once
#include "feed.hpp"
#include <limits>
#include <memory>

namespace mdp {
enum class Outcome { accepted, snapshot_activated, rejected };
struct Counters {
  std::uint64_t received{};
  std::uint64_t accepted{};
  std::uint64_t rejected{};
  std::uint64_t malformed{};
  std::uint64_t sequence_gaps{};
  std::uint64_t transport_losses{};
  std::uint64_t snapshots_completed{};
};

// Confined to the consumer thread. A queue transports packets, never Book objects.
class Pipeline {
  std::unique_ptr<lob::Book> active_{std::make_unique<lob::Book>()};
  std::unique_ptr<lob::Book> staging_;
  std::uint64_t epoch_{};
  std::uint64_t expected_sequence_{1};
  bool live_{};
  Counters counters_{};

  static void increment(std::uint64_t& value) noexcept {
    if (value != std::numeric_limits<std::uint64_t>::max()) ++value;
  }
  void fail_closed() noexcept {
    live_ = false;
    staging_.reset();
  }
  Outcome reject() noexcept {
    increment(counters_.rejected);
    fail_closed();
    return Outcome::rejected;
  }

public:
  [[nodiscard]] bool live() const noexcept { return live_; }
  // The diagnostic book may be old when !live(). Always gate actions on live().
  [[nodiscard]] const lob::Book& book() const noexcept { return *active_; }
  [[nodiscard]] std::uint64_t epoch() const noexcept { return epoch_; }
  [[nodiscard]] std::uint64_t expected_sequence() const noexcept { return expected_sequence_; }
  [[nodiscard]] const Counters& counters() const noexcept { return counters_; }

  void mark_transport_loss() noexcept {
    increment(counters_.transport_losses);
    fail_closed();
  }

  Outcome on_packet(std::span<const std::byte> packet) {
    increment(counters_.received);
    Message message;
    try {
      message = decode(packet);
    } catch (const std::invalid_argument&) {
      increment(counters_.malformed);
      return reject();
    } catch (...) {
      fail_closed();
      throw;
    }
    try {
      if (message.kind == Kind::SnapshotBegin) {
        // A new generation replaces a failed/partial recovery, never rolls back.
        if (message.epoch <= epoch_ || message.sequence != 1) return reject();
        live_ = false;
        staging_ = std::make_unique<lob::Book>();
        epoch_ = message.epoch;
        expected_sequence_ = 2;
        increment(counters_.accepted);
        return Outcome::accepted;
      }
      if (message.epoch != epoch_) return reject();
      if (message.sequence != expected_sequence_) {
        increment(counters_.sequence_gaps);
        return reject();
      }
      // UINT64_MAX is an exhaustion boundary, not a wrapped next sequence of 0.
      // Require a higher-epoch snapshot before this value would be consumed.
      if (message.sequence == std::numeric_limits<std::uint64_t>::max()) return reject();

      Outcome result = Outcome::accepted;
      if (staging_) {
        if (message.kind == Kind::SnapshotOrder) {
          const auto opposite = message.side == lob::Side::buy
              ? staging_->best_ask() : staging_->best_bid();
          const lob::Order order{message.id, message.side, message.price, message.quantity};
          if (opposite && lob::crosses(order, *opposite)) return reject();
          // Snapshots describe resting orders, so matching here is forbidden.
          staging_->add(order);
        } else if (message.kind == Kind::SnapshotEnd) {
          active_.swap(staging_);
          staging_.reset();
          live_ = true;
          increment(counters_.snapshots_completed);
          result = Outcome::snapshot_activated;
        } else {
          return reject();
        }
      } else if (live_) {
        switch (message.kind) {
        case Kind::Add:
          // These are synthetic matching-engine order inputs, not L2 updates.
          active_->add({message.id, message.side, message.price, message.quantity});
          break;
        case Kind::Cancel:
          if (!active_->cancel(message.id)) return reject();
          break;
        case Kind::Heartbeat:
          break;
        default:
          return reject();
        }
      } else {
        return reject();
      }
      ++expected_sequence_;
      increment(counters_.accepted);
      return result;
    } catch (const std::invalid_argument&) {
      return reject();
    } catch (...) {
      // Allocation errors may leave a partially applied live matching event.
      // Preserve diagnostics, invalidate the book, and let the caller abort.
      fail_closed();
      throw;
    }
  }
};

struct OrderRequest {
  lob::Side side;
  lob::Price price;
  lob::Quantity quantity;
};
struct RiskLimits {
  lob::Quantity max_order_quantity{100};
  std::uint64_t max_absolute_position{1'000};
  std::uint64_t max_price_deviation_ticks{10};
};
struct RiskState {
  std::int64_t position{};
  lob::Quantity open_buy_quantity{};
  lob::Quantity open_sell_quantity{};
};
enum class RiskDecision {
  accepted, stale_book, invalid_order, invalid_limits, missing_market,
  price_band, position_limit
};

// Pure conservative pretrade check. The caller supplies outstanding quantities;
// this function neither reserves capacity nor models an order-management system.
inline RiskDecision evaluate_risk(bool live, const lob::Book& book,
                                  const RiskState& state, const OrderRequest& request,
                                  const RiskLimits& limits) {
  if (!live) return RiskDecision::stale_book;
  const auto signed_max = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  if (limits.max_order_quantity == 0 || limits.max_absolute_position > signed_max)
    return RiskDecision::invalid_limits;
  if (request.quantity == 0 || request.price <= 0 ||
      request.quantity > limits.max_order_quantity ||
      (request.side != lob::Side::buy && request.side != lob::Side::sell))
    return RiskDecision::invalid_order;
  const auto bid = book.best_bid();
  const auto ask = book.best_ask();
  if (!bid || !ask || *bid >= *ask) return RiskDecision::missing_market;

  // Never add bid+ask or double a price. The exact midpoint can be half a tick.
  const auto spread = *ask - *bid;  // Positive prices make this subtraction safe.
  const auto midpoint_floor = *bid + spread / 2;
  if (request.price <= midpoint_floor) {
    const auto distance = static_cast<std::uint64_t>(midpoint_floor - request.price);
    if (distance > limits.max_price_deviation_ticks ||
        (spread % 2 != 0 && distance == limits.max_price_deviation_ticks))
      return RiskDecision::price_band;
  } else if (static_cast<std::uint64_t>(request.price - midpoint_floor) >
             limits.max_price_deviation_ticks) {
    return RiskDecision::price_band;
  }

  // Magnitude is defined even for INT64_MIN; no signed addition/subtraction of
  // position and outstanding quantities, and no cancellation across sides.
  const auto magnitude = state.position >= 0 ? static_cast<std::uint64_t>(state.position)
      : static_cast<std::uint64_t>(-(state.position + 1)) + 1U;
  if (magnitude > limits.max_absolute_position) return RiskDecision::position_limit;
  const auto long_capacity = state.position >= 0
      ? limits.max_absolute_position - magnitude : limits.max_absolute_position + magnitude;
  const auto short_capacity = state.position <= 0
      ? limits.max_absolute_position - magnitude : limits.max_absolute_position + magnitude;
  if (state.open_buy_quantity > long_capacity || state.open_sell_quantity > short_capacity)
    return RiskDecision::position_limit;
  if (request.side == lob::Side::buy) {
    if (request.quantity > long_capacity - state.open_buy_quantity)
      return RiskDecision::position_limit;
  } else if (request.quantity > short_capacity - state.open_sell_quantity) {
    return RiskDecision::position_limit;
  }
  return RiskDecision::accepted;
}

inline RiskDecision evaluate_risk(const Pipeline& pipeline, const RiskState& state,
                                  const OrderRequest& request, const RiskLimits& limits) {
  return evaluate_risk(pipeline.live(), pipeline.book(), state, request, limits);
}
}  // namespace mdp
