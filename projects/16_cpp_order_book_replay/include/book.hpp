#pragma once
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <list>
#include <map>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lob {
using Id = std::uint64_t;
using Price = std::int64_t;  // Positive integer ticks; no floating-point equality.
using Quantity = std::uint64_t;
enum class Side { buy, sell };
struct Order {
  Id id;
  Side side;
  Price price;
  Quantity quantity;
  bool operator==(const Order&) const = default;
};
struct Trade {
  Id maker;
  Id taker;
  Price price;
  Quantity quantity;
  bool operator==(const Trade&) const = default;
};
inline void validate(const Order& order) {
  if (order.id == 0 || order.price <= 0 || order.quantity == 0 ||
      (order.side != Side::buy && order.side != Side::sell))
    throw std::invalid_argument("id, tick price and quantity must be positive; side must be buy/sell");
}
inline bool crosses(const Order& taker, Price maker_price) {
  return taker.side == Side::buy ? taker.price >= maker_price : taker.price <= maker_price;
}
inline bool canonical_less(const Order& left, const Order& right) {
  if (left.side != right.side) return left.side == Side::buy;
  return left.side == Side::buy ? left.price > right.price : left.price < right.price;
}

// Single-threaded price/time priority engine. No network or wall-clock dependency.
class Book {
  using Queue = std::list<Order>;
  using Levels = std::map<Price, Queue>;
  struct Location {
    Side side;
    Levels::iterator level;
    Queue::iterator order;
  };
  Levels bids_;
  Levels asks_;
  std::unordered_map<Id, Location> ids_;

public:
  Book() = default;
  // Locations contain iterators into this instance; accidental copies are unsafe.
  Book(const Book&) = delete;
  Book& operator=(const Book&) = delete;
  Book(Book&&) = delete;
  Book& operator=(Book&&) = delete;

  std::vector<Trade> add(Order taker) {
    validate(taker);
    if (ids_.contains(taker.id)) throw std::invalid_argument("duplicate active order id");
    std::vector<Trade> trades;
    auto& opposite = taker.side == Side::buy ? asks_ : bids_;
    while (taker.quantity && !opposite.empty()) {
      auto level = taker.side == Side::buy ? opposite.begin() : std::prev(opposite.end());
      if (!crosses(taker, level->first)) break;
      auto& queue = level->second;
      auto& maker = queue.front();
      const auto quantity = std::min(taker.quantity, maker.quantity);
      trades.push_back({maker.id, taker.id, maker.price, quantity});
      taker.quantity -= quantity;
      maker.quantity -= quantity;
      if (maker.quantity == 0) {
        ids_.erase(maker.id);
        queue.pop_front();
        if (queue.empty()) opposite.erase(level);
      }
    }
    if (taker.quantity) {
      auto& same = taker.side == Side::buy ? bids_ : asks_;
      auto [level, created] = same.try_emplace(taker.price);
      try {
        level->second.push_back(taker);
      } catch (...) {
        if (created) same.erase(level);
        throw;
      }
      try {
        ids_.emplace(taker.id, Location{taker.side, level, std::prev(level->second.end())});
      } catch (...) {
        level->second.pop_back();
        if (created) same.erase(level);
        throw;
      }
    }
    return trades;
  }
  bool cancel(Id id) {
    const auto found = ids_.find(id);
    if (found == ids_.end()) return false;
    const auto location = found->second;
    auto& levels = location.side == Side::buy ? bids_ : asks_;
    location.level->second.erase(location.order);
    if (location.level->second.empty()) levels.erase(location.level);
    ids_.erase(found);
    return true;
  }
  [[nodiscard]] std::size_t size() const { return ids_.size(); }
  [[nodiscard]] bool contains(Id id) const { return ids_.contains(id); }
  [[nodiscard]] std::optional<Price> best_bid() const {
    return bids_.empty() ? std::nullopt : std::optional<Price>(bids_.rbegin()->first);
  }
  [[nodiscard]] std::optional<Price> best_ask() const {
    return asks_.empty() ? std::nullopt : std::optional<Price>(asks_.begin()->first);
  }
  [[nodiscard]] std::vector<Order> snapshot() const {
    std::vector<Order> result;
    result.reserve(size());
    for (auto it = bids_.rbegin(); it != bids_.rend(); ++it)
      result.insert(result.end(), it->second.begin(), it->second.end());
    for (const auto& [price, queue] : asks_) {
      (void)price;
      result.insert(result.end(), queue.begin(), queue.end());
    }
    return result;
  }
  // Deliberately expensive diagnostics; run after every operation in tests, outside timing.
  void check_invariants() const {
    std::unordered_set<Id> seen;
    const auto inspect = [&](const Levels& levels, Side side) {
      for (auto level = levels.begin(); level != levels.end(); ++level) {
        if (level->second.empty()) throw std::logic_error("empty price level");
        for (auto order = level->second.begin(); order != level->second.end(); ++order) {
          validate(*order);
          if (order->side != side || order->price != level->first || !seen.insert(order->id).second)
            throw std::logic_error("invalid level membership or duplicate id");
          const auto location = ids_.find(order->id);
          if (location == ids_.end() || location->second.side != side ||
              location->second.level != level || location->second.order != order)
            throw std::logic_error("order index mismatch");
        }
      }
    };
    inspect(bids_, Side::buy);
    inspect(asks_, Side::sell);
    if (seen.size() != ids_.size()) throw std::logic_error("orphan order index entry");
    if (best_bid() && best_ask() && *best_bid() >= *best_ask())
      throw std::logic_error("crossed resting book");
  }
};
}  // namespace lob
