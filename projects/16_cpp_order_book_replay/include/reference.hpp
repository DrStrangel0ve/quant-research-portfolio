#pragma once
#include "book.hpp"

namespace lob {
// Intentionally simple independent storage/matching algorithm, for differential tests
// and a transparent complexity baseline. Global arrival list, linear price/id scans.
class ReferenceBook {
  std::list<Order> orders_;
public:
  std::vector<Trade> add(Order taker) {
    validate(taker);
    if (std::any_of(orders_.begin(), orders_.end(), [&](const auto& x) { return x.id == taker.id; }))
      throw std::invalid_argument("duplicate active order id");
    std::vector<Trade> trades;
    while (taker.quantity) {
      auto best = orders_.end();
      for (auto it = orders_.begin(); it != orders_.end(); ++it) {
        if (it->side == taker.side) continue;
        if (taker.side == Side::buy ? it->price > taker.price : it->price < taker.price) continue;
        if (best == orders_.end() || (taker.side == Side::buy ? it->price < best->price : it->price > best->price))
          best = it;  // Strict improvement preserves the earliest arrival at equal prices.
      }
      if (best == orders_.end()) break;
      const auto quantity = std::min(best->quantity, taker.quantity);
      trades.push_back({best->id, taker.id, best->price, quantity});
      best->quantity -= quantity;
      taker.quantity -= quantity;
      if (!best->quantity) orders_.erase(best);
    }
    if (taker.quantity) orders_.push_back(taker);
    return trades;
  }
  bool cancel(Id id) {
    const auto found = std::find_if(orders_.begin(), orders_.end(), [&](const auto& x) { return x.id == id; });
    if (found == orders_.end()) return false;
    orders_.erase(found);
    return true;
  }
  [[nodiscard]] std::vector<Order> snapshot() const {
    std::vector<Order> result(orders_.begin(), orders_.end());
    std::stable_sort(result.begin(), result.end(), canonical_less);
    return result;
  }
};
}  // namespace lob
