#include "book.hpp"
#include "reference.hpp"
#include <array>
#include <iostream>
#include <limits>
#include <random>
#include <string>

using namespace lob;
namespace {
std::size_t checks = 0;
void require(bool condition, const std::string& message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);  // Active in Release; no assert().
}
template<class Action> void rejects(Action action, const std::string& message) {
  bool threw = false;
  try { action(); } catch (const std::invalid_argument&) { threw = true; }
  require(threw, message);
}
void examples() {
  Book book;
  require(!book.best_bid() && !book.best_ask() && !book.cancel(0), "empty book");
  book.add({1, Side::sell, 101, 5});
  book.add({2, Side::sell, 101, 7});
  book.add({3, Side::sell, 102, 4});
  require(book.add({4, Side::buy, 101, 8}) ==
          std::vector<Trade>{{1, 4, 101, 5}, {2, 4, 101, 3}}, "FIFO, partial fills and maker price");
  require(book.snapshot() == std::vector<Order>{{2, Side::sell, 101, 4}, {3, Side::sell, 102, 4}}, "partial maker remains first");
  require(book.cancel(2) && !book.cancel(2), "cancel active and repeated cancel");
  require(book.add({5, Side::buy, 105, 10}) == std::vector<Trade>{{3, 5, 102, 4}}, "price improvement");
  require(book.snapshot() == std::vector<Order>{{5, Side::buy, 105, 6}}, "unfilled taker rests at its limit");
  book.add({6, Side::buy, 104, 2});
  require(book.add({7, Side::sell, 103, 7}) == std::vector<Trade>{{5, 7, 105, 6}, {6, 7, 104, 1}}, "sell sweep best price first");
  book.check_invariants();
  require(book.cancel(6) && book.size() == 0, "last order removes its level");
  require(book.add({1, Side::buy, 1, 1}).empty(), "inactive ids may be reused");

  const auto before = book.snapshot();
  for (const Order invalid : std::array<Order, 6>{{
      {0, Side::buy, 1, 1}, {2, Side::buy, 0, 1}, {2, Side::buy, -1, 1},
      {2, Side::buy, 1, 0}, {2, static_cast<Side>(2), 1, 1}, {1, Side::sell, 1, 1}}}) {
    rejects([&] { book.add(invalid); }, "invalid input rejected before matching");
    require(book.snapshot() == before, "rejected input must leave book unchanged");
  }
  book.check_invariants();

  Book extremes;
  extremes.add({1, Side::sell, std::numeric_limits<Price>::max(), std::numeric_limits<Quantity>::max()});
  require(extremes.add({2, Side::buy, std::numeric_limits<Price>::max(), std::numeric_limits<Quantity>::max()}) ==
          std::vector<Trade>{{1, 2, std::numeric_limits<Price>::max(), std::numeric_limits<Quantity>::max()}}, "integer limits do not overflow");
  extremes.check_invariants();

  Book indexed;
  ReferenceBook simple;
  for (Id id = 1; id <= 2048; ++id) {
    const Order order{id, Side::buy, 90 + static_cast<Price>(id % 64), 10};
    indexed.add(order);
    simple.add(order);
  }
  for (Id id = 2; id <= 2048; id += 2) {
    require(indexed.cancel(id) == simple.cancel(id), "cancellation after index rehash");
    indexed.check_invariants();
  }
  require(indexed.snapshot() == simple.snapshot(), "middle-order cancellation retains FIFO");
  require(indexed.add({9999, Side::sell, 1, 10240}) == simple.add({9999, Side::sell, 1, 10240}), "all-level sweep after cancellations");
  indexed.check_invariants();
  require(indexed.size() == 0, "all levels removed after sweep");
}

void differential() {
  const std::array<std::uint64_t, 8> seeds{1, 7, 42, 99, 2027, 65537, 1234567, 987654321};
  for (auto seed : seeds) {
    std::mt19937_64 random(seed);
    Book book;
    ReferenceBook reference;
    Id next_id = 1;
    for (std::size_t event = 0; event < 5000; ++event) {
      try {
        const auto before = book.snapshot();
        const auto kind = random() % 100;
        if (kind < 68) {
          const Order order{next_id++, random() % 2 ? Side::buy : Side::sell,
                            90 + static_cast<Price>(random() % 21), 1 + random() % 30};
          const auto trades = book.add(order);
          require(trades == reference.add(order), "differential trade sequence");
          Quantity filled = 0;
          for (const auto& trade : trades) {
            const auto maker = std::find_if(before.begin(), before.end(), [&](const auto& x) { return x.id == trade.maker; });
            require(maker != before.end(), "maker existed before event");
            require(maker->side != order.side && maker->price == trade.price && crosses(order, trade.price), "trade side/price contract");
            require(trade.taker == order.id && trade.quantity > 0 && trade.quantity <= maker->quantity, "trade id/quantity contract");
            filled += trade.quantity;
          }
          Quantity resting = 0;
          for (const auto& current : book.snapshot()) if (current.id == order.id) resting = current.quantity;
          require(filled + resting == order.quantity, "taker quantity conservation");
          Quantity before_total = 0;
          Quantity after_total = 0;
          for (const auto& old : before) before_total += old.quantity;
          for (const auto& current : book.snapshot()) after_total += current.quantity;
          require(after_total + 2 * filled == before_total + order.quantity, "whole-book quantity conservation");
        } else if (kind < 92) {
          const auto id = !before.empty() && random() % 4 ? before[random() % before.size()].id : next_id + random() % 100;
          require(book.cancel(id) == reference.cancel(id), "differential cancellation");
        } else if (!before.empty()) {
          const auto duplicate = before[random() % before.size()];
          rejects([&] { book.add(duplicate); }, "engine duplicate rejection");
          rejects([&] { reference.add(duplicate); }, "reference duplicate rejection");
          require(book.snapshot() == before, "duplicate leaves state unchanged");
        }
        book.check_invariants();
        require(book.snapshot() == reference.snapshot(), "differential state including FIFO");
      } catch (const std::exception& error) {
        throw std::runtime_error("seed=" + std::to_string(seed) + " event=" + std::to_string(event) + ": " + error.what());
      }
    }
  }
}
}  // namespace
int main() {
  try {
    examples();
    differential();
    std::cout << "PASS: " << checks << " checks; 40000 randomized events across 8 seeds; Release checks active\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
