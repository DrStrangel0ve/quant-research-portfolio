#include "book.hpp"
#include <charconv>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>

namespace {
template<class Integer> Integer integer(std::string_view value) {
  Integer result{};
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc{} || end != value.data() + value.size()) throw std::invalid_argument("invalid integer field");
  return result;
}
std::vector<std::string_view> fields(const std::string& line) {
  std::vector<std::string_view> result;
  std::size_t start = 0;
  for (;;) {
    const auto comma = line.find(',', start);
    result.push_back(std::string_view(line).substr(start, comma == std::string::npos ? comma : comma - start));
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return result;
}
}  // namespace
int main(int argc, char** argv) {
  if (argc != 2) { std::cerr << "usage: book_replay events.csv\n"; return 2; }
  std::ifstream input(argv[1]);
  if (!input) { std::cerr << "cannot open input\n"; return 2; }
  lob::Book book;
  std::size_t line_number = 0, events = 0, trades = 0;
  try {
    for (std::string line; std::getline(input, line);) {
      ++line_number;
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line.empty() || line.front() == '#') continue;
      const auto row = fields(line);
      if (row.size() == 2 && row[0] == "C") {
        const auto id = integer<lob::Id>(row[1]);
        std::cout << "CANCEL," << id << ',' << book.cancel(id) << '\n';
      } else if (row.size() == 5 && row[0] == "A" && (row[2] == "B" || row[2] == "S")) {
        const lob::Order order{integer<lob::Id>(row[1]), row[2] == "B" ? lob::Side::buy : lob::Side::sell,
                               integer<lob::Price>(row[3]), integer<lob::Quantity>(row[4])};
        for (const auto& trade : book.add(order)) {
          ++trades;
          std::cout << "TRADE," << trade.maker << ',' << trade.taker << ',' << trade.price << ',' << trade.quantity << '\n';
        }
      } else throw std::invalid_argument("expected A,id,B|S,positive_ticks,positive_quantity or C,id");
      ++events;
      book.check_invariants();
    }
    if (input.bad()) throw std::runtime_error("input read failure");
    for (const auto& order : book.snapshot())
      std::cout << "REST," << order.id << ',' << (order.side == lob::Side::buy ? 'B' : 'S') << ',' << order.price << ',' << order.quantity << '\n';
    std::cout << "events=" << events << " trades=" << trades << " resting=" << book.size() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "line " << line_number << ": " << error.what() << '\n';
    return 1;
  }
}
