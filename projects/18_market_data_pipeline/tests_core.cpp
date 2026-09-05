#include "pipeline.hpp"
#include <array>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {
using namespace mdp;
using lob::Side;
std::size_t checks{};
void require(bool condition, const std::string& message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);  // Also active in Release.
}
template<class Action> void rejects(Action action, const std::string& message) {
  bool threw = false;
  try { action(); } catch (const std::invalid_argument&) { threw = true; }
  require(threw, message);
}
Message control(Kind kind, std::uint64_t sequence, std::uint64_t epoch) {
  return {kind, sequence, epoch, 0, 0, 0, Side::buy};
}
Outcome deliver(Pipeline& pipeline, const Message& message) {
  return pipeline.on_packet(encode(message));
}
void initialize(Pipeline& pipeline, std::uint64_t epoch) {
  require(deliver(pipeline, control(Kind::SnapshotBegin, 1, epoch)) == Outcome::accepted,
          "snapshot begin");
  require(deliver(pipeline, {Kind::SnapshotOrder, 2, epoch, 1, 99, 5, Side::buy}) ==
              Outcome::accepted, "snapshot bid");
  require(deliver(pipeline, {Kind::SnapshotOrder, 3, epoch, 2, 101, 7, Side::sell}) ==
              Outcome::accepted, "snapshot ask");
  require(deliver(pipeline, control(Kind::SnapshotEnd, 4, epoch)) ==
              Outcome::snapshot_activated, "snapshot activates");
  require(pipeline.live() && pipeline.expected_sequence() == 5 && pipeline.epoch() == epoch,
          "initialized epoch, sequence and liveness");
  pipeline.book().check_invariants();
}

void codec_tests() {
  const Message known{Kind::Add, 0x0102030405060708ULL, 0x1112131415161718ULL,
      0x2122232425262728ULL, 0x3132333435363738LL, 0x4142434445464748ULL, Side::sell};
  const std::array<unsigned char, 48> expected{
      0x4d, 0x44, 0x50, 0x31, 0x01, 0x04, 0x02, 0x00,
      0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
      0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
      0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,
      0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38,
      0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48};
  const auto packet = encode(known);
  for (std::size_t i = 0; i != packet.size(); ++i)
    require(std::to_integer<unsigned char>(packet[i]) == expected[i], "known big-endian bytes");
  require(decode(packet) == known, "known frame decode");
  for (std::size_t size = 0; size != packet.size(); ++size)
    rejects([&] { (void)decode(std::span<const std::byte>(packet).first(size)); },
            "every truncated length rejected");
  std::array<std::byte, 49> oversized{};
  std::copy(packet.begin(), packet.end(), oversized.begin());
  rejects([&] { (void)decode(oversized); }, "oversized packet rejected");

  const auto corrupt = [&](std::size_t offset, std::byte value) {
    auto changed = packet;
    changed[offset] = value;
    rejects([&] { (void)decode(changed); }, "corrupt fixed header rejected");
  };
  for (std::size_t i = 0; i != 4; ++i) corrupt(i, std::byte{0});
  corrupt(4, std::byte{2});
  corrupt(5, std::byte{0});
  corrupt(5, std::byte{7});
  corrupt(6, std::byte{0});
  corrupt(6, std::byte{3});
  corrupt(7, std::byte{1});
  for (const auto offset : std::array<std::size_t, 5>{8, 16, 24, 32, 40}) {
    auto changed = packet;
    put_u64(changed, offset, 0);
    rejects([&] { (void)decode(changed); }, "required zero field rejected");
  }
  auto overflow = packet;
  put_u64(overflow, 32, std::numeric_limits<std::uint64_t>::max());
  rejects([&] { (void)decode(overflow); }, "unsigned tick overflow rejected");
  put_u64(overflow, 32, std::uint64_t{1} << 63U);
  rejects([&] { (void)decode(overflow); }, "INT64_MAX + 1 tick rejected");

  for (const auto kind : {Kind::SnapshotBegin, Kind::SnapshotEnd, Kind::Heartbeat}) {
    const auto message = control(kind, 1, 1);
    require(decode(encode(message)) == message, "control roundtrip");
    for (const auto offset : std::array<std::size_t, 4>{6, 24, 32, 40}) {
      auto invalid = encode(message);
      invalid[offset] = std::byte{1};
      rejects([&] { (void)decode(invalid); }, "control fields must be canonical");
    }
  }
  const Message cancel{Kind::Cancel, 9, 4, 99, 0, 0, Side::buy};
  require(decode(encode(cancel)) == cancel, "cancel roundtrip");
  for (const auto offset : std::array<std::size_t, 3>{6, 32, 40}) {
    auto invalid = encode(cancel);
    invalid[offset] = std::byte{1};
    rejects([&] { (void)decode(invalid); }, "cancel unused fields canonical");
  }
  auto missing_id = encode(cancel);
  put_u64(missing_id, 24, 0);
  rejects([&] { (void)decode(missing_id); }, "cancel id required");
  for (const auto invalid : std::array<Message, 7>{
      Message{Kind::Add, 1, 1, 1, -1, 1, Side::buy},
      Message{Kind::Add, 1, 1, 1, 1, 1, static_cast<Side>(9)},
      Message{Kind::Add, 1, 1, 1, 1, 0, Side::buy},
      Message{Kind::Heartbeat, 1, 1, 1, 0, 0, Side::buy},
      Message{Kind::Heartbeat, 1, 1, 0, 0, 0, Side::sell},
      Message{Kind::Cancel, 1, 1, 1, 1, 0, Side::buy},
      Message{static_cast<Kind>(255), 1, 1, 0, 0, 0, Side::buy}})
    rejects([&] { (void)encode(invalid); }, "invalid local message rejected");

  std::mt19937_64 random(20260905);
  for (std::size_t i = 0; i != 10'000; ++i) {
    const Message message{i % 2 == 0 ? Kind::Add : Kind::SnapshotOrder,
        random() | 1U, random() | 1U, random() | 1U,
        static_cast<lob::Price>((random() >> 1U) | 1U), random() | 1U,
        i % 3 == 0 ? Side::buy : Side::sell};
    require(decode(encode(message)) == message, "random valid frame roundtrip");
  }
  const Message extremes{Kind::Add, std::numeric_limits<std::uint64_t>::max(),
      std::numeric_limits<std::uint64_t>::max(), std::numeric_limits<std::uint64_t>::max(),
      std::numeric_limits<lob::Price>::max(), std::numeric_limits<lob::Quantity>::max(), Side::buy};
  require(decode(encode(extremes)) == extremes, "all representable unsigned maxima survive codec");
}

void recovery_tests() {
  Pipeline pipeline;
  require(!pipeline.live() && pipeline.book().size() == 0 && pipeline.epoch() == 0,
          "book exists but starts unusable");
  require(deliver(pipeline, {Kind::Add, 1, 1, 99, 99, 1, Side::buy}) == Outcome::rejected,
          "no live delta before snapshot");
  initialize(pipeline, 1);
  require(deliver(pipeline, {Kind::Add, 5, 1, 3, 101, 3, Side::buy}) == Outcome::accepted,
          "live aggressive add matches");
  require(pipeline.book().snapshot() == std::vector<lob::Order>{
      {1, Side::buy, 99, 5}, {2, Side::sell, 101, 4}}, "live add exact residuals");
  require(deliver(pipeline, {Kind::Cancel, 6, 1, 1, 0, 0, Side::buy}) == Outcome::accepted,
          "valid live cancellation");
  require(deliver(pipeline, control(Kind::Heartbeat, 7, 1)) == Outcome::accepted,
          "heartbeat advances sequence");
  const auto before_gap = pipeline.book().snapshot();
  require(deliver(pipeline, {Kind::Add, 9, 1, 9, 90, 1, Side::buy}) == Outcome::rejected,
          "gap rejected before application");
  require(!pipeline.live() && pipeline.book().snapshot() == before_gap &&
          pipeline.expected_sequence() == 8 && pipeline.counters().sequence_gaps == 1,
          "gap retains diagnostic book and expected sequence");
  require(deliver(pipeline, control(Kind::Heartbeat, 8, 1)) == Outcome::rejected,
          "late missing packet cannot make stale book live");
  require(deliver(pipeline, control(Kind::SnapshotBegin, 1, 1)) == Outcome::rejected,
          "same epoch snapshot rollback forbidden");
  require(deliver(pipeline, control(Kind::SnapshotBegin, 1, 2)) == Outcome::accepted,
          "higher epoch begins recovery");
  require(deliver(pipeline, {Kind::SnapshotOrder, 2, 2, 10, 98, 11, Side::buy}) ==
              Outcome::accepted, "recovery staged offside");
  require(!pipeline.live() && pipeline.book().snapshot() == before_gap,
          "partial snapshot cannot replace diagnostic active book");
  require(deliver(pipeline, control(Kind::SnapshotEnd, 3, 2)) == Outcome::snapshot_activated,
          "explicit snapshot end activates entire stage");
  require(pipeline.book().snapshot() == std::vector<lob::Order>{{10, Side::buy, 98, 11}},
          "replacement removes old orders");
  require(pipeline.counters().snapshots_completed == 2, "snapshot counter");
  pipeline.book().check_invariants();

  // A duplicate or stale packet also fails closed; accepting it would hide feed disorder.
  require(deliver(pipeline, control(Kind::Heartbeat, 3, 2)) == Outcome::rejected &&
          !pipeline.live(), "duplicate sequence invalidates live state");
  initialize(pipeline, 3);
  const auto before_stale = pipeline.book().snapshot();
  require(deliver(pipeline, {Kind::Cancel, 5, 2, 1, 0, 0, Side::buy}) == Outcome::rejected &&
          !pipeline.live() && pipeline.book().snapshot() == before_stale,
          "stale epoch cannot mutate the active book");
  initialize(pipeline, 4);
  require(deliver(pipeline, {Kind::Cancel, 5, 4, 999, 0, 0, Side::buy}) == Outcome::rejected &&
          !pipeline.live(), "unknown cancellation reveals lost state");
  initialize(pipeline, 5);
  const auto before_duplicate = pipeline.book().snapshot();
  require(deliver(pipeline, {Kind::Add, 5, 5, 1, 105, 100, Side::buy}) == Outcome::rejected &&
          !pipeline.live() && pipeline.book().snapshot() == before_duplicate,
          "duplicate active id rejected before potential matching");
  initialize(pipeline, 6);
  auto bad_packet = encode(control(Kind::Heartbeat, 5, 6));
  bad_packet[7] = std::byte{1};
  require(pipeline.on_packet(bad_packet) == Outcome::rejected && !pipeline.live() &&
          pipeline.counters().malformed == 1, "malformed packet closes live state");
  require(pipeline.on_packet(std::span<const std::byte>{}) == Outcome::rejected &&
          pipeline.counters().malformed == 2, "empty packet closes state without throwing");
  initialize(pipeline, 7);
  pipeline.mark_transport_loss();
  require(!pipeline.live() && pipeline.counters().transport_losses == 1,
          "out-of-band transport loss closes state");
  require(deliver(pipeline, control(Kind::SnapshotBegin, 1, 7)) == Outcome::rejected,
          "transport loss requires higher epoch recovery");
  require(pipeline.counters().received == pipeline.counters().accepted + pipeline.counters().rejected,
          "every nonexceptional packet is accounted exactly once");
}

void snapshot_rejection_tests() {
  Pipeline pipeline;
  initialize(pipeline, 1);
  const auto original = pipeline.book().snapshot();
  require(deliver(pipeline, control(Kind::SnapshotBegin, 1, 2)) == Outcome::accepted,
          "begin new snapshot");
  require(deliver(pipeline, {Kind::SnapshotOrder, 2, 2, 1, 100, 1, Side::buy}) ==
              Outcome::accepted, "first resting order");
  require(deliver(pipeline, {Kind::SnapshotOrder, 3, 2, 2, 100, 1, Side::sell}) ==
              Outcome::rejected, "crossed snapshot rejected instead of matching");
  require(!pipeline.live() && pipeline.book().snapshot() == original, "crossing retains old book");
  require(deliver(pipeline, control(Kind::SnapshotEnd, 3, 2)) == Outcome::rejected,
          "failed stage cannot be completed");
  require(deliver(pipeline, control(Kind::SnapshotBegin, 1, 3)) == Outcome::accepted,
          "new epoch after rejected snapshot");
  require(deliver(pipeline, {Kind::SnapshotOrder, 2, 3, 1, 100, 1, Side::buy}) ==
              Outcome::accepted, "snapshot first id");
  require(deliver(pipeline, {Kind::SnapshotOrder, 3, 3, 1, 99, 1, Side::buy}) ==
              Outcome::rejected, "snapshot duplicate id rejected");
  require(deliver(pipeline, control(Kind::SnapshotBegin, 1, 4)) == Outcome::accepted,
          "new epoch starts stage");
  pipeline.mark_transport_loss();
  require(deliver(pipeline, control(Kind::SnapshotEnd, 2, 4)) == Outcome::rejected,
          "transport loss destroys partial snapshot");

  for (const auto kind : {Kind::Add, Kind::Cancel, Kind::Heartbeat}) {
    const auto epoch = pipeline.epoch() + 1;
    require(deliver(pipeline, control(Kind::SnapshotBegin, 1, epoch)) == Outcome::accepted,
            "begin snapshot to test interleaving");
    Message wrong = control(kind, 2, epoch);
    if (kind == Kind::Add) wrong = {kind, 2, epoch, 100, 100, 1, Side::buy};
    if (kind == Kind::Cancel) wrong.id = 100;
    require(deliver(pipeline, wrong) == Outcome::rejected && !pipeline.live(),
            "only snapshot rows/end permitted during recovery");
  }
  auto epoch = pipeline.epoch() + 1;
  require(deliver(pipeline, control(Kind::SnapshotBegin, 2, epoch)) == Outcome::rejected,
          "snapshot must begin at sequence one");
  require(deliver(pipeline, control(Kind::SnapshotBegin, 1, epoch)) == Outcome::accepted,
          "invalid begin did not accept new generation");
  require(deliver(pipeline, {Kind::SnapshotOrder, 3, epoch, 99, 100, 1, Side::buy}) ==
              Outcome::rejected, "snapshot sequence gap rejected");
  ++epoch;
  require(deliver(pipeline, control(Kind::SnapshotBegin, 1, epoch)) == Outcome::accepted,
          "begin partial stage");
  require(deliver(pipeline, {Kind::SnapshotOrder, 2, epoch, 99, 100, 1, Side::buy}) ==
              Outcome::accepted, "partial stage row");
  ++epoch;
  require(deliver(pipeline, control(Kind::SnapshotBegin, 1, epoch)) == Outcome::accepted,
          "higher epoch may replace partial stage");
  require(deliver(pipeline, control(Kind::SnapshotEnd, 2, epoch)) == Outcome::snapshot_activated &&
          pipeline.book().size() == 0, "empty snapshot is complete and discards partial orders");
  require(deliver(pipeline, control(Kind::SnapshotEnd, 3, epoch)) == Outcome::rejected,
          "unsolicited snapshot end invalidates state");

  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  require(deliver(pipeline, control(Kind::SnapshotBegin, 1, maximum)) == Outcome::accepted,
          "last representable epoch can be accepted");
  require(deliver(pipeline, control(Kind::SnapshotEnd, 2, maximum)) == Outcome::snapshot_activated,
          "last epoch can become live");
  require(deliver(pipeline, {Kind::Add, maximum, maximum, 123, 100, 1, Side::buy}) ==
              Outcome::rejected && pipeline.expected_sequence() == 3 && pipeline.book().size() == 0,
          "maximum sequence never wraps or applies a gap event");
  require(deliver(pipeline, control(Kind::SnapshotBegin, 1, 1)) == Outcome::rejected &&
          pipeline.epoch() == maximum, "epoch never wraps to one");
  pipeline.book().check_invariants();
}

void risk_tests() {
  Pipeline pipeline;
  const RiskLimits limits{10, 20, 2};
  require(evaluate_risk(pipeline, {}, {Side::buy, 100, 1}, limits) == RiskDecision::stale_book,
          "uninitialized book blocks risk requests");
  initialize(pipeline, 1);
  const auto check = [&](RiskState state, OrderRequest order) {
    return evaluate_risk(pipeline, state, order, limits);
  };
  require(check({}, {Side::buy, 100, 10}) == RiskDecision::accepted, "normal order permitted");
  require(check({}, {Side::sell, 100, 0}) == RiskDecision::invalid_order, "zero quantity rejected");
  require(check({}, {Side::buy, 100, 11}) == RiskDecision::invalid_order, "oversized order rejected");
  require(check({}, {Side::buy, 0, 1}) == RiskDecision::invalid_order, "zero price rejected");
  require(check({}, {static_cast<Side>(7), 100, 1}) == RiskDecision::invalid_order,
          "bad side rejected");
  require(check({}, {Side::buy, 98, 1}) == RiskDecision::accepted &&
          check({}, {Side::sell, 102, 1}) == RiskDecision::accepted,
          "price band includes its boundary");
  require(check({}, {Side::buy, 97, 1}) == RiskDecision::price_band &&
          check({}, {Side::sell, 103, 1}) == RiskDecision::price_band, "outside price band blocked");
  require(check({10, 7, 0}, {Side::buy, 100, 3}) == RiskDecision::accepted &&
          check({10, 7, 0}, {Side::buy, 100, 4}) == RiskDecision::position_limit,
          "existing long position and outstanding buys consume capacity");
  require(check({-10, 0, 7}, {Side::sell, 100, 3}) == RiskDecision::accepted &&
          check({-10, 0, 7}, {Side::sell, 100, 4}) == RiskDecision::position_limit,
          "short position and outstanding sells consume capacity");
  require(check({0, 20, 20}, {Side::buy, 100, 1}) == RiskDecision::position_limit,
          "opposite open orders must not net to free risk capacity");
  require(check({0, 21, 0}, {Side::sell, 100, 1}) == RiskDecision::position_limit,
          "preexisting unsafe outstanding exposure rejected on either side");
  // Caller-maintained open quantity is explicit: cancellation/release creates
  // capacity only once the caller updates its state; the pure gate mutates none.
  const RiskState open{10, 10, 0};
  require(check(open, {Side::buy, 100, 1}) == RiskDecision::position_limit &&
          check({10, 9, 0}, {Side::buy, 100, 1}) == RiskDecision::accepted &&
          open.open_buy_quantity == 10, "risk gate does not implicitly reserve or release");
  lob::Book empty;
  require(evaluate_risk(true, empty, {}, {Side::buy, 100, 1}, limits) == RiskDecision::missing_market,
          "two-sided book required");
  empty.add({1, Side::buy, 99, 1});
  require(evaluate_risk(true, empty, {}, {Side::buy, 100, 1}, limits) == RiskDecision::missing_market,
          "one-sided book rejected");
  empty.add({2, Side::sell, 100, 1});
  require(evaluate_risk(true, empty, {}, {Side::buy, 99, 1}, {10, 20, 0}) ==
              RiskDecision::price_band &&
          evaluate_risk(true, empty, {}, {Side::buy, 100, 1}, {10, 20, 0}) ==
              RiskDecision::price_band, "zero band around half-tick midpoint admits no integer price");
  require(evaluate_risk(true, empty, {}, {Side::buy, 99, 1}, {10, 20, 1}) ==
              RiskDecision::accepted &&
          evaluate_risk(true, empty, {}, {Side::buy, 100, 1}, {10, 20, 1}) ==
              RiskDecision::accepted, "half-tick midpoint checked symmetrically");
  require(evaluate_risk(true, empty, {}, {Side::buy, 98, 1}, {10, 20, 1}) ==
              RiskDecision::price_band &&
          evaluate_risk(true, empty, {}, {Side::buy, 101, 1}, {10, 20, 1}) ==
              RiskDecision::price_band, "outer half-tick band boundaries excluded");
  require(evaluate_risk(pipeline, {}, {Side::buy, 100, 1}, {0, 20, 2}) ==
              RiskDecision::invalid_limits, "zero per-order limit classified explicitly");
  lob::Book low_price;
  low_price.add({1, Side::buy, 1, 1});
  low_price.add({2, Side::sell, 2, 1});
  require(evaluate_risk(true, low_price, {}, {Side::buy, 1, 1}, {10, 20, 0}) ==
              RiskDecision::price_band &&
          evaluate_risk(true, low_price, {}, {Side::sell, 2, 1}, {10, 20, 0}) ==
              RiskDecision::price_band, "lowest valid half-tick midpoint is exact");
  require(evaluate_risk(true, low_price, {}, {Side::buy, 1, 1}, {10, 20, 1}) ==
              RiskDecision::accepted &&
          evaluate_risk(true, low_price, {}, {Side::sell, 2, 1}, {10, 20, 1}) ==
              RiskDecision::accepted, "low-price band cannot underflow");

  const auto maximum = std::numeric_limits<std::int64_t>::max();
  const auto umaximum = std::numeric_limits<std::uint64_t>::max();
  lob::Book extreme_book;
  extreme_book.add({1, Side::buy, maximum - 2, 1});
  extreme_book.add({2, Side::sell, maximum, 1});
  const RiskLimits extreme_limits{umaximum, static_cast<std::uint64_t>(maximum), umaximum};
  require(evaluate_risk(true, extreme_book, {}, {Side::buy, maximum, 1}, extreme_limits) ==
              RiskDecision::accepted, "large midpoint and band do not overflow");
  require(evaluate_risk(true, extreme_book, {std::numeric_limits<std::int64_t>::min(), 0, 0},
          {Side::buy, maximum, 1}, extreme_limits) == RiskDecision::position_limit,
          "INT64_MIN position safely rejected");
  require(evaluate_risk(true, extreme_book, {-maximum, 0, 0},
          {Side::buy, maximum, umaximum - 1}, extreme_limits) == RiskDecision::accepted,
          "full short-to-long range without intermediate signed overflow");
  require(evaluate_risk(true, extreme_book, {-maximum, 0, 0},
          {Side::buy, maximum, umaximum}, extreme_limits) == RiskDecision::position_limit,
          "one past full position range rejected without wrapping");
  require(evaluate_risk(true, extreme_book, {maximum, 0, 0},
          {Side::sell, maximum, umaximum - 1}, extreme_limits) == RiskDecision::accepted,
          "full long-to-short range without signed overflow");
  require(evaluate_risk(true, extreme_book, {0, umaximum, umaximum},
          {Side::buy, maximum, 1}, extreme_limits) == RiskDecision::position_limit,
          "huge outstanding quantities never wrap");
  require(evaluate_risk(true, extreme_book, {}, {Side::buy, maximum, 1},
          {10, umaximum, 2}) == RiskDecision::invalid_limits, "unrepresentable position limit rejected");

  // Independent small-integer reference enumerates all fill-side extrema.
  // Here ordinary signed arithmetic is bounded, avoiding the implementation's
  // remaining-capacity formulation and testing both directions of risk.
  for (std::int64_t position = -6; position <= 6; ++position) {
    for (std::uint64_t buys = 0; buys <= 6; ++buys) {
      for (std::uint64_t sells = 0; sells <= 6; ++sells) {
        for (std::uint64_t quantity = 1; quantity <= 3; ++quantity) {
          for (const auto side : {Side::buy, Side::sell}) {
            const auto long_position = position + static_cast<std::int64_t>(buys) +
                (side == Side::buy ? static_cast<std::int64_t>(quantity) : 0);
            const auto short_position = position - static_cast<std::int64_t>(sells) -
                (side == Side::sell ? static_cast<std::int64_t>(quantity) : 0);
            const bool expected = position >= -5 && position <= 5 &&
                long_position <= 5 && short_position >= -5;
            const auto actual = evaluate_risk(pipeline, {position, buys, sells},
                {side, 100, quantity}, {3, 5, 1});
            require((actual == RiskDecision::accepted) == expected,
                    "exhaustive risk decision vs bounded signed reference");
          }
        }
      }
    }
  }
  pipeline.mark_transport_loss();
  require(evaluate_risk(pipeline, {}, {Side::buy, 100, 1}, limits) == RiskDecision::stale_book,
          "retained diagnostic prices cannot authorize stale trading");
}
}  // namespace

int main() {
  try {
    codec_tests();
    recovery_tests();
    snapshot_rejection_tests();
    risk_tests();
    std::cout << "core checks passed: " << checks << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "core test failure after " << checks << " checks: " << error.what() << '\n';
    return 1;
  }
}
