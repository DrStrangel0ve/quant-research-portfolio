#pragma once
#include "feed.hpp"
#include <vector>

namespace mdp {
inline std::vector<Packet> snapshot_packets(std::uint64_t epoch, std::size_t depth = 2) {
  std::vector<Packet> result;
  result.push_back(encode({Kind::SnapshotBegin, 1, epoch}));
  for (std::size_t i = 0; i < depth; ++i) {
    const bool buy = i % 2 == 0;
    const auto level = static_cast<lob::Price>((i / 2) % 32);
    result.push_back(encode({Kind::SnapshotOrder, i + 2, epoch, i + 1,
                            buy ? 10000 - level : 10002 + level, 10,
                            buy ? lob::Side::buy : lob::Side::sell}));
  }
  result.push_back(encode({Kind::SnapshotEnd, depth + 2, epoch}));
  return result;
}
inline std::vector<Packet> workload_packets(std::size_t count, std::size_t depth = 1024) {
  std::vector<Packet> result;
  result.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto sequence = static_cast<std::uint64_t>(depth + 3 + i);
    const auto id = static_cast<std::uint64_t>(depth + 1 + i / 5);
    switch (i % 5) {
      case 0:
        result.push_back(encode({Kind::Add, sequence, 1, id, 9990, 3, lob::Side::buy}));
        break;
      case 1:
        result.push_back(encode({Kind::Cancel, sequence, 1, id}));
        break;
      case 2:
        result.push_back(encode({Kind::Add, sequence, 1, id, 10012, 3, lob::Side::sell}));
        break;
      case 3:
        result.push_back(encode({Kind::Cancel, sequence, 1, id}));
        break;
      default:
        result.push_back(encode({Kind::Heartbeat, sequence, 1}));
    }
  }
  return result;
}
} // namespace mdp
