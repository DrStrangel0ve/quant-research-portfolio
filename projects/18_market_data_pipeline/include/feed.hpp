#pragma once
#include "book.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace mdp {
// Educational order-event protocol; this is not an exchange's wire format.
enum class Kind : std::uint8_t {
  SnapshotBegin = 1, SnapshotOrder = 2, SnapshotEnd = 3,
  Add = 4, Cancel = 5, Heartbeat = 6
};
struct Message {
  Kind kind{};
  std::uint64_t sequence{};
  std::uint64_t epoch{};
  std::uint64_t id{};
  lob::Price price{};
  lob::Quantity quantity{};
  // Non-order messages use the default value here and encode side=0 on wire.
  lob::Side side{lob::Side::buy};
  bool operator==(const Message&) const = default;
};
using Packet = std::array<std::byte, 48>;

inline bool is_order(Kind kind) {
  return kind == Kind::SnapshotOrder || kind == Kind::Add;
}

inline void validate_message(const Message& message) {
  if (message.sequence == 0 || message.epoch == 0)
    throw std::invalid_argument("sequence and epoch must be positive");
  switch (message.kind) {
  case Kind::SnapshotOrder:
  case Kind::Add:
    lob::validate({message.id, message.side, message.price, message.quantity});
    break;
  case Kind::Cancel:
    if (message.id == 0 || message.price != 0 || message.quantity != 0 ||
        message.side != lob::Side::buy)
      throw std::invalid_argument("cancel must contain only an order id");
    break;
  case Kind::SnapshotBegin:
  case Kind::SnapshotEnd:
  case Kind::Heartbeat:
    if (message.id != 0 || message.price != 0 || message.quantity != 0 ||
        message.side != lob::Side::buy)
      throw std::invalid_argument("control message has noncanonical fields");
    break;
  default:
    throw std::invalid_argument("unknown message kind");
  }
}

inline void put_u64(Packet& packet, std::size_t offset, std::uint64_t value) {
  for (std::size_t i = 0; i != 8; ++i)
    packet[offset + i] = static_cast<std::byte>((value >> (56U - 8U * i)) & 0xffU);
}
inline std::uint64_t get_u64(std::span<const std::byte> bytes, std::size_t offset) {
  std::uint64_t result = 0;
  for (std::size_t i = 0; i != 8; ++i)
    result = (result << 8U) | std::to_integer<std::uint64_t>(bytes[offset + i]);
  return result;
}

inline Packet encode(const Message& message) {
  validate_message(message);
  Packet packet{};
  packet[0] = std::byte{'M'};
  packet[1] = std::byte{'D'};
  packet[2] = std::byte{'P'};
  packet[3] = std::byte{'1'};
  packet[4] = std::byte{1};
  packet[5] = static_cast<std::byte>(message.kind);
  packet[6] = is_order(message.kind)
      ? (message.side == lob::Side::buy ? std::byte{1} : std::byte{2}) : std::byte{0};
  put_u64(packet, 8, message.sequence);
  put_u64(packet, 16, message.epoch);
  put_u64(packet, 24, message.id);
  put_u64(packet, 32, static_cast<std::uint64_t>(message.price));
  put_u64(packet, 40, message.quantity);
  return packet;
}

inline Message decode(std::span<const std::byte> bytes) {
  if (bytes.size() != Packet{}.size())
    throw std::invalid_argument("packet must contain exactly 48 bytes");
  if (bytes[0] != std::byte{'M'} || bytes[1] != std::byte{'D'} ||
      bytes[2] != std::byte{'P'} || bytes[3] != std::byte{'1'} ||
      bytes[4] != std::byte{1} || bytes[7] != std::byte{0})
    throw std::invalid_argument("bad magic, version or reserved byte");
  Message message{};
  message.kind = static_cast<Kind>(std::to_integer<std::uint8_t>(bytes[5]));
  const auto side = std::to_integer<unsigned>(bytes[6]);
  if (is_order(message.kind)) {
    if (side != 1 && side != 2) throw std::invalid_argument("invalid order side");
    message.side = side == 1 ? lob::Side::buy : lob::Side::sell;
  } else if (side != 0) {
    throw std::invalid_argument("non-order wire side must be zero");
  }
  message.sequence = get_u64(bytes, 8);
  message.epoch = get_u64(bytes, 16);
  message.id = get_u64(bytes, 24);
  const auto price = get_u64(bytes, 32);
  if (price > static_cast<std::uint64_t>(std::numeric_limits<lob::Price>::max()))
    throw std::invalid_argument("tick price cannot fit signed price type");
  message.price = static_cast<lob::Price>(price);
  message.quantity = get_u64(bytes, 40);
  validate_message(message);
  return message;
}
}  // namespace mdp
