#pragma once
#include <chrono>
#include <stdexcept>

namespace mdp {
// Receiver-owned, caller-injected monotonic time permits deterministic tests.
class Watchdog {
public:
  using Clock = std::chrono::steady_clock;
private:
  Clock::duration timeout_;
  Clock::time_point last_{};
  bool observed_{};
  bool latched_{};
public:
  explicit Watchdog(Clock::duration timeout) : timeout_(timeout) {
    if (timeout <= Clock::duration::zero()) throw std::invalid_argument("positive watchdog timeout required");
  }
  void observe(Clock::time_point now) { last_ = now; observed_ = true; latched_ = false; }
  bool expired(Clock::time_point now) {
    if (!observed_ || latched_ || now - last_ < timeout_) return false;
    latched_ = true;
    return true;
  }
};
} // namespace mdp
