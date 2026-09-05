#include "ingress.hpp"
#include "udp.hpp"
#include "watchdog.hpp"
#include "workload.hpp"
#include <iostream>
#include <thread>

int main() {
  try {
    using Clock = std::chrono::steady_clock;
    mdp::UdpSocket receiver_socket;
    mdp::UdpSocket sender_socket;
    const auto port = receiver_socket.port();
    mdp::Ingress<mdp::SpscQueue, 64> ingress;
    mdp::Pipeline pipeline;
    std::atomic<std::uint64_t> datagrams{0};
    std::atomic<bool> receiver_failed{false};
    std::exception_ptr receiver_error;
    std::jthread receiver([&](std::stop_token stop) {
      try {
        mdp::Watchdog watchdog(std::chrono::milliseconds(250));
        while (!stop.stop_requested()) {
          mdp::Packet packet{};
          const auto received = receiver_socket.receive(packet, 20);
          const auto now = Clock::now();
          // Even ready traffic must not conceal a long receiver scheduling gap.
          if (watchdog.expired(now)) ingress.report_loss();
          if (received.status == mdp::ReceiveStatus::timeout) {
            continue;
          }
          watchdog.observe(now);
          if (received.status == mdp::ReceiveStatus::truncated || received.bytes != packet.size())
            ingress.report_loss();
          else
            ingress.push(packet);
          datagrams.fetch_add(1, std::memory_order_release);
        }
      } catch (...) {
        receiver_error = std::current_exception();
        try { ingress.report_loss(); } catch (...) { /* Fatal flag also blocks decisions. */ }
        receiver_failed.store(true, std::memory_order_release);
      }
    });
    auto require = [](bool condition, const char* reason) {
      if (!condition) throw std::runtime_error(reason);
    };
    std::uint64_t sent = 0;
    auto transmit = [&](std::span<const std::byte> bytes) {
      sender_socket.send_to_loopback(bytes, port);
      ++sent;
      const auto deadline = Clock::now() + std::chrono::seconds(5);
      while (datagrams.load(std::memory_order_acquire) < sent) {
        ingress.consume_one(pipeline);
        require(!receiver_failed.load(std::memory_order_acquire), "receiver failed");
        require(Clock::now() < deadline, "loopback receive deadline");
        std::this_thread::yield();
      }
      while (ingress.consume_one(pipeline)) {}
    };
    const auto decision = [&] {
      require(!receiver_failed.load(std::memory_order_acquire), "receiver failed before risk check");
      ingress.synchronize_loss(pipeline);
      return mdp::evaluate_risk(pipeline, {}, {lob::Side::buy, 10001, 1}, {});
    };
    require(decision() == mdp::RiskDecision::stale_book, "startup must reject");
    for (const auto& packet : mdp::snapshot_packets(1)) transmit(packet);
    require(decision() == mdp::RiskDecision::accepted, "snapshot allows illustrative risk check");
    transmit(mdp::encode({mdp::Kind::Heartbeat, 6, 1})); // seq5 deliberately missing
    require(decision() == mdp::RiskDecision::stale_book, "sequence gap must reject");
    transmit(mdp::encode({mdp::Kind::Heartbeat, 5, 1}));
    require(!pipeline.live(), "late delta cannot heal gap");
    for (const auto& packet : mdp::snapshot_packets(2)) transmit(packet);
    require(pipeline.live(), "second epoch recovers");
    const auto valid_prefix = mdp::encode({mdp::Kind::Heartbeat, 5, 2});
    std::vector<std::byte> oversized(valid_prefix.begin(), valid_prefix.end());
    oversized.push_back(std::byte{0});
    transmit(oversized);
    require(!pipeline.live(), "oversize valid prefix must reject");
    for (const auto& packet : mdp::snapshot_packets(3)) transmit(packet);
    require(pipeline.live(), "third epoch recovers");
    const auto deadline = Clock::now() + std::chrono::seconds(5);
    while (pipeline.live()) {
      ingress.consume_one(pipeline);
      require(!receiver_failed.load(std::memory_order_acquire), "receiver failed during idle wait");
      require(Clock::now() < deadline, "watchdog deadline");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    require(decision() == mdp::RiskDecision::stale_book, "silent feed must reject");
    for (const auto& packet : mdp::snapshot_packets(4)) transmit(packet);
    require(decision() == mdp::RiskDecision::accepted, "final snapshot recovers");
    receiver.request_stop();
    receiver.join();
    if (receiver_error) std::rethrow_exception(receiver_error);
    pipeline.book().check_invariants();
    require(ingress.drops() == 0, "unexpected local queue overflow");
    std::cout << "UDP loopback: datagrams=" << sent
              << " snapshots=" << pipeline.counters().snapshots_completed
              << " sequence_gaps=" << pipeline.counters().sequence_gaps
              << " transport_losses=" << pipeline.counters().transport_losses
              << " final_epoch=" << pipeline.epoch()
              << " live=" << pipeline.live() << "\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
