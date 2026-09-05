#include "queues.hpp"
#include "udp.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

namespace {

std::size_t checks = 0;

void require(bool condition, const char* description) {
    ++checks;
    if (!condition) {
        throw std::runtime_error(description);
    }
}

template <typename Exception, typename Function>
void require_throws(Function&& function, const char* description) {
    bool caught = false;
    try {
        function();
    } catch (const Exception&) {
        caught = true;
    }
    require(caught, description);
}

template <typename Queue>
void capacity_and_wraparound() {
    Queue queue;
    std::uint64_t out = 999;
    require(queue.empty(), "new queue is empty");
    require(!queue.try_pop(out) && out == 999, "empty pop leaves output intact");
    for (std::uint64_t round = 0; round < 200; ++round) {
        for (std::size_t index = 0; index < Queue::capacity; ++index) {
            require(queue.try_push(round * Queue::capacity + index), "exact usable capacity");
        }
        require(!queue.empty(), "full queue is not empty");
        require(!queue.try_push(999999), "full queue refuses extra item");
        for (std::size_t index = 0; index < Queue::capacity; ++index) {
            require(queue.try_pop(out), "all accepted items remain available");
            require(out == round * Queue::capacity + index, "FIFO survives wraparound");
        }
        require(queue.empty(), "queue empties after exact drain");
        require(!queue.try_pop(out), "failed full push did not insert item");
    }
}

struct Payload {
    std::uint64_t sequence = 0;
    std::uint64_t complement = 0;
    std::array<std::uint64_t, 4> guards{};
    bool operator==(const Payload&) const = default;
};

Payload make_payload(std::uint64_t sequence) {
    return {sequence, ~sequence, {sequence * 3, sequence ^ 0xa55a, sequence + 9, sequence * 17}};
}

template <typename Queue>
void concurrent_fifo() {
    Queue queue;
    constexpr std::uint64_t count = 200000;
    std::atomic<bool> abort{false};
    std::uint64_t pushed = 0;
    std::uint64_t popped = 0;
    // Deadlock watchdog only, not a throughput assertion. CTest also imposes a
    // process timeout. No correctness claim depends on a thread winning a race.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    const auto should_abort = [&] {
        if (abort.load(std::memory_order_relaxed)) {
            return true;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            abort.store(true, std::memory_order_relaxed);
            return true;
        }
        return false;
    };
    std::jthread producer([&] {
        try {
            for (std::uint64_t sequence = 1; sequence <= count; ++sequence) {
                if (abort.load(std::memory_order_relaxed)) {
                    return;
                }
                const auto payload = make_payload(sequence);
                while (!queue.try_push(payload)) {
                    if (should_abort()) {
                        return;
                    }
                    std::this_thread::yield();
                }
                ++pushed;
            }
        } catch (...) {
            abort.store(true, std::memory_order_relaxed);
        }
    });
    std::jthread consumer([&] {
        try {
            for (std::uint64_t sequence = 1; sequence <= count; ++sequence) {
                if (abort.load(std::memory_order_relaxed)) {
                    return;
                }
                Payload payload;
                while (!queue.try_pop(payload)) {
                    if (should_abort()) {
                        return;
                    }
                    std::this_thread::yield();
                }
                if (!(payload == make_payload(sequence))) {
                    abort.store(true, std::memory_order_relaxed);
                    return;
                }
                ++popped;
            }
        } catch (...) {
            abort.store(true, std::memory_order_relaxed);
        }
    });
    producer.join();
    consumer.join();
    require(!abort.load(), "concurrent transfer completed without torn/lost/reordered payloads");
    require(pushed == count && popped == count, "all unique concurrent payloads transferred");
    // Both workers are joined, so this observation cannot race with production.
    require(queue.empty(), "concurrent queue fully drained");
}

void udp_datagrams() {
    mdp::UdpSocket sender;
    mdp::UdpSocket receiver;
    require(sender.port() != 0 && receiver.port() != 0, "ephemeral loopback ports assigned");
    require(sender.port() != receiver.port(), "live sockets have separate ports");

    std::array<std::byte, 48> expected{};
    for (std::size_t index = 0; index < expected.size(); ++index) {
        expected[index] = static_cast<std::byte>(index);
    }
    std::array<std::byte, 48> actual{};
    auto result = receiver.receive(actual, 0);
    require(result.status == mdp::ReceiveStatus::timeout && result.bytes == 0,
            "empty nonblocking receive times out");
    sender.send_to_loopback(expected, receiver.port());
    result = receiver.receive(actual, 1000);
    require(result.status == mdp::ReceiveStatus::received && result.bytes == expected.size(),
            "complete datagram received");
    require(actual == expected, "datagram bytes preserved");
    result = receiver.receive(actual, std::chrono::milliseconds(5));
    require(result.status == mdp::ReceiveStatus::timeout, "bounded receive times out after drain");

    std::array<std::byte, 49> oversize{};
    for (std::size_t index = 0; index < expected.size(); ++index) {
        oversize[index] = expected[index];
    }
    sender.send_to_loopback(oversize, receiver.port());
    result = receiver.receive(actual, 1000);
    require(result.status == mdp::ReceiveStatus::truncated && result.bytes == actual.size(),
            "49-byte datagram cannot impersonate valid 48-byte packet");
    require(actual == expected, "truncated prefix deliberately looks valid");
    std::array<std::byte, 4096> large{};
    sender.send_to_loopback(large, receiver.port());
    result = receiver.receive(actual, 1000);
    require(result.status == mdp::ReceiveStatus::truncated, "large datagram truncation explicit");
    sender.send_to_loopback(expected, receiver.port());
    result = receiver.receive(actual, 1000);
    require(result.status == mdp::ReceiveStatus::received && actual == expected,
            "truncated remainder discarded before next datagram");

    sender.send_to_loopback({}, receiver.port());
    result = receiver.receive(actual, 1000);
    require(result.status == mdp::ReceiveStatus::received && result.bytes == 0,
            "zero-byte UDP is a received datagram, not timeout or EOF");
    require_throws<std::invalid_argument>([&] { (void)receiver.receive({}, 0); },
                                          "empty receive storage rejected");
    require_throws<std::invalid_argument>([&] { (void)receiver.receive(actual, -1); },
                                          "negative timeout rejected");
    require_throws<std::invalid_argument>([&] { sender.send_to_loopback(expected, 0); },
                                          "invalid destination rejected");
    std::array<std::byte, 65508> too_large{};
    require_throws<std::invalid_argument>(
        [&] { sender.send_to_loopback(too_large, receiver.port()); },
        "oversize IPv4 UDP payload rejected before send");
    require_throws<std::system_error>([&] { mdp::UdpSocket conflict(receiver.port()); },
                                      "bind errors explicit and constructor cleans socket");

    const auto assigned_port = receiver.port();
    mdp::UdpSocket moved(std::move(receiver));
    require(receiver.port() == 0 && moved.port() == assigned_port, "move transfers ownership");
    require_throws<std::logic_error>([&] { (void)receiver.receive(actual, 0); },
                                     "moved-from receiver closed");
    mdp::UdpSocket reassigned;
    reassigned = std::move(moved);
    require(moved.port() == 0 && reassigned.port() == assigned_port, "move assignment transfers ownership");
    sender.send_to_loopback(expected, reassigned.port());
    result = reassigned.receive(actual, 1000);
    require(result.status == mdp::ReceiveStatus::received && actual == expected,
            "moved socket remains functional");
    reassigned.close();
    reassigned.close();
    require(reassigned.port() == 0, "close idempotent");
    require_throws<std::logic_error>([&] { (void)reassigned.receive(actual, 0); },
                                     "closed receive fails immediately");
    require_throws<std::logic_error>([&] { reassigned.send_to_loopback(expected, sender.port()); },
                                     "closed send fails immediately");

    std::uint16_t released_port = 0;
    try {
        mdp::UdpSocket scoped;
        released_port = scoped.port();
        throw std::runtime_error("exercise stack unwinding");
    } catch (const std::runtime_error&) {
    }
    mdp::UdpSocket rebound(released_port);
    require(rebound.port() == released_port && released_port != 0,
            "stack unwinding releases bound socket");
}

}  // namespace

int main() {
    static_assert(!std::is_copy_constructible_v<mdp::SpscQueue<Payload, 7>>);
    static_assert(!std::is_move_constructible_v<mdp::SpscQueue<Payload, 7>>);
    static_assert(!std::is_copy_constructible_v<mdp::MutexQueue<Payload, 7>>);
    static_assert(!std::is_copy_constructible_v<mdp::UdpSocket>);
    static_assert(std::is_nothrow_move_constructible_v<mdp::UdpSocket>);
    try {
        capacity_and_wraparound<mdp::SpscQueue<std::uint64_t, 1>>();
        capacity_and_wraparound<mdp::SpscQueue<std::uint64_t, 7>>();
        capacity_and_wraparound<mdp::SpscQueue<std::uint64_t, 8>>();
        capacity_and_wraparound<mdp::MutexQueue<std::uint64_t, 1>>();
        capacity_and_wraparound<mdp::MutexQueue<std::uint64_t, 7>>();
        capacity_and_wraparound<mdp::MutexQueue<std::uint64_t, 8>>();
        concurrent_fifo<mdp::SpscQueue<Payload, 7>>();
        concurrent_fifo<mdp::SpscQueue<Payload, 256>>();
        concurrent_fifo<mdp::MutexQueue<Payload, 7>>();
        udp_datagrams();
        std::cout << "transport checks=" << checks
                  << " concurrent_payloads=600000: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "transport test failure: " << error.what() << '\n';
        return 1;
    }
}
