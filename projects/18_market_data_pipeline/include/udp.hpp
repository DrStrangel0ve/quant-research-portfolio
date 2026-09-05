#pragma once

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <system_error>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace mdp {

enum class ReceiveStatus { received, timeout, truncated };

struct ReceiveResult {
    ReceiveStatus status;
    // Bytes copied to caller storage, not original size of a truncated datagram.
    std::size_t bytes;
};

// A local IPv4 UDP adapter, deliberately restricted to loopback. Each socket has
// one receiving owner. Do not close/move/destroy a socket while another thread is
// using it. Close is idempotent; errors are exceptions, not end-of-stream.
class UdpSocket {
#ifdef _WIN32
    using Handle = SOCKET;
    static constexpr Handle invalid = INVALID_SOCKET;

    struct WinsockRuntime {
        WinsockRuntime() {
            WSADATA data{};
            const int result = WSAStartup(MAKEWORD(2, 2), &data);
            if (result != 0) {
                throw std::system_error(result, std::system_category(), "WSAStartup");
            }
        }
        ~WinsockRuntime() { WSACleanup(); }
    };

    static void initialize() {
        static WinsockRuntime runtime;
        (void)runtime;
    }

    [[noreturn]] static void socket_error(const char* operation) {
        throw std::system_error(WSAGetLastError(), std::system_category(), operation);
    }
#else
    using Handle = int;
    static constexpr Handle invalid = -1;
    static void initialize() {}
    [[noreturn]] static void socket_error(const char* operation) {
        throw std::system_error(errno, std::generic_category(), operation);
    }
#endif

public:
    explicit UdpSocket(std::uint16_t requested_port = 0) {
        initialize();
        socket_ = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (socket_ == invalid) {
            socket_error("socket");
        }
        try {
            sockaddr_in address{};
            address.sin_family = AF_INET;
            address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            address.sin_port = htons(requested_port);
            if (::bind(socket_, reinterpret_cast<const sockaddr*>(&address),
                       static_cast<int>(sizeof(address))) != 0) {
                socket_error("bind loopback");
            }
#ifdef _WIN32
            int address_size = sizeof(address);
#else
            socklen_t address_size = sizeof(address);
#endif
            if (::getsockname(socket_, reinterpret_cast<sockaddr*>(&address),
                              &address_size) != 0) {
                socket_error("getsockname");
            }
            port_ = ntohs(address.sin_port);
            // Readiness can become stale; nonblocking recv guarantees the
            // explicit receive deadline still bounds the wait in that case.
#ifdef _WIN32
            u_long nonblocking = 1;
            if (::ioctlsocket(socket_, FIONBIO, &nonblocking) != 0) {
                socket_error("ioctlsocket");
            }
#else
            const int flags = ::fcntl(socket_, F_GETFL, 0);
            if (flags < 0 || ::fcntl(socket_, F_SETFL, flags | O_NONBLOCK) != 0) {
                socket_error("fcntl nonblocking");
            }
#endif
        } catch (...) {
            close();
            throw;
        }
    }

    ~UdpSocket() { close(); }
    UdpSocket(const UdpSocket&) = delete;
    UdpSocket& operator=(const UdpSocket&) = delete;
    UdpSocket(UdpSocket&& other) noexcept
        : socket_(std::exchange(other.socket_, invalid)),
          port_(std::exchange(other.port_, std::uint16_t{0})) {}
    UdpSocket& operator=(UdpSocket&& other) noexcept {
        if (this != &other) {
            close();
            socket_ = std::exchange(other.socket_, invalid);
            port_ = std::exchange(other.port_, std::uint16_t{0});
        }
        return *this;
    }

    [[nodiscard]] std::uint16_t port() const noexcept { return port_; }

    void close() noexcept {
        if (socket_ != invalid) {
#ifdef _WIN32
            ::closesocket(socket_);
#else
            ::close(socket_);
#endif
            socket_ = invalid;
            port_ = 0;
        }
    }

    void send_to_loopback(std::span<const std::byte> data,
                          std::uint16_t destination_port) const {
        require_open();
        if (destination_port == 0 || data.size() > 65507) {
            throw std::invalid_argument("Invalid UDP destination or IPv4 payload size");
        }
        sockaddr_in destination{};
        destination.sin_family = AF_INET;
        destination.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        destination.sin_port = htons(destination_port);
        const char* buffer = data.empty() ? "" : reinterpret_cast<const char*>(data.data());
        const auto sent = ::sendto(socket_, buffer, static_cast<int>(data.size()), 0,
                                  reinterpret_cast<const sockaddr*>(&destination),
                                  static_cast<int>(sizeof(destination)));
        if (sent < 0) {
            socket_error("sendto loopback");
        }
        if (static_cast<std::size_t>(sent) != data.size()) {
            throw std::runtime_error("UDP send did not transmit the complete datagram");
        }
    }

    [[nodiscard]] ReceiveResult receive(std::span<std::byte> data, int timeout_ms) const {
        return receive(data, std::chrono::milliseconds(timeout_ms));
    }

    [[nodiscard]] ReceiveResult receive(std::span<std::byte> data,
                                        std::chrono::milliseconds timeout) const {
        require_open();
        if (data.empty() || data.size() > static_cast<std::size_t>(INT_MAX) ||
            timeout.count() < 0) {
            throw std::invalid_argument("Receive needs a nonempty bounded buffer and timeout >= 0");
        }
        using Clock = std::chrono::steady_clock;
        // Clamp before deadline arithmetic and OS int conversions. Very large
        // requested timeouts are capped at INT_MAX milliseconds (~24.8 days).
        const auto bounded_timeout = std::min(timeout, std::chrono::milliseconds(INT_MAX));
        const auto deadline = Clock::now() + bounded_timeout;
        bool first_poll = true;
        for (;;) {
            const auto now = Clock::now();
            if (!first_poll && now >= deadline) {
                return {ReceiveStatus::timeout, 0};
            }
            first_poll = false;
            const auto remaining = now >= deadline ? std::chrono::milliseconds(0)
                : std::chrono::ceil<std::chrono::milliseconds>(deadline - now);
            const int wait_ms = static_cast<int>(remaining.count());
#ifdef _WIN32
            fd_set readers;
            FD_ZERO(&readers);
            FD_SET(socket_, &readers);
            timeval wait_time{};
            wait_time.tv_sec = wait_ms / 1000;
            wait_time.tv_usec = (wait_ms % 1000) * 1000;
            const int ready = ::select(0, &readers, nullptr, nullptr, &wait_time);
            if (ready == SOCKET_ERROR) {
                if (WSAGetLastError() == WSAEINTR) {
                    continue;
                }
                socket_error("select");
            }
#else
            pollfd reader{socket_, POLLIN, 0};
            const int ready = ::poll(&reader, 1, wait_ms);
            if (ready < 0) {
                if (errno == EINTR) {
                    continue;
                }
                socket_error("poll");
            }
            if (ready > 0 && (reader.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
                throw std::runtime_error("UDP poll reported a socket error");
            }
#endif
            if (ready == 0) {
                return {ReceiveStatus::timeout, 0};
            }
#ifdef _WIN32
            const int received = ::recvfrom(socket_, reinterpret_cast<char*>(data.data()),
                                             static_cast<int>(data.size()), 0, nullptr, nullptr);
            if (received == SOCKET_ERROR) {
                const int error = WSAGetLastError();
                if (error == WSAEMSGSIZE) {
                    // WinSock fills the supplied prefix and discards the rest.
                    return {ReceiveStatus::truncated, data.size()};
                }
                if (error == WSAEWOULDBLOCK || error == WSAEINTR) {
                    continue;
                }
                throw std::system_error(error, std::system_category(), "recvfrom");
            }
            return {ReceiveStatus::received, static_cast<std::size_t>(received)};
#else
            iovec buffer{data.data(), data.size()};
            msghdr message{};
            message.msg_iov = &buffer;
            message.msg_iovlen = 1;
            const auto received = ::recvmsg(socket_, &message, 0);
            if (received < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
                    continue;
                }
                socket_error("recvmsg");
            }
            const auto bytes = std::min(data.size(), static_cast<std::size_t>(received));
            if ((message.msg_flags & MSG_TRUNC) != 0) {
                return {ReceiveStatus::truncated, bytes};
            }
            return {ReceiveStatus::received, bytes};
#endif
        }
    }

private:
    void require_open() const {
        if (socket_ == invalid) {
            throw std::logic_error("UDP socket is closed");
        }
    }

    Handle socket_ = invalid;
    std::uint16_t port_ = 0;
};

}  // namespace mdp
