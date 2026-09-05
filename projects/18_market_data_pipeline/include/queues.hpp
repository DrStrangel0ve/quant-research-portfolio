#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <limits>
#include <mutex>
#include <type_traits>

namespace mdp {

// Exactly Capacity usable slots; the extra slot distinguishes full from empty.
// One producer calls try_push and one consumer calls try_pop/empty. Destruction
// requires both threads to have stopped. This is not an MPSC/MPMC queue.
template <typename T, std::size_t Capacity>
class SpscQueue {
    static_assert(Capacity > 0 && Capacity < std::numeric_limits<std::size_t>::max());
    static_assert(std::is_default_constructible_v<T>);
    static_assert(std::is_nothrow_copy_assignable_v<T>,
                  "Queue payload assignment must not throw");

    // 64 is a layout choice for common desktop CPUs, not a universal cache-line
    // guarantee. Each index occupies its own aligned object, separate from data.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)  // Deliberate padding keeps the atomic indices apart.
#endif
    struct alignas(64) Index {
        std::atomic<std::size_t> value{0};
    };
#ifdef _MSC_VER
#pragma warning(pop)
#endif

public:
    static constexpr std::size_t capacity = Capacity;
    SpscQueue() = default;
    SpscQueue(const SpscQueue&) = delete;
    SpscQueue& operator=(const SpscQueue&) = delete;
    SpscQueue(SpscQueue&&) = delete;
    SpscQueue& operator=(SpscQueue&&) = delete;

    [[nodiscard]] bool try_push(const T& item) noexcept {
        const auto head = head_.value.load(std::memory_order_relaxed);
        const auto next_head = next(head);
        if (next_head == tail_.value.load(std::memory_order_acquire)) {
            return false;
        }
        slots_[head] = item;
        head_.value.store(next_head, std::memory_order_release);
        return true;
    }

    [[nodiscard]] bool try_pop(T& item) noexcept {
        const auto tail = tail_.value.load(std::memory_order_relaxed);
        if (tail == head_.value.load(std::memory_order_acquire)) {
            return false;
        }
        item = slots_[tail];
        tail_.value.store(next(tail), std::memory_order_release);
        return true;
    }

    // Consumer-side observation only: producer may publish immediately after it.
    [[nodiscard]] bool empty() const noexcept {
        return tail_.value.load(std::memory_order_relaxed) ==
               head_.value.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool indices_are_lock_free() const noexcept {
        return head_.value.is_lock_free() && tail_.value.is_lock_free();
    }

private:
    static constexpr std::size_t next(std::size_t index) noexcept {
        return index == Capacity ? 0 : index + 1;
    }

    Index head_{};
    Index tail_{};
    std::array<T, Capacity + 1> slots_{};
};

template <typename T, std::size_t Capacity>
using SPSC = SpscQueue<T, Capacity>;

// Same bounded try interface and usable capacity as SpscQueue. This reference
// serializes access with a mutex, has no condition variables and allocates no
// storage during push/pop. Both queues require nonthrowing payload assignment.
template <typename T, std::size_t Capacity>
class MutexQueue {
    static_assert(Capacity > 0);
    static_assert(std::is_default_constructible_v<T>);
    static_assert(std::is_nothrow_copy_assignable_v<T>);

public:
    static constexpr std::size_t capacity = Capacity;
    MutexQueue() = default;
    MutexQueue(const MutexQueue&) = delete;
    MutexQueue& operator=(const MutexQueue&) = delete;
    MutexQueue(MutexQueue&&) = delete;
    MutexQueue& operator=(MutexQueue&&) = delete;

    [[nodiscard]] bool try_push(const T& item) {
        const std::lock_guard guard(mutex_);
        if (size_ == Capacity) {
            return false;
        }
        slots_[head_] = item;
        head_ = next(head_);
        ++size_;
        return true;
    }

    [[nodiscard]] bool try_pop(T& item) {
        const std::lock_guard guard(mutex_);
        if (size_ == 0) {
            return false;
        }
        item = slots_[tail_];
        tail_ = next(tail_);
        --size_;
        return true;
    }

    [[nodiscard]] bool empty() const {
        const std::lock_guard guard(mutex_);
        return size_ == 0;
    }

private:
    static constexpr std::size_t next(std::size_t index) noexcept {
        return index == Capacity - 1 ? 0 : index + 1;
    }

    mutable std::mutex mutex_;
    std::array<T, Capacity> slots_{};
    std::size_t head_ = 0;
    std::size_t tail_ = 0;
    std::size_t size_ = 0;
};

}  // namespace mdp
