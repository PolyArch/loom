#ifndef LOOM_COMMON_EXECUTIONCONTROL_H
#define LOOM_COMMON_EXECUTIONCONTROL_H

#include <chrono>
#include <cstdint>
#include <optional>

namespace loom {

/// Non-owning, invocation-local interruption query. It is execution policy,
/// never candidate identity or semantic work accounting. Providers observe it
/// only at their documented atomic owner boundaries.
class ExecutionControlView final {
public:
  using StopQuery = bool (*)(const void *context);
  using RemainingTimeQuery =
      std::optional<std::chrono::steady_clock::duration> (*)(
          const void *context);

  constexpr ExecutionControlView() = default;
  constexpr ExecutionControlView(
      const void *context, StopQuery query,
      RemainingTimeQuery remainingTimeQuery = nullptr)
      : context_(context), query_(query),
        remainingTimeQuery_(remainingTimeQuery) {}

  bool stopRequested() const { return query_ && query_(context_); }
  std::optional<std::chrono::steady_clock::duration> remainingTime() const {
    return remainingTimeQuery_ ? remainingTimeQuery_(context_) : std::nullopt;
  }

private:
  const void *context_ = nullptr;
  StopQuery query_ = nullptr;
  RemainingTimeQuery remainingTimeQuery_ = nullptr;
};

/// Invocation-local physical resource budget projected from the execution
/// site's admitted claim. Missing or zero dimensions are unconstrained by that
/// site. These values affect scheduling only and never enter candidate
/// identity.
struct ExecutionResourceBudget final {
  std::optional<std::uint64_t> cpuCores;
  std::optional<std::uint64_t> memoryBytes;
};

/// Host-process observations attached to provider diagnostics and interrupted
/// results. Concurrent provider activity can contribute to the CPU, allocator,
/// and resident-memory values. They explain execution and cannot affect
/// Mapping identity or ordering.
struct ExecutionResourceStatistics final {
  std::uint64_t activeWallTimeNanoseconds = 0;
  std::uint64_t allocatedMemoryBytes = 0;
  std::optional<std::uint64_t> peakResidentMemoryBytes;
  std::optional<std::uint64_t> processCpuTimeDeltaNanoseconds;
};

class ExecutionResourceTracker final {
public:
  ExecutionResourceTracker();

  ExecutionResourceStatistics observe() const;

private:
  std::chrono::steady_clock::time_point begin_;
  std::optional<std::uint64_t> beginCpuTimeNanoseconds_;
};

} // namespace loom

#endif // LOOM_COMMON_EXECUTIONCONTROL_H
