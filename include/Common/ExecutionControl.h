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

  constexpr ExecutionControlView() = default;
  constexpr ExecutionControlView(const void *context, StopQuery query)
      : context_(context), query_(query) {}

  bool stopRequested() const { return query_ && query_(context_); }

private:
  const void *context_ = nullptr;
  StopQuery query_ = nullptr;
};

/// Process observations attached to an interrupted provider result. These
/// values explain execution and cannot affect Mapping identity or ordering.
struct ExecutionResourceStatistics final {
  std::uint64_t activeWallTimeNanoseconds = 0;
  std::uint64_t allocatedMemoryBytes = 0;
  std::optional<std::uint64_t> peakResidentMemoryBytes;
};

class ExecutionResourceTracker final {
public:
  ExecutionResourceTracker();

  ExecutionResourceStatistics observe() const;

private:
  std::chrono::steady_clock::time_point begin_;
};

} // namespace loom

#endif // LOOM_COMMON_EXECUTIONCONTROL_H
