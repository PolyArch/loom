#include "Common/ExecutionControl.h"

#include "llvm/Support/Process.h"

#include <chrono>
#include <cstdint>
#include <limits>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace loom {
namespace {

std::uint64_t saturatedNanoseconds(std::chrono::steady_clock::duration value) {
  const auto nanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(value).count();
  if (nanoseconds <= 0)
    return 0;
  using Count = decltype(nanoseconds);
  if constexpr (sizeof(Count) > sizeof(std::uint64_t))
    if (nanoseconds >
        static_cast<Count>(std::numeric_limits<std::uint64_t>::max()))
      return std::numeric_limits<std::uint64_t>::max();
  return static_cast<std::uint64_t>(nanoseconds);
}

std::optional<std::uint64_t> peakResidentMemoryBytes() {
#if defined(__unix__) || defined(__APPLE__)
  struct rusage usage{};
  if (::getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0)
    return std::nullopt;
  const std::uint64_t resident = static_cast<std::uint64_t>(usage.ru_maxrss);
#if defined(__APPLE__)
  return resident;
#else
  if (resident > std::numeric_limits<std::uint64_t>::max() / 1024)
    return std::numeric_limits<std::uint64_t>::max();
  return resident * 1024;
#endif
#else
  return std::nullopt;
#endif
}

} // namespace

ExecutionResourceTracker::ExecutionResourceTracker()
    : begin_(std::chrono::steady_clock::now()) {}

ExecutionResourceStatistics ExecutionResourceTracker::observe() const {
  const std::size_t allocated = llvm::sys::Process::GetMallocUsage();
  return {saturatedNanoseconds(std::chrono::steady_clock::now() - begin_),
          allocated > std::numeric_limits<std::uint64_t>::max()
              ? std::numeric_limits<std::uint64_t>::max()
              : static_cast<std::uint64_t>(allocated),
          peakResidentMemoryBytes()};
}

} // namespace loom
