#include "Common/ExecutionControl.h"

#include "llvm/Support/Process.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

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

std::optional<std::uint64_t> linuxResidentStatusBytes(const char *field) {
#if defined(__linux__)
  std::FILE *status = std::fopen("/proc/self/status", "r");
  if (!status)
    return std::nullopt;
  char line[256];
  std::optional<std::uint64_t> result;
  const std::size_t fieldLength = std::strlen(field);
  while (std::fgets(line, sizeof(line), status)) {
    if (std::strncmp(line, field, fieldLength) != 0)
      continue;
    unsigned long long kibibytes = 0;
    if (std::sscanf(line + fieldLength, "%llu kB", &kibibytes) == 1) {
      const std::uint64_t value = static_cast<std::uint64_t>(kibibytes);
      result = value > std::numeric_limits<std::uint64_t>::max() / 1024
                   ? std::numeric_limits<std::uint64_t>::max()
                   : value * 1024;
    }
    break;
  }
  if (std::fclose(status) != 0)
    return std::nullopt;
  return result;
#else
  (void)field;
  return std::nullopt;
#endif
}

std::optional<std::uint64_t> peakResidentMemoryBytes() {
#if defined(__linux__)
  return linuxResidentStatusBytes("VmHWM:");
#elif defined(__unix__) || defined(__APPLE__)
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

std::optional<std::uint64_t> currentResidentMemoryBytes() {
#if defined(__linux__)
  return linuxResidentStatusBytes("VmRSS:");
#else
  return std::nullopt;
#endif
}

std::optional<std::uint64_t> processCpuTimeNanoseconds() {
#if defined(__unix__) || defined(__APPLE__)
  struct rusage usage{};
  if (::getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_utime.tv_sec < 0 ||
      usage.ru_utime.tv_usec < 0 || usage.ru_stime.tv_sec < 0 ||
      usage.ru_stime.tv_usec < 0)
    return std::nullopt;
  const auto timeNanoseconds = [](const timeval &value) {
    constexpr std::uint64_t nanosecondsPerSecond = UINT64_C(1000000000);
    constexpr std::uint64_t nanosecondsPerMicrosecond = UINT64_C(1000);
    const std::uint64_t seconds = static_cast<std::uint64_t>(value.tv_sec);
    const std::uint64_t microseconds =
        static_cast<std::uint64_t>(value.tv_usec);
    if (seconds >
        std::numeric_limits<std::uint64_t>::max() / nanosecondsPerSecond)
      return std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t secondNanoseconds = seconds * nanosecondsPerSecond;
    const std::uint64_t microsecondNanoseconds =
        microseconds * nanosecondsPerMicrosecond;
    return microsecondNanoseconds >
                   std::numeric_limits<std::uint64_t>::max() - secondNanoseconds
               ? std::numeric_limits<std::uint64_t>::max()
               : secondNanoseconds + microsecondNanoseconds;
  };
  const std::uint64_t user = timeNanoseconds(usage.ru_utime);
  const std::uint64_t system = timeNanoseconds(usage.ru_stime);
  return system > std::numeric_limits<std::uint64_t>::max() - user
             ? std::numeric_limits<std::uint64_t>::max()
             : user + system;
#else
  return std::nullopt;
#endif
}

} // namespace

bool releaseUnusedProcessMemory() {
#if defined(__GLIBC__)
  return ::malloc_trim(0) != 0;
#else
  return false;
#endif
}

ExecutionResourceTracker::ExecutionResourceTracker()
    : begin_(std::chrono::steady_clock::now()),
      beginCpuTimeNanoseconds_(processCpuTimeNanoseconds()) {}

ExecutionResourceStatistics ExecutionResourceTracker::observe() const {
  const std::size_t allocated = llvm::sys::Process::GetMallocUsage();
  const std::optional<std::uint64_t> currentCpu = processCpuTimeNanoseconds();
  const std::optional<std::uint64_t> processCpuDelta =
      beginCpuTimeNanoseconds_ && currentCpu &&
              *currentCpu >= *beginCpuTimeNanoseconds_
          ? std::optional<std::uint64_t>(*currentCpu -
                                         *beginCpuTimeNanoseconds_)
          : std::nullopt;
  return {saturatedNanoseconds(std::chrono::steady_clock::now() - begin_),
          allocated > std::numeric_limits<std::uint64_t>::max()
              ? std::numeric_limits<std::uint64_t>::max()
              : static_cast<std::uint64_t>(allocated),
          currentResidentMemoryBytes(), peakResidentMemoryBytes(),
          processCpuDelta};
}

} // namespace loom
