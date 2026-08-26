#ifndef LOOM_RUNTIME_GEM5DISPATCHABI_H
#define LOOM_RUNTIME_GEM5DISPATCHABI_H

#include <cstdint>

namespace loom::runtime {

enum class Gem5RootLifecycleAction : std::uint32_t {
  Start = 0,
  Completion = 1,
};

inline constexpr std::uint64_t gem5ThreadDispatchTargetLow = 0x00;
inline constexpr std::uint64_t gem5ThreadDispatchTargetHigh = 0x04;
inline constexpr std::uint64_t gem5ThreadDispatchControl = 0x08;
inline constexpr std::uint64_t gem5ThreadDispatchStatus = 0x0c;
inline constexpr std::uint64_t gem5ThreadDispatchOccurrenceLow = 0x10;
inline constexpr std::uint64_t gem5ThreadDispatchError = 0x14;
inline constexpr std::uint64_t gem5ThreadDispatchInvocationLow = 0x18;
inline constexpr std::uint64_t gem5ThreadDispatchInvocationHigh = 0x1c;
inline constexpr std::uint64_t gem5ThreadDispatchInvocationSize = 0x20;
inline constexpr std::uint64_t gem5ThreadDispatchOccurrenceHigh = 0x24;
inline constexpr std::uint64_t gem5ThreadDispatchRootEntityLow = 0x28;
inline constexpr std::uint64_t gem5ThreadDispatchRootEntityHigh = 0x2c;
inline constexpr std::uint64_t gem5ThreadDispatchRootEvent = 0x30;
inline constexpr std::uint64_t gem5ThreadDispatchRootOccurrenceLow = 0x34;
inline constexpr std::uint64_t gem5ThreadDispatchRootOccurrenceHigh = 0x38;
inline constexpr std::uint64_t gem5ThreadDispatchWorkerSlotBase = 0x1000;
inline constexpr std::uint64_t gem5ThreadDispatchWorkerSlotStride = 0x08;
inline constexpr std::uint64_t gem5ThreadDispatchWorkerCompletion = 0x00;
inline constexpr std::uint64_t gem5ThreadDispatchWorkerFailure = 0x04;
inline constexpr std::uint64_t gem5ThreadDispatchApertureBytes = 0x10000;
inline constexpr std::uint64_t gem5MaximumDynamicSpatialInvocations = 4096;

inline constexpr std::uint32_t gem5ThreadDispatchStart = 1U << 0;
inline constexpr std::uint32_t gem5ThreadDispatchReset = 1U << 1;
inline constexpr std::uint32_t gem5ThreadDispatchBusy = 1U << 0;
inline constexpr std::uint32_t gem5ThreadDispatchDone = 1U << 1;
inline constexpr std::uint32_t gem5ThreadDispatchFailed = 1U << 2;

inline constexpr std::uint32_t gem5RootLifecycleTraceMagic = 0x4c524531U;

} // namespace loom::runtime

#endif // LOOM_RUNTIME_GEM5DISPATCHABI_H
