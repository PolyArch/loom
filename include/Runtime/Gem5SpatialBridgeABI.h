#ifndef LOOM_RUNTIME_GEM5SPATIALBRIDGEABI_H
#define LOOM_RUNTIME_GEM5SPATIALBRIDGEABI_H

#include <cstdint>

namespace loom::runtime {

inline constexpr std::uint64_t gem5SpatialBridgeStatus = 0x00;
inline constexpr std::uint64_t gem5SpatialBridgeControl = 0x04;
inline constexpr std::uint64_t gem5SpatialBridgeError = 0x08;
inline constexpr std::uint64_t gem5SpatialBridgeSequenceLow = 0x0c;
inline constexpr std::uint64_t gem5SpatialBridgeSequenceHigh = 0x10;
inline constexpr std::uint64_t gem5SpatialBridgeStaticLaunchLow = 0x14;
inline constexpr std::uint64_t gem5SpatialBridgeStaticLaunchHigh = 0x18;
inline constexpr std::uint64_t gem5SpatialBridgeStaticLaunchSize = 0x1c;
inline constexpr std::uint64_t gem5SpatialBridgeCompletionTickLow = 0x20;
inline constexpr std::uint64_t gem5SpatialBridgeCompletionTickHigh = 0x24;
inline constexpr std::uint64_t gem5SpatialBridgeInvocationLow = 0x28;
inline constexpr std::uint64_t gem5SpatialBridgeInvocationHigh = 0x2c;
inline constexpr std::uint64_t gem5SpatialBridgeInvocationSize = 0x30;

inline constexpr std::uint32_t gem5SpatialBridgeStart = 1U << 0;
inline constexpr std::uint32_t gem5SpatialBridgeReset = 1U << 1;
inline constexpr std::uint32_t gem5SpatialBridgeBusy = 1U << 0;
inline constexpr std::uint32_t gem5SpatialBridgeDone = 1U << 1;
inline constexpr std::uint32_t gem5SpatialBridgeFailed = 1U << 2;

} // namespace loom::runtime

#endif // LOOM_RUNTIME_GEM5SPATIALBRIDGEABI_H
