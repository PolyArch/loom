#ifndef LOOM_HARDWARE_RTL_MEMORYSERVICETRANSPORT_H
#define LOOM_HARDWARE_RTL_MEMORYSERVICETRANSPORT_H

#include <cstdint>

namespace loom::hardware::rtl {

inline constexpr std::uint32_t portableMemoryRequestKindWidth = 1;
inline constexpr std::uint32_t portableMemoryActiveLanesKindWidth = 1;
inline constexpr std::uint32_t portableMemoryAccessFormWidth = 2;
inline constexpr std::uint32_t portableMemoryAddressFormWidth = 1;
inline constexpr std::uint32_t portableMemoryElementWidthFieldWidth = 64;
inline constexpr std::uint32_t portableMemoryLaneCountFieldWidth = 64;
inline constexpr std::uint32_t portableMemoryAddressLaneWidthFieldWidth = 32;
inline constexpr std::uint32_t portableMemoryBaseAddressFieldWidth = 64;
inline constexpr std::uint32_t portableMemoryContextFieldWidth = 64;
inline constexpr std::uint32_t portableMemoryHandshakeWidth = 1;

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_MEMORYSERVICETRANSPORT_H
