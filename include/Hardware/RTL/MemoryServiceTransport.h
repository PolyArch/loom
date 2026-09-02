#ifndef LOOM_HARDWARE_RTL_MEMORYSERVICETRANSPORT_H
#define LOOM_HARDWARE_RTL_MEMORYSERVICETRANSPORT_H

#include "Fabric/Identity/FabricRefs.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <optional>
#include <utility>

namespace loom::fabric {
class FabricArtifactView;
}

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

struct PortableMemoryServiceLayout final {
  std::uint32_t addressWidthBits = 0;
  std::uint32_t dataWidthBits = 0;
  std::uint32_t maskWidthBits = 0;
  std::uint32_t maximumAddressLaneWidthBits = 0;
};

/// Derives the exact portable carriers from the complete Fabric memory access
/// domain. Hardware emission and mapped-runtime validation consume this same
/// value; neither owns an independent address-width limit.
llvm::Expected<PortableMemoryServiceLayout>
derivePortableMemoryServiceLayout(const fabric::FabricArtifactView &fabric);

/// The byte-address domain of the portable profile. Provider Range and Prefix
/// rows, constant base offsets, the request base address, and runtime memory
/// images all carry this width, so it is the one support ceiling of an
/// address lane: a wider lane has no exact byte address in the profile.
inline constexpr std::uint32_t portableMemoryByteAddressWidthBits =
    portableMemoryBaseAddressFieldWidth;

/// The address arithmetic of the portable profile, derived from the
/// Fabric-derived layout. Every consumer (RTL provider decode and local
/// storage, the mapped-runtime facts, and the simulation harness) evaluates
/// the complete expression `base + lane * elementBytes + byteOffset` in
/// `calculationWidthBits`, requires every bit at or above
/// `byteAddressWidthBits` to be zero, and only then narrows to the
/// byte-address domain. Membership of a wrapped value in a Range or Prefix row
/// is not an overflow proof.
struct PortableMemoryAddressArithmetic final {
  /// The widest address lane the layout carries; never above the byte-address
  /// domain.
  std::uint32_t laneWidthBits = 0;
  std::uint32_t byteAddressWidthBits = portableMemoryByteAddressWidthBits;
  /// A lane of at most `byteAddressWidthBits` times an element byte count
  /// below 2^(byteAddressWidthBits - 3), plus two byte-address terms, stays
  /// below 2^(2 * byteAddressWidthBits - 2); twice the byte-address width is
  /// therefore exact for every admitted request.
  std::uint32_t calculationWidthBits = 2 * portableMemoryByteAddressWidthBits;
};

/// Absent exactly when the widest lane exceeds the byte-address domain: the
/// profile then has no exact address arithmetic and the consumer reports its
/// typed Unsupported outcome.
std::optional<PortableMemoryAddressArithmetic>
derivePortableMemoryAddressArithmetic(
    const PortableMemoryServiceLayout &layout);

/// Canonical, invocation-local index of Fabric-owned memory request sources.
/// Codes are dense and unique across the exact Module artifact. The index is
/// derived once; Mapping, RTL, and runtime consumers perform point lookups and
/// never reconstruct codes from positions or rescan the Fabric per row.
class PortableMemoryRequestContextIndex final {
public:
  static llvm::Expected<PortableMemoryRequestContextIndex>
  get(const fabric::FabricArtifactView &fabric);

  llvm::Expected<std::uint64_t>
  code(fabric::FabricMemoryOccurrenceRef memory,
       std::uint64_t operationRowOrdinal) const;

  llvm::Expected<std::uint64_t>
  first(fabric::FabricMemoryOccurrenceRef memory) const;

private:
  struct Range final {
    std::uint64_t first = 0;
    std::uint64_t count = 0;
  };

  explicit PortableMemoryRequestContextIndex(
      std::map<std::uint64_t, Range> ranges)
      : ranges_(std::move(ranges)) {}

  std::map<std::uint64_t, Range> ranges_;
};

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_MEMORYSERVICETRANSPORT_H
