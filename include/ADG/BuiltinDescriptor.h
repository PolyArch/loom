#ifndef LOOM_ADG_BUILTINDESCRIPTOR_H
#define LOOM_ADG_BUILTINDESCRIPTOR_H

#include "ADG/MemoryLibrary.h"
#include "ADG/SpecialMathCapabilityProfile.h"

#include "Fabric/IR/FabricEnums.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <system_error>

namespace loom::adg {

enum class BuiltinTargetPreset : std::uint8_t { Small, Coverage, Large };

inline constexpr std::uint64_t builtinSystemClockPeriodFs = 1'000'000;
inline constexpr std::uint64_t builtinSystemMemoryCompletionCycles = 20;

struct BuiltinFuOccurrenceCounts final {
  std::uint32_t dedicatedScalarAdd;
  std::uint32_t mac;
  std::uint32_t vectorCompute;
  std::uint32_t loopControl;
  std::uint32_t tokenControl;
  std::uint32_t vectorAdapter;
  std::uint32_t vectorStructural;
  std::uint32_t specialMath;
};

constexpr std::uint32_t builtinCeilDiv(std::uint32_t value,
                                       std::uint32_t divisor) {
  return (value + divisor - 1) / divisor;
}

constexpr BuiltinFuOccurrenceCounts
builtinBalancedFuOccurrences(std::uint32_t peCount) {
  return {
      peCount == 0 ? 0 : std::max(1U, builtinCeilDiv(peCount, 8)),
      builtinCeilDiv(peCount, 2),
      builtinCeilDiv(peCount, 4),
      builtinCeilDiv(peCount, 4),
      builtinCeilDiv(peCount, 4),
      peCount == 0 ? 0 : std::max(1U, builtinCeilDiv(peCount, 8)),
      peCount == 0 ? 0 : std::max(1U, builtinCeilDiv(peCount, 8)),
      peCount == 0 ? 0 : std::max(1U, builtinCeilDiv(peCount, 16)),
  };
}

constexpr BuiltinFuOccurrenceCounts builtinCoverageSpatialFuOccurrences() {
  BuiltinFuOccurrenceCounts result = builtinBalancedFuOccurrences(27);
  result.tokenControl = 8;
  return result;
}

constexpr bool
isValidBuiltinFuOccurrenceCounts(const BuiltinFuOccurrenceCounts &counts,
                                 std::uint32_t peCount) {
  return counts.dedicatedScalarAdd <= peCount && counts.mac <= peCount &&
         counts.vectorCompute <= peCount && counts.loopControl <= peCount &&
         counts.tokenControl <= peCount && counts.vectorAdapter <= peCount &&
         counts.vectorStructural <= peCount && counts.specialMath <= peCount;
}

/// Geometry and occupancy of every private cache the builtin target declares:
/// the two InstructionCore L1 caches of each HostCore and AccCore, and the
/// AccCore SpatialCore memory-path cache. Line size and associativity are one
/// System-wide choice; only capacity and outstanding-miss capacity differ by
/// role. The SpatialCore cache's outstanding-miss capacity is the System
/// memory service's `temporalResidentContexts`, which stays its sole owner.
struct BuiltinPrivateCacheScale final {
  std::uint64_t instructionCoreCacheBytes;
  std::uint64_t spatialMemoryCacheBytes;
  std::uint32_t lineBytes;
  std::uint32_t associativity;
  std::uint32_t hitLatencyCycles;
  std::uint32_t inOrderMissStatusEntries;
  std::uint32_t outOfOrderMissStatusEntries;
};

constexpr bool isPowerOfTwoCacheGeometry(std::uint32_t lineBytes) {
  return lineBytes != 0 && (lineBytes & (lineBytes - 1)) == 0;
}

constexpr bool
isValidBuiltinPrivateCacheScale(const BuiltinPrivateCacheScale &caches) {
  return caches.instructionCoreCacheBytes != 0 &&
         caches.spatialMemoryCacheBytes != 0 &&
         isPowerOfTwoCacheGeometry(caches.lineBytes) &&
         caches.associativity != 0 && caches.hitLatencyCycles != 0 &&
         caches.inOrderMissStatusEntries != 0 &&
         caches.outOfOrderMissStatusEntries != 0 &&
         caches.instructionCoreCacheBytes %
                 (static_cast<std::uint64_t>(caches.lineBytes) *
                  caches.associativity) ==
             0 &&
         caches.spatialMemoryCacheBytes %
                 (static_cast<std::uint64_t>(caches.lineBytes) *
                  caches.associativity) ==
             0;
}

constexpr BuiltinPrivateCacheScale builtinDefaultPrivateCacheScale() {
  return {16 * 1024, 32 * 1024, 64, 4, 1, 4, 8};
}

struct BuiltinTargetScale final {
  std::uint32_t accCoreCount;
  std::uint32_t meshDimension;
  std::uint32_t spatialMeshLanesPerDirection;
  std::uint32_t temporalMeshLanesPerDirection;
  std::uint32_t spatialPeCount;
  std::uint32_t temporalPeCount;
  BuiltinFuOccurrenceCounts spatialFuOccurrences;
  BuiltinFuOccurrenceCounts temporalFuOccurrences;
  std::uint32_t spatialMemoryCount;
  std::uint32_t temporalMemoryCount;
  std::uint32_t temporalResidentContexts;
  /// Depth of every interconnect FIFO: mesh link FIFOs, memory output
  /// staging FIFOs, and the cross-schedule boundary staging FIFOs.
  std::uint32_t interconnectFifoDepth;
  /// Dequeue scheduling discipline of tag-carrying interconnect FIFOs.
  /// Untagged interconnect FIFOs remain strict regardless of this value.
  ::fabric::FifoQueueDiscipline interconnectFifoQueueDiscipline;
  /// Channels each tag-carrying virtual-channel interconnect FIFO guarantees
  /// one pool slot. Zero under strict order; otherwise at least one and at
  /// most the FIFO depth and the Temporal tag value count.
  std::uint32_t interconnectFifoReservedChannels;
  /// Elementary-math formats, behavior, and accuracy exposed by every
  /// SpecialMathFu occurrence. Divide and remainder resources are invariant.
  BuiltinSpecialMathCapabilityProfile specialMathCapabilityProfile;
  LocalMemoryPortVariant localMemoryPortVariant;
  std::uint32_t crossScheduleBoundaryLanesPerTemporalPe;
  std::uint32_t gatewayCount;
  std::uint64_t memoryCapacityBytes;
  /// Private cache realization declared by every InstructionCore and by every
  /// AccCore SpatialCore memory path.
  BuiltinPrivateCacheScale privateCaches;
  /// Firings one memory actor bound to any Operation Engine of the target may
  /// hold outstanding before the oldest retires. It is the engine's own
  /// memory-level parallelism; the access cache's outstanding-miss capacity
  /// and the service's outstanding guarantee still bound the line fills.
  std::uint32_t memoryOperationIssueDepth;
};

constexpr bool isValidBuiltinTargetScale(const BuiltinTargetScale &scale) {
  return scale.accCoreCount != 0 && scale.meshDimension > 1 &&
         scale.spatialMeshLanesPerDirection != 0 &&
         scale.spatialMeshLanesPerDirection <= maximumMeshLanesPerDirection &&
         scale.temporalMeshLanesPerDirection != 0 &&
         scale.temporalMeshLanesPerDirection <= maximumMeshLanesPerDirection &&
         scale.spatialPeCount != 0 && scale.temporalPeCount != 0 &&
         isValidBuiltinFuOccurrenceCounts(scale.spatialFuOccurrences,
                                          scale.spatialPeCount) &&
         isValidBuiltinFuOccurrenceCounts(scale.temporalFuOccurrences,
                                          scale.temporalPeCount) &&
         scale.spatialMemoryCount != 0 && scale.temporalMemoryCount != 0 &&
         scale.temporalResidentContexts != 0 &&
         scale.interconnectFifoDepth != 0 &&
         ((scale.interconnectFifoQueueDiscipline ==
               ::fabric::FifoQueueDiscipline::StrictFifo &&
           scale.interconnectFifoReservedChannels == 0) ||
          (scale.interconnectFifoQueueDiscipline ==
               ::fabric::FifoQueueDiscipline::PerTagVirtualChannel &&
           scale.interconnectFifoReservedChannels != 0 &&
           scale.interconnectFifoReservedChannels <=
               scale.interconnectFifoDepth &&
           scale.interconnectFifoReservedChannels <=
               scale.temporalResidentContexts)) &&
         isValidBuiltinSpecialMathCapabilityProfile(
             scale.specialMathCapabilityProfile) &&
         isValidLocalMemoryPortVariant(scale.localMemoryPortVariant) &&
         scale.crossScheduleBoundaryLanesPerTemporalPe != 0 &&
         scale.gatewayCount != 0 && scale.memoryCapacityBytes != 0 &&
         isValidBuiltinPrivateCacheScale(scale.privateCaches) &&
         scale.memoryOperationIssueDepth != 0;
}

struct BuiltinTargetDescriptor final {
  BuiltinTargetPreset preset;
  llvm::StringLiteral name;
  llvm::StringLiteral templateIdentity;
  std::uint32_t schemaMajor;
  std::uint32_t schemaMinor;
  BuiltinTargetScale scale;
};

inline constexpr BuiltinTargetDescriptor builtinSmallTarget{
    BuiltinTargetPreset::Small,
    "small",
    "loom.adg.builtin.general_purpose",
    8,
    3,
    {4, 4, 2, 2, 12, 4, builtinBalancedFuOccurrences(12),
     builtinBalancedFuOccurrences(4), 1, 1, 2, 2,
     ::fabric::FifoQueueDiscipline::StrictFifo, 0,
     BuiltinSpecialMathCapabilityProfile::PortableProviderClosed,
     LocalMemoryPortVariant::SharedElementVector, 5, 2, 64 * 1024,
     builtinDefaultPrivateCacheScale(),
     ::fabric::serializedMemoryOperationIssueDepth}};

inline constexpr BuiltinTargetDescriptor builtinCoverageTarget{
    BuiltinTargetPreset::Coverage,
    "coverage",
    "loom.adg.builtin.general_purpose",
    8,
    3,
    {8, 6, 2, 2, 27, 9, builtinCoverageSpatialFuOccurrences(),
     builtinBalancedFuOccurrences(9), 4, 4, 4, 4,
     ::fabric::FifoQueueDiscipline::PerTagVirtualChannel, 3,
     BuiltinSpecialMathCapabilityProfile::PortableProviderClosed,
     LocalMemoryPortVariant::SharedElementVector, 5, 4, 256 * 1024,
     builtinDefaultPrivateCacheScale(),
     ::fabric::serializedMemoryOperationIssueDepth}};

inline constexpr BuiltinTargetDescriptor builtinLargeTarget{
    BuiltinTargetPreset::Large,
    "large",
    "loom.adg.builtin.general_purpose",
    8,
    3,
    {16, 8, 2, 2, 48, 16, builtinBalancedFuOccurrences(48),
     builtinBalancedFuOccurrences(16), 4, 4, 8, 16,
     ::fabric::FifoQueueDiscipline::PerTagVirtualChannel, 4,
     BuiltinSpecialMathCapabilityProfile::PortableProviderClosed,
     LocalMemoryPortVariant::SharedElementVector, 5, 8, 1024 * 1024,
     builtinDefaultPrivateCacheScale(),
     ::fabric::serializedMemoryOperationIssueDepth}};

inline llvm::Expected<const BuiltinTargetDescriptor *>
getBuiltinTargetDescriptor(BuiltinTargetPreset preset) {
  switch (preset) {
  case BuiltinTargetPreset::Small:
    return &builtinSmallTarget;
  case BuiltinTargetPreset::Coverage:
    return &builtinCoverageTarget;
  case BuiltinTargetPreset::Large:
    return &builtinLargeTarget;
  }
  return llvm::createStringError(std::errc::invalid_argument,
                                 "invalid builtin target preset enum value");
}

inline const BuiltinTargetDescriptor *
findBuiltinTargetDescriptor(llvm::StringRef templateIdentity,
                            std::uint32_t schemaMajor,
                            std::uint32_t schemaMinor) {
  return templateIdentity == builtinCoverageTarget.templateIdentity &&
                 schemaMajor == builtinCoverageTarget.schemaMajor &&
                 schemaMinor == builtinCoverageTarget.schemaMinor
             ? &builtinCoverageTarget
             : nullptr;
}

} // namespace loom::adg

#endif // LOOM_ADG_BUILTINDESCRIPTOR_H
