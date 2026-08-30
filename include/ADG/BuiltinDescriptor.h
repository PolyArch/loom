#ifndef LOOM_ADG_BUILTINDESCRIPTOR_H
#define LOOM_ADG_BUILTINDESCRIPTOR_H

#include "ADG/MemoryLibrary.h"

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
  LocalMemoryPortVariant localMemoryPortVariant;
  std::uint32_t crossScheduleBoundaryLanesPerTemporalPe;
  std::uint32_t gatewayCount;
  std::uint64_t memoryCapacityBytes;
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
         isValidLocalMemoryPortVariant(scale.localMemoryPortVariant) &&
         scale.crossScheduleBoundaryLanesPerTemporalPe != 0 &&
         scale.gatewayCount != 0 && scale.memoryCapacityBytes != 0;
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
    0,
    {4, 4, 2, 2, 12, 4, builtinBalancedFuOccurrences(12),
     builtinBalancedFuOccurrences(4), 1, 1, 2, 2,
     ::fabric::FifoQueueDiscipline::StrictFifo,
     LocalMemoryPortVariant::SharedElementVector, 5, 2, 64 * 1024}};

inline constexpr BuiltinTargetDescriptor builtinCoverageTarget{
    BuiltinTargetPreset::Coverage,
    "coverage",
    "loom.adg.builtin.general_purpose",
    8,
    0,
    {8, 6, 2, 2, 27, 9, builtinCoverageSpatialFuOccurrences(),
     builtinBalancedFuOccurrences(9), 4, 4, 4, 4,
     ::fabric::FifoQueueDiscipline::StrictFifo,
     LocalMemoryPortVariant::SharedElementVector, 5, 4, 256 * 1024}};

inline constexpr BuiltinTargetDescriptor builtinLargeTarget{
    BuiltinTargetPreset::Large,
    "large",
    "loom.adg.builtin.general_purpose",
    8,
    0,
    {16, 8, 2, 2, 48, 16, builtinBalancedFuOccurrences(48),
     builtinBalancedFuOccurrences(16), 4, 4, 8, 16,
     ::fabric::FifoQueueDiscipline::StrictFifo,
     LocalMemoryPortVariant::SharedElementVector, 5, 8, 1024 * 1024}};

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
