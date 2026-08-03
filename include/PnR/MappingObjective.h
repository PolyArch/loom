#ifndef LOOM_PNR_MAPPINGOBJECTIVE_H
#define LOOM_PNR_MAPPINGOBJECTIVE_H

#include "Common/ResolvedPnrPolicy.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom::pnr {

class SpatialCandidateState;

struct MappingObjectiveRegistryDescriptor final {
  llvm::StringRef identity;
  std::uint32_t schemaMajor;
  std::uint32_t schemaMinor;
};

struct MappingViolationDescriptor final {
  ResolvedPnrViolationKind kind;
  llvm::StringRef spelling;
};

enum class MappingMeasureKind : std::uint32_t {
#define LOOM_MAPPING_MEASURE(Name, Ordinal, DisplayName) Name = Ordinal,
#include "Common/MappingObjectiveKinds.def"
};

inline constexpr std::uint32_t mappingMeasureKindCount = 0
#define LOOM_MAPPING_MEASURE(Name, Ordinal, DisplayName) +1
#include "Common/MappingObjectiveKinds.def"
    ;

struct MappingMeasureDescriptor final {
  MappingMeasureKind kind;
  llvm::StringRef spelling;
};

const MappingObjectiveRegistryDescriptor &mappingObjectiveRegistryDescriptor();
llvm::ArrayRef<MappingViolationDescriptor> mappingViolationDescriptors();
llvm::ArrayRef<MappingMeasureDescriptor> mappingMeasureDescriptors();

/// Projects one exact Mapping-owned violation magnitude. A kind without a
/// complete CandidateState owner returns ObjectiveUnavailable rather than a
/// provisional value.
llvm::Expected<std::uint64_t>
spatialMappingViolationValue(const SpatialCandidateState &candidate,
                             ResolvedPnrViolationKind kind);

/// Projects one Mapping-owned domain-independent measure from the exact
/// candidate state. The candidate remains the sole owner of incremental route
/// occupancy; this query does not cache or reconstruct that state.
std::uint64_t spatialMappingMeasureValue(const SpatialCandidateState &candidate,
                                         MappingMeasureKind kind);

} // namespace loom::pnr

#endif // LOOM_PNR_MAPPINGOBJECTIVE_H
