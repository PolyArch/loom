#ifndef LOOM_PNR_MAPPINGOBJECTIVE_H
#define LOOM_PNR_MAPPINGOBJECTIVE_H

#include "Common/ResolvedPnrPolicy.h"
#include "Dataflow/IR/DataflowStructuralRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::mapping {
class SpatialMappingView;
}

namespace loom::pnr {

class FrozenSpatialPnrProblem;
class SpatialCandidateState;

struct SpatialMappingTraversalClaimContribution final {
  ::dataflow::CanonicalGraphProducerEndpointRef logicalNet;
  std::uint64_t value = 0;
};

/// Cold, removable reconstruction of the Mapping-owned
/// TotalSelectedTraversalClaim measure. Each claim is counted once per
/// selected logical-net RouteTree, matching CandidateState's incremental
/// net-by-claim owner. The projection is never persisted or used as legality.
struct SpatialMappingTraversalClaimProjection final {
  std::uint64_t total = 0;
  std::vector<SpatialMappingTraversalClaimContribution> logicalNets;
};

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

/// Returns whether Spatial CandidateState has the complete unique owner for
/// this violation projection. Search preflight uses this before allocating a
/// candidate; unavailable dimensions cannot be replaced by a provisional
/// value.
bool spatialMappingViolationAvailable(ResolvedPnrViolationKind kind);

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

llvm::Expected<SpatialMappingTraversalClaimProjection>
projectSpatialMappingTraversalClaims(
    const FrozenSpatialPnrProblem &problem,
    const ::loom::mapping::SpatialMappingView &mapping);

} // namespace loom::pnr

#endif // LOOM_PNR_MAPPINGOBJECTIVE_H
