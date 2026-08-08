#ifndef LOOM_PNR_MAPPINGOBJECTIVE_H
#define LOOM_PNR_MAPPINGOBJECTIVE_H

#include "Common/ResolvedPnrPolicy.h"
#include "DSE/Objective.h"
#include "Dataflow/IR/DataflowStructuralRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace loom::mapping {
class SpatialMappingView;
}

namespace loom::pnr {

class FrozenSpatialPnrProblem;
class SpatialCandidateState;
class SystemCandidateState;

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

llvm::Expected<std::uint64_t>
systemMappingViolationValue(const SystemCandidateState &candidate,
                            ResolvedPnrViolationKind kind);

llvm::Expected<std::uint64_t>
systemMappingMeasureValue(const SystemCandidateState &candidate,
                          MappingMeasureKind kind);

/// Preflighted adapter from the exact selected Mapping objective catalog to
/// candidate-owned V/G values. Candidate overloads supply only their native
/// projections; ObjectiveProgram remains the sole ranking and energy owner.
class MappingObjectiveProgram final {
public:
  static llvm::Expected<MappingObjectiveProgram>
  get(const ResolvedObjectiveCatalogs &catalogs,
      const ResolvedPnrObjectiveSelection &selection);

  llvm::Expected<dse::ObjectiveVector>
  evaluate(const SpatialCandidateState &candidate) const;
  llvm::Expected<dse::ObjectiveVector>
  evaluate(const SystemCandidateState &candidate) const;
  llvm::Expected<dse::ObjectiveWideValue>
  selectedEnergy(const dse::ObjectiveVector &vector) const;
  llvm::Expected<dse::ObjectiveSignedDifference>
  selectedEnergyDifference(const dse::ObjectiveVector &left,
                           const dse::ObjectiveVector &right) const;
  llvm::Expected<int>
  compareSelectedRank(const dse::ObjectiveVector &left,
                      llvm::ArrayRef<std::uint8_t> leftCandidateKey,
                      const dse::ObjectiveVector &right,
                      llvm::ArrayRef<std::uint8_t> rightCandidateKey) const;

private:
  MappingObjectiveProgram(dse::ObjectiveProgram program,
                          std::uint64_t selectedViolations,
                          std::uint64_t selectedMeasures,
                          std::uint32_t selectedTotalOrdering,
                          std::uint32_t selectedSearchEnergy)
      : program_(std::move(program)), selectedViolations_(selectedViolations),
        selectedMeasures_(selectedMeasures),
        selectedTotalOrdering_(selectedTotalOrdering),
        selectedSearchEnergy_(selectedSearchEnergy) {}

  dse::ObjectiveProgram program_;
  std::uint64_t selectedViolations_ = 0;
  std::uint64_t selectedMeasures_ = 0;
  std::uint32_t selectedTotalOrdering_ = 0;
  std::uint32_t selectedSearchEnergy_ = 0;
};

llvm::Expected<SpatialMappingTraversalClaimProjection>
projectSpatialMappingTraversalClaims(
    const FrozenSpatialPnrProblem &problem,
    const ::loom::mapping::SpatialMappingView &mapping);

} // namespace loom::pnr

#endif // LOOM_PNR_MAPPINGOBJECTIVE_H
