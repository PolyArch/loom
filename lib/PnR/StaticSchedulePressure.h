#ifndef LOOM_LIB_PNR_STATICSCHEDULEPRESSURE_H
#define LOOM_LIB_PNR_STATICSCHEDULEPRESSURE_H

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowStaticScheduleAnalysis.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "PnR/PnrIndex.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace loom::pnr {

class FrozenSpatialRealizationIndex;
class SpatialCandidateState;

namespace detail {

using StaticActorCriticality = ::dataflow::StaticActorCriticality;
using StaticActorEdgeCriticality = ::dataflow::StaticActorEdgeCriticality;
using StaticRecurrenceFeedback = ::dataflow::StaticRecurrenceFeedback;
using StaticGraphRecurrenceTopology =
    ::dataflow::StaticGraphRecurrenceTopology;
using StaticScheduleAnalysis = ::dataflow::StaticScheduleAnalysis;
using ::dataflow::deriveStaticScheduleAnalysis;

struct SpatialSchedulePressureEdge final {
  PnrIndex firstRoot = 0;
  PnrIndex secondRoot = 0;
  std::uint64_t weight = 0;
};

class SpatialSchedulePressureIndex final {
public:
  static llvm::Expected<std::shared_ptr<const SpatialSchedulePressureIndex>>
  build(const ::dataflow::CanonicalDataflowProgramView &dataflow,
        const ::loom::mapping::TechMappingView &techMapping,
        const FrozenSpatialRealizationIndex &realizations);

  std::uint64_t computePlacementContribution(PnrIndex placement) const;
  std::uint64_t memoryPlacementContribution(PnrIndex placement) const;
  /// Recurrence-critical work the placement serializes because its occurrence
  /// issues resident instructions in rotation. A Spatial-schedule placement
  /// contributes zero; a Temporal-schedule placement contributes the summed
  /// `recurrenceCriticalLength` of the actors it hosts. This is the same
  /// analysis fact the schedule-pressure contribution already charges, kept
  /// separately so a total ordering can rank recurrence serialization on its
  /// own level without a second criticality owner.
  std::uint64_t
  computePlacementRecurrenceTemporalBinding(PnrIndex placement) const;
  std::uint64_t
  memoryPlacementRecurrenceTemporalBinding(PnrIndex placement) const;
  std::uint64_t
  edgeWeight(const ::dataflow::ActorTokenResultRef &producer,
             const ::dataflow::ActorTokenOperandRef &consumer) const {
    return analysis_.edgeWeight(producer, consumer);
  }
  llvm::ArrayRef<SpatialSchedulePressureEdge> edges() const { return edges_; }
  const StaticScheduleAnalysis &analysis() const { return analysis_; }
  llvm::ArrayRef<PnrIndex> incidentEdges(PnrIndex root) const;
  PnrIndex computeRootCount() const { return computeRootCount_; }
  PnrIndex rootCount() const { return rootCount_; }

private:
  StaticScheduleAnalysis analysis_;
  std::vector<std::uint64_t> computePlacementContributions_;
  std::vector<std::uint64_t> memoryPlacementContributions_;
  std::vector<std::uint64_t> computePlacementRecurrenceTemporalBindings_;
  std::vector<std::uint64_t> memoryPlacementRecurrenceTemporalBindings_;
  std::vector<SpatialSchedulePressureEdge> edges_;
  std::vector<PnrIndex> incidenceOffsets_;
  std::vector<PnrIndex> incidenceEdges_;
  PnrIndex computeRootCount_ = 0;
  PnrIndex rootCount_ = 0;
};

llvm::Expected<std::uint64_t>
measureStaticSchedulePressure(const SpatialCandidateState &candidate);

llvm::Expected<std::uint64_t> projectStaticSchedulePressureAfterComputeChange(
    const SpatialCandidateState &candidate, PnrIndex realization,
    PnrIndex placement);

llvm::Expected<std::uint64_t> projectStaticSchedulePressureAfterMemoryChange(
    const SpatialCandidateState &candidate, PnrIndex realization,
    PnrIndex placement);

llvm::Expected<std::uint64_t> measureRecurrenceTemporalBindingPressure(
    const SpatialCandidateState &candidate);

llvm::Expected<std::uint64_t>
projectRecurrenceTemporalBindingPressureAfterComputeChange(
    const SpatialCandidateState &candidate, PnrIndex realization,
    PnrIndex placement);

llvm::Expected<std::uint64_t>
projectRecurrenceTemporalBindingPressureAfterMemoryChange(
    const SpatialCandidateState &candidate, PnrIndex realization,
    PnrIndex placement);

/// Per-covered-graph partition of the two placement-derived measures owned by
/// this index. Both vectors are indexed by the TechMapping cover ordinal.
struct GraphSchedulePressureProjection final {
  std::vector<std::uint64_t> staticSchedulePressure;
  std::vector<std::uint64_t> recurrenceTemporalBindingPressure;
};

llvm::Expected<GraphSchedulePressureProjection>
projectStaticSchedulePressureByGraph(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::mapping::SpatialMappingView &mapping);

} // namespace detail
} // namespace loom::pnr

#endif // LOOM_LIB_PNR_STATICSCHEDULEPRESSURE_H
