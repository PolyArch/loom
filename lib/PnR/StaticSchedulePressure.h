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

llvm::Expected<std::vector<std::uint64_t>> projectStaticSchedulePressureByGraph(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::mapping::SpatialMappingView &mapping);

} // namespace detail
} // namespace loom::pnr

#endif // LOOM_LIB_PNR_STATICSCHEDULEPRESSURE_H
