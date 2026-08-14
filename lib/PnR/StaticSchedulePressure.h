#ifndef LOOM_LIB_PNR_STATICSCHEDULEPRESSURE_H
#define LOOM_LIB_PNR_STATICSCHEDULEPRESSURE_H

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
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

struct StaticActorCriticality final {
  ::dataflow::ActorRef actor;
  ::dataflow::GraphRef graph;
  std::uint64_t graphCriticalLength = 0;
  std::uint64_t recurrenceCriticalLength = 0;
  bool temporalStateCarrier = false;
};

struct StaticActorEdgeCriticality final {
  ::dataflow::ActorTokenResultRef producer;
  ::dataflow::ActorTokenOperandRef consumer;
  ::dataflow::GraphRef graph;
  std::uint64_t weight = 0;
};

class StaticScheduleAnalysis final {
public:
  llvm::ArrayRef<StaticActorCriticality> actors() const { return actors_; }
  llvm::ArrayRef<StaticActorEdgeCriticality> edges() const { return edges_; }

  const StaticActorCriticality *findActor(::dataflow::ActorRef actor) const;
  std::uint64_t
  edgeWeight(const ::dataflow::ActorTokenResultRef &producer,
             const ::dataflow::ActorTokenOperandRef &consumer) const;

  // Internal frozen analysis storage. This header is private to LoomPnR.
  std::vector<StaticActorCriticality> actors_;
  std::vector<StaticActorEdgeCriticality> edges_;
};

llvm::Expected<StaticScheduleAnalysis> deriveStaticScheduleAnalysis(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::GraphRef> covers);

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
