#ifndef LOOM_LIB_PNR_SPATIALRECURRENCETIMINGPERSISTENT_H
#define LOOM_LIB_PNR_SPATIALRECURRENCETIMINGPERSISTENT_H

#include "SpatialRecurrenceTimingInternal.h"

#include "Dataflow/IR/DataflowStructuralRefs.h"
#include "PnR/SpatialRecurrenceTiming.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace loom::fabric {
class FabricArtifactView;
}

namespace loom::mapping {
class SpatialMappingView;
class TechMappingView;
} // namespace loom::mapping

namespace loom::pnr::detail {

struct FrozenPersistentMemoryUseTiming final {
  ::dataflow::RootedGraphLaunchRef launch;
  std::optional<std::uint64_t> localCompletionCycles;
  bool requiresBoundaryCompletion = false;
};

struct FrozenPersistentRecurrenceActorTiming final {
  std::vector<std::optional<std::uint64_t>> fixedPublications;
  std::optional<std::uint64_t> nextState;
  std::optional<std::uint64_t> memoryIssueLatencyCycles;
  std::vector<FrozenPersistentMemoryUseTiming> memoryUses;
};

struct FrozenPersistentRecurrenceEdgeTiming final {
  SpatialRecurrenceEdgeDisposition disposition =
      SpatialRecurrenceEdgeDisposition::ComputeInternal;
  std::uint64_t transportLatencyCycles = 0;
};

/// Candidate-invariant recurrence demand reconstructed from one exact
/// Dataflow/TechMapping/Fabric/SpatialMapping tuple. Boundary completion is
/// deliberately absent and is supplied only by the enclosing System choice.
class FrozenSpatialRecurrenceTimingDemand final {
public:
  llvm::ArrayRef<FrozenRecurrenceActor> actors() const { return actors_; }
  llvm::ArrayRef<FrozenRecurrenceEdge> edges() const { return edges_; }
  llvm::ArrayRef<FrozenRecurrenceGraph> graphs() const { return graphs_; }
  llvm::ArrayRef<PnrIndex> graphActors() const { return graphActors_; }
  llvm::ArrayRef<PnrIndex> graphEdges() const { return graphEdges_; }
  llvm::ArrayRef<PnrIndex> graphTopologicalActors() const {
    return graphTopologicalActors_;
  }
  llvm::ArrayRef<PnrIndex> feedbackEdges() const { return feedbackEdges_; }
  llvm::ArrayRef<FrozenPersistentRecurrenceActorTiming> actorTimings() const {
    return actorTimings_;
  }
  llvm::ArrayRef<FrozenPersistentRecurrenceEdgeTiming> edgeTimings() const {
    return edgeTimings_;
  }
  std::uint64_t retainedBytes() const;

private:
  std::vector<FrozenRecurrenceActor> actors_;
  std::vector<FrozenRecurrenceEdge> edges_;
  std::vector<FrozenRecurrenceGraph> graphs_;
  std::vector<PnrIndex> graphActors_;
  std::vector<PnrIndex> graphEdges_;
  std::vector<PnrIndex> graphTopologicalActors_;
  std::vector<PnrIndex> feedbackEdges_;
  std::vector<FrozenPersistentRecurrenceActorTiming> actorTimings_;
  std::vector<FrozenPersistentRecurrenceEdgeTiming> edgeTimings_;

  friend llvm::Expected<
      std::shared_ptr<const FrozenSpatialRecurrenceTimingDemand>>
  freezeSpatialMappingGraphRecurrenceTimingDemand(
      const ::dataflow::CanonicalDataflowProgramView &,
      const ::loom::mapping::TechMappingView &,
      const ::loom::fabric::FabricArtifactView &,
      const ::loom::mapping::SpatialMappingView &, ::dataflow::GraphRef);
};

using SpatialBoundaryMemoryCompletionResolver =
    llvm::function_ref<llvm::Expected<std::optional<std::uint64_t>>(
        const ::dataflow::ContextualActorRef &)>;

llvm::Expected<std::shared_ptr<const FrozenSpatialRecurrenceTimingDemand>>
freezeSpatialMappingGraphRecurrenceTimingDemand(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::mapping::SpatialMappingView &mapping,
    ::dataflow::GraphRef graph);

llvm::Expected<SpatialRecurrenceTimingProjection>
projectFrozenSpatialRecurrenceTimingDemand(
    const FrozenSpatialRecurrenceTimingDemand &demand,
    std::optional<::dataflow::RootedGraphLaunchRef> exactLaunch,
    SpatialBoundaryMemoryCompletionResolver boundaryCompletion);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SPATIALRECURRENCETIMINGPERSISTENT_H
