#ifndef LOOM_LIB_PNR_SPATIALRECURRENCETIMINGINTERNAL_H
#define LOOM_LIB_PNR_SPATIALRECURRENCETIMINGINTERNAL_H

#include "PnR/PnrIndex.h"
#include "PnR/SpatialRecurrenceTiming.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace dataflow {
class CanonicalDataflowProgramView;
}

namespace loom::fabric {
class FabricArtifactView;
}

namespace loom::mapping {
class TechMappingView;
}

namespace loom::pnr {

class FrozenSpatialHandshakeIndex;
class FrozenSpatialMemoryIndex;
class FrozenSpatialPnrProblem;
class FrozenSpatialRealizationIndex;
class FrozenSpatialTransferIndex;
class RouteTreeState;
class SpatialCandidateState;

namespace detail {

class StaticScheduleAnalysis;

enum class FrozenRecurrenceActorOwnerKind : std::uint8_t {
  Compute,
  Memory,
};

struct FrozenRecurrenceActor final {
  ::dataflow::ActorRef actor;
  ::dataflow::GraphRef graph;
  FrozenRecurrenceActorOwnerKind ownerKind =
      FrozenRecurrenceActorOwnerKind::Compute;
  PnrIndex owner = 0;
  PnrIndex ownerActor = 0;
};

enum class FrozenRecurrenceEdgeDisposition : std::uint8_t {
  ComputeInternal,
  MemoryInternal,
  Residual,
};

struct FrozenRecurrenceEdge final {
  ::dataflow::ActorTokenResultRef producer;
  ::dataflow::ActorTokenOperandRef consumer;
  ::dataflow::GraphRef graph;
  PnrIndex producerActor = 0;
  PnrIndex consumerActor = 0;
  FrozenRecurrenceEdgeDisposition disposition =
      FrozenRecurrenceEdgeDisposition::ComputeInternal;
  PnrIndex logicalNet = getInvalidPnrIndex();
  PnrIndex sink = getInvalidPnrIndex();
  bool feedback = false;
  std::uint64_t dependenceDistance = 0;
};

struct FrozenComputeActorArchitecturalTiming final {
  PnrIndex placement = 0;
  PnrIndex resultOffset = 0;
  PnrIndex resultCount = 0;
  std::optional<std::uint32_t> carryNextStateLatencyCycles;
};

struct FrozenRecurrenceGraph final {
  ::dataflow::GraphRef graph;
  bool nonFeedbackAcyclic = true;
  PnrIndex actorOffset = 0;
  PnrIndex actorCount = 0;
  PnrIndex edgeOffset = 0;
  PnrIndex edgeCount = 0;
  PnrIndex topologicalOffset = 0;
  PnrIndex topologicalCount = 0;
};

class SpatialRecurrenceTimingIndex final {
public:
  static llvm::Expected<std::shared_ptr<const SpatialRecurrenceTimingIndex>>
  build(const ::dataflow::CanonicalDataflowProgramView &dataflow,
        const ::loom::mapping::TechMappingView &techMapping,
        const ::loom::fabric::FabricArtifactView &fabric,
        const FrozenSpatialRealizationIndex &realizations,
        const FrozenSpatialMemoryIndex &memory,
        const FrozenSpatialTransferIndex &transfers,
        const FrozenSpatialHandshakeIndex &handshake,
        const StaticScheduleAnalysis &schedule);

  llvm::ArrayRef<FrozenRecurrenceActor> actors() const { return actors_; }
  llvm::ArrayRef<FrozenRecurrenceEdge> edges() const { return edges_; }
  llvm::ArrayRef<FrozenRecurrenceGraph> graphs() const { return graphs_; }
  llvm::ArrayRef<PnrIndex> graphActors() const { return graphActors_; }
  llvm::ArrayRef<PnrIndex> graphEdges() const { return graphEdges_; }
  llvm::ArrayRef<PnrIndex> graphTopologicalActors() const {
    return graphTopologicalActors_;
  }
  llvm::ArrayRef<PnrIndex> feedbackEdges() const { return feedbackEdges_; }
  llvm::ArrayRef<PnrIndex> computeTimingOffsets() const {
    return computeTimingOffsets_;
  }
  llvm::ArrayRef<FrozenComputeActorArchitecturalTiming>
  computeTimings() const {
    return computeTimings_;
  }
  llvm::ArrayRef<std::optional<std::uint32_t>>
  computeResultPublicationLatencies() const {
    return computeResultPublicationLatencies_;
  }

private:
  std::vector<FrozenRecurrenceActor> actors_;
  std::vector<FrozenRecurrenceEdge> edges_;
  std::vector<FrozenRecurrenceGraph> graphs_;
  std::vector<PnrIndex> graphActors_;
  std::vector<PnrIndex> graphEdges_;
  std::vector<PnrIndex> graphTopologicalActors_;
  std::vector<PnrIndex> feedbackEdges_;
  std::vector<PnrIndex> computeTimingOffsets_;
  std::vector<FrozenComputeActorArchitecturalTiming> computeTimings_;
  std::vector<std::optional<std::uint32_t>>
      computeResultPublicationLatencies_;
};

llvm::Expected<SpatialRecurrenceTimingProjection>
projectSpatialRecurrenceTiming(
    const SpatialCandidateState &candidate,
    llvm::ArrayRef<const RouteTreeState *> routeTrees);

} // namespace detail
} // namespace loom::pnr

#endif // LOOM_LIB_PNR_SPATIALRECURRENCETIMINGINTERNAL_H
