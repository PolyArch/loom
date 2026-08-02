#ifndef LOOM_PNR_SPATIALPNRPROBLEM_H
#define LOOM_PNR_SPATIALPNRPROBLEM_H

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "PnR/FrozenConstraintIndex.h"
#include "PnR/PnrConfig.h"
#include "PnR/PnrIndex.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace loom::pnr {

struct FrozenSpatialComputePlacement final {
  PnrIndex realization = 0;
  ::loom::fabric::FabricFuOccurrenceRef fu;
  ::loom::fabric::FabricPeOccurrenceRef parentPe;
  PnrIndex contextOffset = 0;
  PnrIndex contextCount = 0;
};

struct FrozenSpatialComputeRealization final {
  ::loom::mapping::TechComputeRealizationRef reference;
  ::loom::fabric::FabricFuCapabilityTemplateRef capabilityTemplate;
  PnrIndex actorOffset = 0;
  PnrIndex actorCount = 0;
  PnrIndex placementOffset = 0;
  PnrIndex placementCount = 0;
};

struct FrozenSpatialMemoryPlacement final {
  PnrIndex realization = 0;
  ::loom::fabric::FabricMemoryOccurrenceRef memory;
};

struct FrozenSpatialMemoryActorBinding final {
  ::dataflow::ActorRef actor;
  ::loom::fabric::FabricMemoryEngineTemplateOperationPortRef operationPort;
};

struct FrozenSpatialMemoryRealization final {
  ::loom::mapping::TechMemoryRealizationRef reference;
  ::loom::fabric::FabricMemoryEngineTemplateRef engine;
  PnrIndex actorOffset = 0;
  PnrIndex actorCount = 0;
  PnrIndex placementOffset = 0;
  PnrIndex placementCount = 0;
};

class FrozenSpatialRealizationIndex final {
public:
  llvm::ArrayRef<FrozenSpatialComputeRealization> computeRealizations() const {
    return computeRealizations_;
  }
  llvm::ArrayRef<::dataflow::ActorRef> computeActors() const {
    return computeActors_;
  }
  llvm::ArrayRef<FrozenSpatialComputePlacement> computePlacements() const {
    return computePlacements_;
  }
  llvm::ArrayRef<::loom::fabric::InstructionContextRef>
  computeInstructionContexts() const {
    return computeInstructionContexts_;
  }
  llvm::ArrayRef<FrozenSpatialMemoryRealization> memoryRealizations() const {
    return memoryRealizations_;
  }
  llvm::ArrayRef<FrozenSpatialMemoryActorBinding> memoryActors() const {
    return memoryActors_;
  }
  llvm::ArrayRef<FrozenSpatialMemoryPlacement> memoryPlacements() const {
    return memoryPlacements_;
  }

private:
  std::vector<FrozenSpatialComputeRealization> computeRealizations_;
  std::vector<::dataflow::ActorRef> computeActors_;
  std::vector<FrozenSpatialComputePlacement> computePlacements_;
  std::vector<::loom::fabric::InstructionContextRef>
      computeInstructionContexts_;
  std::vector<FrozenSpatialMemoryRealization> memoryRealizations_;
  std::vector<FrozenSpatialMemoryActorBinding> memoryActors_;
  std::vector<FrozenSpatialMemoryPlacement> memoryPlacements_;

  friend class FrozenSpatialPnrProblemBuilder;
};

struct FrozenSpatialRoutingEndpoint final {
  ::loom::fabric::FabricTransportEndpointRef reference;
  ::loom::fabric::FabricPortDirection direction;
  ::fabric::DataPathType dataPath;
};

struct FrozenSpatialTraversal final {
  ::loom::fabric::FabricPhysicalTraversalRef reference;
  PnrIndex sourceOffset = 0;
  PnrIndex sourceCount = 0;
  PnrIndex destinationOffset = 0;
  PnrIndex destinationCount = 0;
};

struct FrozenSpatialRoutingArc final {
  PnrIndex target = 0;
  PnrIndex traversal = 0;
  std::uint32_t payloadCapacityBits = 0;
  std::uint32_t tagCapacityBits = 0;
};

class FrozenSpatialRoutingGraph final {
public:
  llvm::ArrayRef<FrozenSpatialRoutingEndpoint> routingEndpoints() const {
    return endpoints_;
  }
  llvm::ArrayRef<FrozenSpatialTraversal> traversals() const {
    return traversals_;
  }
  llvm::ArrayRef<PnrIndex> traversalEndpoints() const {
    return traversalEndpoints_;
  }
  llvm::ArrayRef<PnrIndex> adjacencyOffsets() const {
    return adjacencyOffsets_;
  }
  llvm::ArrayRef<PnrIndex> arcSources() const { return arcSources_; }
  llvm::ArrayRef<FrozenSpatialRoutingArc> routingArcs() const { return arcs_; }

private:
  std::vector<FrozenSpatialRoutingEndpoint> endpoints_;
  std::vector<FrozenSpatialTraversal> traversals_;
  std::vector<PnrIndex> traversalEndpoints_;
  std::vector<PnrIndex> adjacencyOffsets_;
  std::vector<PnrIndex> arcSources_;
  std::vector<FrozenSpatialRoutingArc> arcs_;

  friend class FrozenSpatialPnrProblemBuilder;
};

class FrozenSpatialPnrCacheKey final {
public:
  using Storage = std::array<std::uint8_t, 32>;

  const Storage &bytes() const { return bytes_; }

  friend bool operator==(const FrozenSpatialPnrCacheKey &lhs,
                         const FrozenSpatialPnrCacheKey &rhs) {
    return lhs.bytes_ == rhs.bytes_;
  }
  friend bool operator!=(const FrozenSpatialPnrCacheKey &lhs,
                         const FrozenSpatialPnrCacheKey &rhs) {
    return !(lhs == rhs);
  }

private:
  explicit FrozenSpatialPnrCacheKey(Storage bytes) : bytes_(bytes) {}

  Storage bytes_;

  friend class FrozenSpatialPnrProblemBuilder;
};

class FrozenSpatialPnrProblem final {
public:
  FrozenSpatialPnrProblem(const FrozenSpatialPnrProblem &) = delete;
  FrozenSpatialPnrProblem(FrozenSpatialPnrProblem &&) = delete;
  FrozenSpatialPnrProblem &operator=(const FrozenSpatialPnrProblem &) = delete;
  FrozenSpatialPnrProblem &operator=(FrozenSpatialPnrProblem &&) = delete;

  const ArtifactIdentity &dataflowIdentity() const { return dataflowIdentity_; }
  const ArtifactIdentity &techMappingIdentity() const {
    return techMappingIdentity_;
  }
  const ArtifactIdentity &fabricIdentity() const { return fabricIdentity_; }
  const ArtifactIdentity &constraintSetIdentity() const {
    return constraintSetIdentity_;
  }
  const ResolvedPnrConfigView &config() const { return config_; }
  llvm::ArrayRef<DeterministicWorkBudgetEntry> workBudget() const {
    return workBudget_;
  }
  const FrozenConstraintIndex &constraints() const { return constraints_; }
  const FrozenSpatialRealizationIndex &realizations() const {
    return realizations_;
  }
  const FrozenSpatialRoutingGraph &routing() const { return routing_; }
  const FrozenSpatialPnrCacheKey &cacheKey() const { return cacheKey_; }

private:
  FrozenSpatialPnrProblem(ArtifactIdentity dataflowIdentity,
                          ArtifactIdentity techMappingIdentity,
                          ArtifactIdentity fabricIdentity,
                          ArtifactIdentity constraintSetIdentity,
                          ResolvedPnrConfigView config,
                          std::vector<DeterministicWorkBudgetEntry> workBudget,
                          FrozenConstraintIndex constraints,
                          FrozenSpatialRealizationIndex realizations,
                          FrozenSpatialRoutingGraph routing,
                          FrozenSpatialPnrCacheKey cacheKey)
      : dataflowIdentity_(std::move(dataflowIdentity)),
        techMappingIdentity_(std::move(techMappingIdentity)),
        fabricIdentity_(std::move(fabricIdentity)),
        constraintSetIdentity_(std::move(constraintSetIdentity)),
        config_(std::move(config)), workBudget_(std::move(workBudget)),
        constraints_(std::move(constraints)),
        realizations_(std::move(realizations)), routing_(std::move(routing)),
        cacheKey_(cacheKey) {}

  ArtifactIdentity dataflowIdentity_;
  ArtifactIdentity techMappingIdentity_;
  ArtifactIdentity fabricIdentity_;
  ArtifactIdentity constraintSetIdentity_;
  ResolvedPnrConfigView config_;
  std::vector<DeterministicWorkBudgetEntry> workBudget_;
  FrozenConstraintIndex constraints_;
  FrozenSpatialRealizationIndex realizations_;
  FrozenSpatialRoutingGraph routing_;
  FrozenSpatialPnrCacheKey cacheKey_;

  friend class FrozenSpatialPnrProblemBuilder;
};

using FrozenSpatialPnrProblemHandle =
    std::shared_ptr<const FrozenSpatialPnrProblem>;

llvm::Expected<FrozenSpatialPnrProblemHandle> freezeSpatialPnrProblem(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ResolvedPnrConfigView &config,
    const ::loom::mapping::SpatialMappingConstraintSetView &constraints);

llvm::Error revalidateFrozenSpatialPnrCacheHit(
    const FrozenSpatialPnrProblem &problem,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ResolvedPnrConfigView &config,
    const ::loom::mapping::SpatialMappingConstraintSetView &constraints);

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALPNRPROBLEM_H
