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
#include <optional>
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

enum class FrozenSpatialGrantPolicyKind : std::uint32_t {
  None,
  FixedPriority,
  RoundRobin,
};

struct FrozenSpatialCapacityDimension final {
  std::uint32_t capacity = 0;
  std::uint32_t initialOccupancy = 0;
};

struct FrozenSpatialResourceState final {
  ::loom::fabric::FabricResourceStateRef reference;
  PnrIndex capacityOffset = 0;
  PnrIndex capacityCount = 0;
};

struct FrozenSpatialResourceClaim final {
  PnrIndex state = 0;
  std::uint32_t dimension = 0;
  std::uint32_t amount = 0;
};

struct FrozenSpatialResourceCommit final {
  std::uint32_t event = 0;
  std::uint32_t transition = 0;
};

struct FrozenSpatialInternalTransaction final {
  PnrIndex claimOffset = 0;
  PnrIndex claimCount = 0;
};

struct FrozenSpatialTimingContract final {
  PnrIndex eventRankOffset = 0;
  PnrIndex eventRankCount = 0;
};

struct FrozenSpatialUsePattern final {
  ::loom::fabric::FabricUsePatternRef reference;
  std::uint32_t requester = 0;
  std::uint32_t eligibility = 0;
  std::uint32_t acquireEvent = 0;
  std::uint32_t releaseEvent = 0;
  std::optional<FrozenSpatialResourceCommit> commit;
  PnrIndex timingContract = 0;
  PnrIndex claimOffset = 0;
  PnrIndex claimCount = 0;
  PnrIndex transactionOffset = 0;
  PnrIndex transactionCount = 0;
};

struct FrozenSpatialResourceOwner final {
  ::loom::fabric::FabricInventoryOwnerRef reference;
  PnrIndex stateOffset = 0;
  PnrIndex stateCount = 0;
  PnrIndex patternOffset = 0;
  PnrIndex patternCount = 0;
  PnrIndex timingOffset = 0;
  PnrIndex timingCount = 0;
  PnrIndex grantOrderOffset = 0;
  PnrIndex grantOrderCount = 0;
  FrozenSpatialGrantPolicyKind grantPolicy = FrozenSpatialGrantPolicyKind::None;
  std::optional<std::uint32_t> roundRobinResetRequester;
  std::uint32_t resourceTransitionCount = 0;
  std::uint32_t requesterCount = 0;
  std::uint32_t eligibilityCount = 0;
  std::uint32_t eventCount = 0;
};

class FrozenSpatialResourceIndex final {
public:
  llvm::ArrayRef<FrozenSpatialResourceOwner> resourceOwners() const {
    return owners_;
  }
  llvm::ArrayRef<FrozenSpatialResourceState> resourceStates() const {
    return states_;
  }
  llvm::ArrayRef<FrozenSpatialCapacityDimension> capacityDimensions() const {
    return capacityDimensions_;
  }
  llvm::ArrayRef<FrozenSpatialUsePattern> usePatterns() const {
    return patterns_;
  }
  llvm::ArrayRef<FrozenSpatialResourceClaim> claims() const { return claims_; }
  llvm::ArrayRef<FrozenSpatialInternalTransaction>
  internalTransactions() const {
    return internalTransactions_;
  }
  llvm::ArrayRef<PnrIndex> transactionClaims() const {
    return transactionClaims_;
  }
  llvm::ArrayRef<FrozenSpatialTimingContract> timingContracts() const {
    return timingContracts_;
  }
  llvm::ArrayRef<std::uint32_t> eventRanks() const { return eventRanks_; }
  llvm::ArrayRef<std::uint32_t> grantRequesterOrder() const {
    return grantRequesterOrder_;
  }

private:
  std::vector<FrozenSpatialResourceOwner> owners_;
  std::vector<FrozenSpatialResourceState> states_;
  std::vector<FrozenSpatialCapacityDimension> capacityDimensions_;
  std::vector<FrozenSpatialUsePattern> patterns_;
  std::vector<FrozenSpatialResourceClaim> claims_;
  std::vector<FrozenSpatialInternalTransaction> internalTransactions_;
  std::vector<PnrIndex> transactionClaims_;
  std::vector<FrozenSpatialTimingContract> timingContracts_;
  std::vector<std::uint32_t> eventRanks_;
  std::vector<std::uint32_t> grantRequesterOrder_;

  friend class FrozenSpatialResourceIndexBuilder;
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
  PnrIndex resourceStateOffset = 0;
  PnrIndex resourceStateCount = 0;
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
  llvm::ArrayRef<PnrIndex> traversalResourceStates() const {
    return traversalResourceStates_;
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
  std::vector<PnrIndex> traversalResourceStates_;
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
  const FrozenSpatialResourceIndex &resources() const { return resources_; }
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
                          FrozenSpatialResourceIndex resources,
                          FrozenSpatialRoutingGraph routing,
                          FrozenSpatialPnrCacheKey cacheKey)
      : dataflowIdentity_(std::move(dataflowIdentity)),
        techMappingIdentity_(std::move(techMappingIdentity)),
        fabricIdentity_(std::move(fabricIdentity)),
        constraintSetIdentity_(std::move(constraintSetIdentity)),
        config_(std::move(config)), workBudget_(std::move(workBudget)),
        constraints_(std::move(constraints)),
        realizations_(std::move(realizations)),
        resources_(std::move(resources)), routing_(std::move(routing)),
        cacheKey_(cacheKey) {}

  ArtifactIdentity dataflowIdentity_;
  ArtifactIdentity techMappingIdentity_;
  ArtifactIdentity fabricIdentity_;
  ArtifactIdentity constraintSetIdentity_;
  ResolvedPnrConfigView config_;
  std::vector<DeterministicWorkBudgetEntry> workBudget_;
  FrozenConstraintIndex constraints_;
  FrozenSpatialRealizationIndex realizations_;
  FrozenSpatialResourceIndex resources_;
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
