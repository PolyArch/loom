#ifndef LOOM_PNR_SPATIALPNRPROBLEM_H
#define LOOM_PNR_SPATIALPNRPROBLEM_H

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Identity/FabricHandshake.h"
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
#include <variant>
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
  llvm::ArrayRef<PnrIndex> computeActorRealizations() const {
    return computeActorRealizations_;
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
  llvm::ArrayRef<PnrIndex> memoryActorRealizations() const {
    return memoryActorRealizations_;
  }
  llvm::ArrayRef<FrozenSpatialMemoryPlacement> memoryPlacements() const {
    return memoryPlacements_;
  }

private:
  std::vector<FrozenSpatialComputeRealization> computeRealizations_;
  std::vector<::dataflow::ActorRef> computeActors_;
  std::vector<PnrIndex> computeActorRealizations_;
  std::vector<FrozenSpatialComputePlacement> computePlacements_;
  std::vector<::loom::fabric::InstructionContextRef>
      computeInstructionContexts_;
  std::vector<FrozenSpatialMemoryRealization> memoryRealizations_;
  std::vector<FrozenSpatialMemoryActorBinding> memoryActors_;
  std::vector<PnrIndex> memoryActorRealizations_;
  std::vector<FrozenSpatialMemoryPlacement> memoryPlacements_;

  friend class FrozenSpatialPnrProblemBuilder;
  friend class FrozenSpatialPortIndexBuilder;
};

enum class FrozenSpatialPortDemandKind : std::uint32_t {
  Compute,
  Memory,
};

using FrozenSpatialActorTerminalRef =
    std::variant<::dataflow::ActorTokenResultRef,
                 ::dataflow::ActorTokenOperandRef>;
using FrozenSpatialTemplateTerminalRef =
    std::variant<::loom::fabric::FabricFuTemplatePortRef,
                 ::loom::fabric::FabricMemoryEngineTemplateEndpointRef>;
using FrozenSpatialGraphBoundaryTerminalRef =
    std::variant<::dataflow::GraphIngressTokenRef,
                 ::dataflow::GraphEgressTokenRef>;

struct FrozenSpatialPortDemand final {
  FrozenSpatialPortDemandKind kind = FrozenSpatialPortDemandKind::Compute;
  PnrIndex realization = 0;
  FrozenSpatialActorTerminalRef terminal;
  FrozenSpatialTemplateTerminalRef templateTerminal;
  std::uint32_t payloadWidthBits = 0;
  PnrIndex logicalNet = 0;
  PnrIndex placementDomainOffset = 0;
  PnrIndex placementDomainCount = 0;
};

struct FrozenSpatialPortPlacementDomain final {
  PnrIndex placement = 0;
  PnrIndex attachmentOptionOffset = 0;
  PnrIndex attachmentOptionCount = 0;
};

enum class FrozenSpatialAttachmentOwnerKind : std::uint32_t {
  PlacementDomain,
  GraphBoundary,
};

struct FrozenSpatialAttachmentOption final {
  PnrIndex endpoint = 0;
  std::optional<PnrIndex> localTraversal;
  FrozenSpatialAttachmentOwnerKind ownerKind =
      FrozenSpatialAttachmentOwnerKind::PlacementDomain;
  PnrIndex owner = 0;
};

struct FrozenSpatialGraphBoundary final {
  FrozenSpatialGraphBoundaryTerminalRef terminal;
  std::uint32_t payloadWidthBits = 0;
  PnrIndex logicalNet = 0;
  PnrIndex attachmentOptionOffset = 0;
  PnrIndex attachmentOptionCount = 0;
};

class FrozenSpatialPortIndex final {
public:
  llvm::ArrayRef<FrozenSpatialPortDemand> portDemands() const {
    return portDemands_;
  }
  llvm::ArrayRef<FrozenSpatialPortPlacementDomain> placementDomains() const {
    return placementDomains_;
  }
  llvm::ArrayRef<FrozenSpatialAttachmentOption> attachmentOptions() const {
    return attachmentOptions_;
  }
  llvm::ArrayRef<FrozenSpatialGraphBoundary> graphBoundaries() const {
    return graphBoundaries_;
  }
  llvm::ArrayRef<PnrIndex> computeRealizationDemandOffsets() const {
    return computeRealizationDemandOffsets_;
  }
  llvm::ArrayRef<PnrIndex> computeRealizationDemands() const {
    return computeRealizationDemands_;
  }
  llvm::ArrayRef<PnrIndex> memoryRealizationDemandOffsets() const {
    return memoryRealizationDemandOffsets_;
  }
  llvm::ArrayRef<PnrIndex> memoryRealizationDemands() const {
    return memoryRealizationDemands_;
  }
  llvm::ArrayRef<PnrIndex> endpointAttachmentOffsets() const {
    return endpointAttachmentOffsets_;
  }
  llvm::ArrayRef<PnrIndex> endpointAttachmentOptions() const {
    return endpointAttachmentOptions_;
  }

private:
  std::vector<FrozenSpatialPortDemand> portDemands_;
  std::vector<FrozenSpatialPortPlacementDomain> placementDomains_;
  std::vector<FrozenSpatialAttachmentOption> attachmentOptions_;
  std::vector<FrozenSpatialGraphBoundary> graphBoundaries_;
  std::vector<PnrIndex> computeRealizationDemandOffsets_;
  std::vector<PnrIndex> computeRealizationDemands_;
  std::vector<PnrIndex> memoryRealizationDemandOffsets_;
  std::vector<PnrIndex> memoryRealizationDemands_;
  std::vector<PnrIndex> endpointAttachmentOffsets_;
  std::vector<PnrIndex> endpointAttachmentOptions_;

  friend class FrozenSpatialPortIndexBuilder;
};

enum class FrozenSpatialTerminalBindingKind : std::uint32_t {
  PortDemand,
  GraphBoundary,
};

struct FrozenSpatialTerminalBinding final {
  FrozenSpatialTerminalBindingKind kind =
      FrozenSpatialTerminalBindingKind::PortDemand;
  PnrIndex index = 0;
};

struct FrozenSpatialLogicalNet final {
  ::dataflow::CanonicalGraphProducerEndpointRef producer;
  PnrIndex sinkOffset = 0;
  PnrIndex sinkCount = 0;
};

class FrozenSpatialTransferIndex final {
public:
  llvm::ArrayRef<FrozenSpatialLogicalNet> logicalNets() const {
    return logicalNets_;
  }
  llvm::ArrayRef<::dataflow::CanonicalGraphConsumerEndpointRef>
  logicalNetSinks() const {
    return logicalNetSinks_;
  }
  llvm::ArrayRef<FrozenSpatialTerminalBinding>
  logicalNetSourceBindings() const {
    return logicalNetSourceBindings_;
  }
  llvm::ArrayRef<FrozenSpatialTerminalBinding> logicalNetSinkBindings() const {
    return logicalNetSinkBindings_;
  }

private:
  std::vector<FrozenSpatialLogicalNet> logicalNets_;
  std::vector<::dataflow::CanonicalGraphConsumerEndpointRef> logicalNetSinks_;
  std::vector<FrozenSpatialTerminalBinding> logicalNetSourceBindings_;
  std::vector<FrozenSpatialTerminalBinding> logicalNetSinkBindings_;

  friend class FrozenSpatialTransferIndexBuilder;
  friend class FrozenSpatialPortIndexBuilder;
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

/// Dense, cache-only projection of immediate atomic capacity envelopes. Each
/// entry is the exact raw overuse selected by one hot Candidate decision; the
/// Fabric ResourceContract remains the sole owner of capacities, claims, and
/// event semantics.
class FrozenSpatialCapacityIndex final {
public:
  llvm::ArrayRef<std::uint64_t> computeInstructionContextOveruse() const {
    return computeInstructionContextOveruse_;
  }
  llvm::ArrayRef<std::uint64_t> memoryOperationPlanOveruse() const {
    return memoryOperationPlanOveruse_;
  }

private:
  std::vector<std::uint64_t> computeInstructionContextOveruse_;
  std::vector<std::uint64_t> memoryOperationPlanOveruse_;

  friend class FrozenSpatialCapacityIndexBuilder;
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
  PnrIndex routeClaimOffset = 0;
  PnrIndex routeClaimCount = 0;
};

/// One dense, route-selected claim key. The activation group is retained only
/// for cold verification and diagnostics. Candidate hot paths use the ordinal
/// of this record and never compare persistent Fabric references.
struct FrozenSpatialRouteClaim final {
  ::loom::fabric::FabricTraversalActivationGroupView activationGroup;
  PnrIndex capacityDimension = 0;
  std::uint32_t amount = 0;
  std::uint64_t qCost = 0;
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
  llvm::ArrayRef<FrozenSpatialRouteClaim> routeClaims() const {
    return routeClaims_;
  }
  llvm::ArrayRef<PnrIndex> traversalClaimKeys() const {
    return traversalClaimKeys_;
  }
  llvm::ArrayRef<PnrIndex> traversalReplicationGroups() const {
    return traversalReplicationGroups_;
  }
  llvm::ArrayRef<PnrIndex> capacityRouteClaimOffsets() const {
    return capacityRouteClaimOffsets_;
  }
  llvm::ArrayRef<PnrIndex> capacityRouteClaims() const {
    return capacityRouteClaims_;
  }
  llvm::ArrayRef<PnrIndex> routeClaimTraversalOffsets() const {
    return routeClaimTraversalOffsets_;
  }
  llvm::ArrayRef<PnrIndex> routeClaimTraversals() const {
    return routeClaimTraversals_;
  }
  llvm::ArrayRef<PnrIndex> traversalArcOffsets() const {
    return traversalArcOffsets_;
  }
  llvm::ArrayRef<PnrIndex> traversalArcs() const { return traversalArcs_; }
  llvm::ArrayRef<PnrIndex> adjacencyOffsets() const {
    return adjacencyOffsets_;
  }
  llvm::ArrayRef<PnrIndex> reverseAdjacencyOffsets() const {
    return reverseAdjacencyOffsets_;
  }
  llvm::ArrayRef<PnrIndex> reverseArcOrdinals() const {
    return reverseArcOrdinals_;
  }
  llvm::ArrayRef<PnrIndex> arcSources() const { return arcSources_; }
  llvm::ArrayRef<FrozenSpatialRoutingArc> routingArcs() const { return arcs_; }

private:
  std::vector<FrozenSpatialRoutingEndpoint> endpoints_;
  std::vector<FrozenSpatialTraversal> traversals_;
  std::vector<PnrIndex> traversalEndpoints_;
  std::vector<PnrIndex> traversalResourceStates_;
  std::vector<FrozenSpatialRouteClaim> routeClaims_;
  std::vector<PnrIndex> traversalClaimKeys_;
  std::vector<PnrIndex> traversalReplicationGroups_;
  std::vector<PnrIndex> capacityRouteClaimOffsets_;
  std::vector<PnrIndex> capacityRouteClaims_;
  std::vector<PnrIndex> routeClaimTraversalOffsets_;
  std::vector<PnrIndex> routeClaimTraversals_;
  std::vector<PnrIndex> traversalArcOffsets_;
  std::vector<PnrIndex> traversalArcs_;
  std::vector<PnrIndex> adjacencyOffsets_;
  std::vector<PnrIndex> reverseAdjacencyOffsets_;
  std::vector<PnrIndex> reverseArcOrdinals_;
  std::vector<PnrIndex> arcSources_;
  std::vector<FrozenSpatialRoutingArc> arcs_;

  friend class FrozenSpatialPnrProblemBuilder;
};

struct FrozenSpatialHandshakeArc final {
  PnrIndex source = 0;
  PnrIndex destination = 0;
};

struct FrozenSpatialHandshakeFragment final {
  PnrIndex contributionOffset = 0;
  PnrIndex contributionCount = 0;
};

struct FrozenSpatialHandshakeAllTraversalGroup final {
  PnrIndex witnessOffset = 0;
  PnrIndex witnessCount = 0;
  PnrIndex fragment = 0;
};

struct FrozenSpatialMemoryOperationHandshakeDomain final {
  PnrIndex placement = 0;
  PnrIndex actor = 0;
  PnrIndex planOffset = 0;
  PnrIndex planCount = 0;
};

struct FrozenSpatialMemoryOperationHandshakePlan final {
  PnrIndex usePattern = 0;
  PnrIndex fragmentOffset = 0;
  PnrIndex fragmentCount = 0;
};

/// One immutable, cache-oriented flattening of the Fabric-owned handshake
/// owner models. Persistent references are retained only in the node reverse
/// table for cold diagnostics and final projection; every search incidence is
/// expressed with dense PnrIndex values.
class FrozenSpatialHandshakeIndex final {
public:
  PnrIndex nodeCount() const {
    return static_cast<PnrIndex>(nodeSignals_.size());
  }
  llvm::ArrayRef<std::optional<::loom::fabric::HandshakeSignalRef>>
  nodeSignals() const {
    return nodeSignals_;
  }
  llvm::ArrayRef<FrozenSpatialHandshakeArc> arcs() const { return arcs_; }
  llvm::ArrayRef<PnrIndex> adjacencyOffsets() const {
    return adjacencyOffsets_;
  }
  llvm::ArrayRef<PnrIndex> reverseAdjacencyOffsets() const {
    return reverseAdjacencyOffsets_;
  }
  llvm::ArrayRef<PnrIndex> reverseArcOrdinals() const {
    return reverseArcOrdinals_;
  }
  llvm::ArrayRef<FrozenSpatialHandshakeFragment> fragments() const {
    return fragments_;
  }
  llvm::ArrayRef<PnrIndex> fragmentArcOrdinals() const {
    return fragmentArcOrdinals_;
  }
  llvm::ArrayRef<PnrIndex> fixedFragments() const { return fixedFragments_; }
  llvm::ArrayRef<PnrIndex> traversalFragmentOffsets() const {
    return traversalFragmentOffsets_;
  }
  llvm::ArrayRef<PnrIndex> traversalFragments() const {
    return traversalFragments_;
  }
  llvm::ArrayRef<FrozenSpatialHandshakeAllTraversalGroup>
  allTraversalGroups() const {
    return allTraversalGroups_;
  }
  llvm::ArrayRef<PnrIndex> allTraversalGroupWitnesses() const {
    return allTraversalGroupWitnesses_;
  }
  llvm::ArrayRef<PnrIndex> traversalAllGroupOffsets() const {
    return traversalAllGroupOffsets_;
  }
  llvm::ArrayRef<PnrIndex> traversalAllGroups() const {
    return traversalAllGroups_;
  }
  llvm::ArrayRef<PnrIndex> computePlacementFragmentOffsets() const {
    return computePlacementFragmentOffsets_;
  }
  llvm::ArrayRef<PnrIndex> computePlacementFragments() const {
    return computePlacementFragments_;
  }
  llvm::ArrayRef<FrozenSpatialMemoryOperationHandshakeDomain>
  memoryOperationDomains() const {
    return memoryOperationDomains_;
  }
  llvm::ArrayRef<PnrIndex> memoryPlacementDomainOffsets() const {
    return memoryPlacementDomainOffsets_;
  }
  llvm::ArrayRef<FrozenSpatialMemoryOperationHandshakePlan>
  memoryOperationPlans() const {
    return memoryOperationPlans_;
  }
  llvm::ArrayRef<PnrIndex> memoryPlanFragments() const {
    return memoryPlanFragments_;
  }

private:
  std::vector<std::optional<::loom::fabric::HandshakeSignalRef>> nodeSignals_;
  std::vector<FrozenSpatialHandshakeArc> arcs_;
  std::vector<PnrIndex> adjacencyOffsets_;
  std::vector<PnrIndex> reverseAdjacencyOffsets_;
  std::vector<PnrIndex> reverseArcOrdinals_;
  std::vector<FrozenSpatialHandshakeFragment> fragments_;
  std::vector<PnrIndex> fragmentArcOrdinals_;
  std::vector<PnrIndex> fixedFragments_;
  std::vector<PnrIndex> traversalFragmentOffsets_;
  std::vector<PnrIndex> traversalFragments_;
  std::vector<FrozenSpatialHandshakeAllTraversalGroup> allTraversalGroups_;
  std::vector<PnrIndex> allTraversalGroupWitnesses_;
  std::vector<PnrIndex> traversalAllGroupOffsets_;
  std::vector<PnrIndex> traversalAllGroups_;
  std::vector<PnrIndex> computePlacementFragmentOffsets_;
  std::vector<PnrIndex> computePlacementFragments_;
  std::vector<FrozenSpatialMemoryOperationHandshakeDomain>
      memoryOperationDomains_;
  std::vector<PnrIndex> memoryPlacementDomainOffsets_;
  std::vector<FrozenSpatialMemoryOperationHandshakePlan> memoryOperationPlans_;
  std::vector<PnrIndex> memoryPlanFragments_;

  friend class FrozenSpatialHandshakeIndexBuilder;
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
  const FrozenSpatialTransferIndex &transfers() const { return transfers_; }
  const FrozenSpatialPortIndex &ports() const { return ports_; }
  const FrozenSpatialResourceIndex &resources() const { return resources_; }
  const FrozenSpatialCapacityIndex &capacity() const { return capacity_; }
  const FrozenSpatialRoutingGraph &routing() const { return routing_; }
  const FrozenSpatialHandshakeIndex &handshake() const { return handshake_; }
  const FrozenSpatialPnrCacheKey &cacheKey() const { return cacheKey_; }

private:
  FrozenSpatialPnrProblem(
      ArtifactIdentity dataflowIdentity, ArtifactIdentity techMappingIdentity,
      ArtifactIdentity fabricIdentity, ArtifactIdentity constraintSetIdentity,
      ResolvedPnrConfigView config,
      std::vector<DeterministicWorkBudgetEntry> workBudget,
      FrozenConstraintIndex constraints,
      FrozenSpatialRealizationIndex realizations,
      FrozenSpatialTransferIndex transfers, FrozenSpatialPortIndex ports,
      FrozenSpatialResourceIndex resources, FrozenSpatialCapacityIndex capacity,
      FrozenSpatialRoutingGraph routing, FrozenSpatialHandshakeIndex handshake,
      FrozenSpatialPnrCacheKey cacheKey)
      : dataflowIdentity_(std::move(dataflowIdentity)),
        techMappingIdentity_(std::move(techMappingIdentity)),
        fabricIdentity_(std::move(fabricIdentity)),
        constraintSetIdentity_(std::move(constraintSetIdentity)),
        config_(std::move(config)), workBudget_(std::move(workBudget)),
        constraints_(std::move(constraints)),
        realizations_(std::move(realizations)),
        transfers_(std::move(transfers)), ports_(std::move(ports)),
        resources_(std::move(resources)), capacity_(std::move(capacity)),
        routing_(std::move(routing)), handshake_(std::move(handshake)),
        cacheKey_(cacheKey) {}

  ArtifactIdentity dataflowIdentity_;
  ArtifactIdentity techMappingIdentity_;
  ArtifactIdentity fabricIdentity_;
  ArtifactIdentity constraintSetIdentity_;
  ResolvedPnrConfigView config_;
  std::vector<DeterministicWorkBudgetEntry> workBudget_;
  FrozenConstraintIndex constraints_;
  FrozenSpatialRealizationIndex realizations_;
  FrozenSpatialTransferIndex transfers_;
  FrozenSpatialPortIndex ports_;
  FrozenSpatialResourceIndex resources_;
  FrozenSpatialCapacityIndex capacity_;
  FrozenSpatialRoutingGraph routing_;
  FrozenSpatialHandshakeIndex handshake_;
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
