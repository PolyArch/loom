#ifndef LOOM_MAPPING_ARTIFACT_MAPPINGARTIFACT_H
#define LOOM_MAPPING_ARTIFACT_MAPPINGARTIFACT_H

#include "Common/Artifact.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/IR/UsePatternValue.h"
#include "Fabric/Identity/FabricHandshake.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Mapping/Artifact/ConfiguredHardwareProjection.h"
#include "Mapping/IR/MappingOps.h"
#include "Mapping/IR/MappingSchema.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace loom::mapping {

class SpatialMappingConstraintSetView;

/// Canonicalizes one complete in-memory Mapping root for final verification.
/// This syntax layer normalizes schema-owned record order and Mapping-local
/// IDs. Exact upstream import and profile completeness are enforced by the
/// finalizer that publishes a MappingArtifact.
llvm::Expected<CanonicalSemanticBytes>
writeCanonicalMappingAssembly(::mapping::TechOp root);

/// Canonicalizes one complete in-memory SpatialMapping root. Route-node
/// ordinals and schema-owned record order are assigned from typed semantic
/// keys; exact upstream closure remains the Spatial finalizer's responsibility.
llvm::Expected<CanonicalSemanticBytes>
writeCanonicalSpatialMappingAssembly(::mapping::SpatialOp root);

struct TechComputeActorView final {
  ::dataflow::ActorRef actor;
  ::loom::fabric::FabricFuTemplateNodeRef fabricOperation;
  std::vector<std::uint64_t> operandPorts;
  std::vector<std::uint64_t> resultPorts;
};

struct TechComputeBoundaryView final {
  ::dataflow::ActorRef actor;
  ::loom::fabric::FabricPortDirection direction;
  std::uint64_t portOrdinal;
  ::loom::fabric::FabricFuTemplatePortRef fabricPort;
};

struct TechComputeRealizationView final {
  std::uint64_t entityId;
  ::loom::fabric::FabricFuCapabilityTemplateRef capabilityTemplate;
  std::vector<TechComputeActorView> actors;
  std::vector<TechComputeBoundaryView> boundaries;
};

using TechMemoryGraphEndpointRef =
    std::variant<::dataflow::CanonicalGraphProducerEndpointRef,
                 ::dataflow::CanonicalGraphConsumerEndpointRef>;

struct TechMemoryActorView final {
  ::dataflow::ActorRef actor;
  ::loom::fabric::FabricMemoryEngineTemplateOperationPortRef operationPort;
  ::loom::fabric::FabricMemoryEngineTemplateCapabilityAlternativeRef capability;
  std::vector<::loom::fabric::FabricMemoryEngineTemplateEndpointRef>
      operandPorts;
  std::vector<::loom::fabric::FabricMemoryEngineTemplateEndpointRef>
      resultPorts;
};

struct TechMemoryGraphBoundaryView final {
  TechMemoryGraphEndpointRef terminal;
  ::loom::fabric::FabricMemoryEngineTemplateEndpointRef endpoint;
};

struct TechMemoryInternalEdgeView final {
  ::dataflow::CanonicalGraphProducerEndpointRef producer;
  ::dataflow::CanonicalGraphConsumerEndpointRef consumer;
  ::loom::fabric::FabricMemoryEngineTemplateInternalConnectionRef connection;
};

/// Closed legality result for the physical internal-connection domain owned by
/// one Memory Realization. A consumer has one internal source, while one
/// physical connection may serve multiple consumers only when they share the
/// exact producer.
enum class TechMemoryInternalConnectionLegality {
  Admissible,
  ConsumerHasMultipleSources,
  ConnectionHasMultipleProducers,
};

/// Rebuilds the internal-connection legality relation from selected edges.
/// Candidate generation and strict artifact import consume this single
/// projection; the result is not persisted in Mapping.
llvm::Expected<TechMemoryInternalConnectionLegality>
deriveTechMemoryInternalConnectionLegality(
    llvm::ArrayRef<TechMemoryInternalEdgeView> internalEdges,
    const ArtifactIdentity &dataflowOwner);

struct TechMemoryRealizationView final {
  std::uint64_t entityId;
  ::loom::fabric::FabricMemoryEngineTemplateRef engine;
  std::vector<TechMemoryActorView> actors;
  std::vector<TechMemoryGraphBoundaryView> graphBoundaries;
  std::vector<TechMemoryInternalEdgeView> internalEdges;
};

/// One external arrival that must remain distinguishable at a Temporal
/// Memory Operation Engine. Selected engine-internal edges are absent because
/// they address their destination operation-table row directly.
struct TechMemoryExternalIngressView final {
  ::loom::fabric::FabricMemoryEngineTemplateEndpointRef endpoint;
  ::dataflow::CanonicalGraphProducerEndpointRef producer;
};

/// Derives the exact external ingress relation from the selected actors and
/// selected internal edges. Generator admission, strict artifact import, and
/// Spatial occurrence conflicts consume this single interpretation.
llvm::Expected<std::vector<TechMemoryExternalIngressView>>
deriveTechMemoryExternalIngresses(
    const TechMemoryRealizationView &realization,
    const ::dataflow::CanonicalDataflowProgramView &dataflow);
llvm::Expected<bool> techMemoryExternalIngressesAreDistinct(
    const TechMemoryRealizationView &realization,
    const ::dataflow::CanonicalDataflowProgramView &dataflow);
llvm::Expected<std::vector<std::uint8_t>> canonicalTechMemoryExternalIngressKey(
    const TechMemoryExternalIngressView &ingress,
    const ArtifactIdentity &dataflowOwner);

/// One exact residual graph-local transfer obligation derived from D/T after
/// realization-internal sinks have been removed. The producer remains the
/// persistent SpatialLogicalNetKey; this view is a removable import cache and
/// introduces no Mapping-local identity.
struct TechResidualLogicalNetView final {
  ::dataflow::CanonicalGraphProducerEndpointRef producer;
  std::vector<::dataflow::CanonicalGraphConsumerEndpointRef> sinks;
};

/// Rebuilds the activity-definedness admission required by the exact selected
/// compute capability. This is derived only from the canonical actor graph and
/// Fabric resource contract; TechMapping does not persist the result.
llvm::Expected<bool> deriveTechComputeActivityDefinednessAdmission(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    ::dataflow::ActorRef actor,
    const ::loom::fabric::ResolvedFabricOpCapabilityView &capability);

/// Verifies the exact realization-wide topology and correspondence relation
/// after each actor's typed operation capability has been resolved. Generator
/// and strict importer share these owners so candidate pruning cannot diverge
/// from persistent-artifact admission.
llvm::Error verifyTechComputeRealizationClosure(
    const TechComputeRealizationView &realization,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricArtifactView &fabric);
llvm::Error verifyTechMemoryRealizationClosure(
    const TechMemoryRealizationView &realization,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricArtifactView &fabric);

/// Resolves one externally routed Dataflow terminal through the exact
/// Canonical Service role ordering retained by a Tech memory actor row.
llvm::Expected<::loom::fabric::FabricMemoryEngineTemplateEndpointRef>
resolveTechMemoryActorTerminal(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMemoryActorView &actor,
    const ::dataflow::ActorTokenOperandRef &terminal);
llvm::Expected<::loom::fabric::FabricMemoryEngineTemplateEndpointRef>
resolveTechMemoryActorTerminal(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMemoryActorView &actor,
    const ::dataflow::ActorTokenResultRef &terminal);

/// Canonical prospective persistent-payload keys used by Mapping-local
/// identity assignment and TechMapping seed enumeration. These encoders are
/// the sole owner of row ordering; callers must not duplicate their formula.
llvm::Expected<std::vector<std::uint8_t>>
canonicalTechMatchActorKey(const TechComputeActorView &actor,
                           const ArtifactIdentity &dataflowOwner);
llvm::Expected<std::vector<std::uint8_t>>
canonicalTechMatchActorKey(const TechMemoryActorView &actor,
                           const ArtifactIdentity &dataflowOwner);
llvm::Expected<std::vector<std::uint8_t>>
canonicalTechMatchRowKey(const TechComputeRealizationView &realization,
                         const ArtifactIdentity &dataflowOwner);
llvm::Expected<std::vector<std::uint8_t>>
canonicalTechMatchRowKey(const TechMemoryRealizationView &realization,
                         const ArtifactIdentity &dataflowOwner);

/// Immutable read-only projection of one independently verified mapping.tech
/// object. Every member is a typed reference into the exact bound Dataflow or
/// Fabric artifact; copied semantic descriptions and authoring handles are
/// deliberately absent.
class TechMappingView final {
public:
  static llvm::Expected<TechMappingView>
  import(const ArtifactIdentity &mappingIdentity, ::mapping::TechOp root,
         const ::dataflow::CanonicalDataflowProgramView &dataflow,
         const ::loom::fabric::FabricArtifactView &fabric);

  const ArtifactIdentity &identity() const { return identity_; }
  const ArtifactIdentity &dataflowIdentity() const { return dataflowIdentity_; }
  const ArtifactIdentity &fabricIdentity() const { return fabricIdentity_; }
  llvm::ArrayRef<::dataflow::GraphRef> covers() const { return covers_; }
  llvm::ArrayRef<TechComputeRealizationView> computeRealizations() const {
    return computeRealizations_;
  }
  llvm::ArrayRef<TechMemoryRealizationView> memoryRealizations() const {
    return memoryRealizations_;
  }
  llvm::ArrayRef<TechResidualLogicalNetView> residualLogicalNets() const {
    return residualLogicalNets_;
  }
  const TechResidualLogicalNetView *residualLogicalNet(
      const ::dataflow::CanonicalGraphProducerEndpointRef &producer) const;

private:
  TechMappingView(ArtifactIdentity identity, ArtifactIdentity dataflowIdentity,
                  ArtifactIdentity fabricIdentity,
                  std::vector<::dataflow::GraphRef> covers,
                  std::vector<TechComputeRealizationView> computeRealizations,
                  std::vector<TechMemoryRealizationView> memoryRealizations,
                  std::vector<TechResidualLogicalNetView> residualLogicalNets)
      : identity_(std::move(identity)),
        dataflowIdentity_(std::move(dataflowIdentity)),
        fabricIdentity_(std::move(fabricIdentity)), covers_(std::move(covers)),
        computeRealizations_(std::move(computeRealizations)),
        memoryRealizations_(std::move(memoryRealizations)),
        residualLogicalNets_(std::move(residualLogicalNets)) {}

  ArtifactIdentity identity_;
  ArtifactIdentity dataflowIdentity_;
  ArtifactIdentity fabricIdentity_;
  std::vector<::dataflow::GraphRef> covers_;
  std::vector<TechComputeRealizationView> computeRealizations_;
  std::vector<TechMemoryRealizationView> memoryRealizations_;
  std::vector<TechResidualLogicalNetView> residualLogicalNets_;
};

/// The immutable result of failure-atomic publication or strict import of one
/// exact mapping.tech 5.0 object.
class FinalizedTechMapping final {
public:
  const ArtifactRootReference &reference() const { return reference_; }
  const CanonicalSemanticBytes &canonicalBytes() const {
    return canonicalBytes_;
  }
  const TechMappingView &view() const { return view_; }

private:
  FinalizedTechMapping(ArtifactRootReference reference,
                       CanonicalSemanticBytes canonicalBytes,
                       TechMappingView view)
      : reference_(std::move(reference)),
        canonicalBytes_(std::move(canonicalBytes)), view_(std::move(view)) {}

  ArtifactRootReference reference_;
  CanonicalSemanticBytes canonicalBytes_;
  TechMappingView view_;

  friend llvm::Expected<FinalizedTechMapping>
  finalizeTechMapping(::mapping::TechOp source, const ArtifactStore &store);
  friend llvm::Expected<FinalizedTechMapping>
  finalizeTechMapping(::mapping::TechOp source,
                      const ::dataflow::CanonicalDataflowProgramView &dataflow,
                      const ::loom::fabric::FabricArtifactView &fabric,
                      const ArtifactStore &store);
  friend llvm::Expected<FinalizedTechMapping>
  importTechMapping(const ArtifactRootReference &reference,
                    const ArtifactStore &store);
};

llvm::Expected<FinalizedTechMapping>
finalizeTechMapping(::mapping::TechOp source, const ArtifactStore &store);

/// Finalizes against exact upstream views already sealed by their family
/// finalizers or strict importers. The corresponding objects must still be
/// durably present in `store`; this overload only avoids reparsing the same
/// immutable upstream artifacts for every candidate in one invocation.
llvm::Expected<FinalizedTechMapping>
finalizeTechMapping(::mapping::TechOp source,
                    const ::dataflow::CanonicalDataflowProgramView &dataflow,
                    const ::loom::fabric::FabricArtifactView &fabric,
                    const ArtifactStore &store);

llvm::Expected<FinalizedTechMapping>
importTechMapping(const ArtifactRootReference &reference,
                  const ArtifactStore &store);

struct SpatialPhysicalRefinementView final {
  ::loom::fabric::FabricPhysicalRefinementDomainRef domain;
  std::vector<std::uint8_t> canonicalValue;
};

struct SpatialComputeBindingView final {
  std::uint64_t realization = 0;
  ::loom::fabric::FabricFuOccurrenceRef occurrence;
  ::loom::fabric::InstructionContextRef context;
  std::vector<SpatialPhysicalRefinementView> refinements;
};

struct SpatialMemoryWholeIntervalView final {};

struct SpatialMemoryByteRangeView final {
  std::uint64_t offsetBytes = 0;
  std::uint64_t sizeBytes = 0;
};

using SpatialMemoryIntervalView =
    std::variant<SpatialMemoryWholeIntervalView, SpatialMemoryByteRangeView>;

struct SpatialMemoryLocalRegionView final {
  ::loom::fabric::FabricMemoryServiceRegionRef serviceRegion;
  std::uint64_t physicalOffsetBytes = 0;
};

struct SpatialMemoryBoundaryProxyView final {};

using SpatialMemoryBindingTargetView =
    std::variant<SpatialMemoryLocalRegionView, SpatialMemoryBoundaryProxyView>;

using SpatialMemoryOperationPlacementView =
    std::variant<::loom::fabric::FabricMemoryOperationPortRef,
                 ::loom::fabric::FabricMemoryOperationContextRef>;

using SpatialMemoryDispatchTargetView =
    std::variant<::loom::fabric::LocalMemoryServiceRef,
                 ::loom::fabric::ManagerEndpointRef>;

using SpatialMemoryConsistencyTargetView =
    std::variant<::loom::fabric::MemoryConsistencyDomainRef,
                 ::loom::fabric::ManagerEndpointRef>;

struct SpatialAddressedMemoryUseView final {
  ::dataflow::RootedGraphLaunchRef launch;
  std::uint64_t binding = 0;
  SpatialMemoryDispatchTargetView dispatch;
};

struct SpatialFenceMemoryUseView final {
  ::dataflow::RootedGraphLaunchRef launch;
  SpatialMemoryConsistencyTargetView consistency;
};

struct SpatialAddressedMemoryOperationView final {
  ::dataflow::ActorRef actor;
  SpatialMemoryOperationPlacementView placement;
  std::vector<SpatialAddressedMemoryUseView> uses;
};

struct SpatialFenceMemoryOperationView final {
  ::dataflow::ActorRef actor;
  SpatialMemoryOperationPlacementView placement;
  std::vector<SpatialFenceMemoryUseView> uses;
};

using SpatialMemoryOperationView =
    std::variant<SpatialAddressedMemoryOperationView,
                 SpatialFenceMemoryOperationView>;

struct SpatialMemoryEngineBindingView final {
  std::uint64_t realization = 0;
  ::loom::fabric::FabricMemoryOccurrenceRef occurrence;
  std::vector<SpatialMemoryOperationView> operations;
};

struct SpatialExposureEntryView final {
  ::dataflow::MemoryExposureRef exposure;
  ::loom::fabric::SubordinateEndpointRef terminal;
  SpatialMemoryDispatchTargetView dispatch;
};

struct SpatialMemoryBindingView final {
  std::uint64_t entityId = 0;
  ::dataflow::LogicalMemoryRootOrViewRef logicalMemory;
  SpatialMemoryIntervalView interval;
  SpatialMemoryBindingTargetView target;
  std::vector<SpatialExposureEntryView> exposures;
};

struct SpatialActorTransitionEventRef final {
  ::dataflow::ActorRef actor;
  std::uint32_t transition = 0;

  friend bool operator==(const SpatialActorTransitionEventRef &lhs,
                         const SpatialActorTransitionEventRef &rhs) {
    return lhs.actor == rhs.actor && lhs.transition == rhs.transition;
  }
};

struct SpatialRegisterFifoTransferView final {
  ::dataflow::CanonicalGraphProducerEndpointRef logicalNet;
  ::dataflow::CanonicalGraphConsumerEndpointRef sink;
  ::loom::fabric::FabricPeOccurrenceRef pe;
  ::loom::fabric::FabricOrdinal registerFifo = 0;
  ::loom::fabric::FabricPhysicalTraversalRef writeTraversal;
  ::loom::fabric::FabricPhysicalTraversalRef readTraversal;
  llvm::APInt tag = llvm::APInt(1, 0);
};

/// Derives the unique issue transition owned by one canonical memory actor.
/// This is the sole Mapping projection used by strict import, materialization,
/// and PnR event indexing.
llvm::Expected<SpatialActorTransitionEventRef> deriveSpatialMemoryIssueEvent(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    ::dataflow::ActorRef actor);

using SpatialActivityEventRef =
    std::variant<SpatialActorTransitionEventRef,
                 ::dataflow::CanonicalGraphProducerEndpointRef,
                 ::dataflow::CanonicalGraphConsumerEndpointRef>;

/// Canonical owner-local comparison key for the closed Spatial activity-event
/// union. This is a derived key, not another persistent reference family.
llvm::Expected<std::vector<std::uint8_t>>
encodeSpatialActivityEventKey(const ArtifactIdentity &dataflowIdentity,
                              const SpatialActivityEventRef &event);

/// One exact compute-resource use mechanically required by a selected
/// realization. Mapping owns this projection so candidate materialization,
/// strict import, and PnR cannot diverge on event or Fabric UsePattern
/// selection.
struct SpatialComputeUseRequirement final {
  std::uint64_t realization = 0;
  SpatialActivityEventRef trigger;
  ::loom::fabric::FabricUsePatternRef pattern;
  std::vector<SpatialActivityEventRef> release;
};

llvm::Expected<std::vector<SpatialComputeUseRequirement>>
deriveSpatialComputeBindingUseRequirements(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechComputeRealizationView &realization,
    const ::loom::fabric::FabricArtifactView &fabric,
    const SpatialComputeBindingView &binding,
    llvm::ArrayRef<SpatialRegisterFifoTransferView> registerFifoTransfers = {});

llvm::Expected<std::vector<SpatialComputeUseRequirement>>
deriveSpatialComputeUseRequirements(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> bindings,
    llvm::ArrayRef<SpatialRegisterFifoTransferView> registerFifoTransfers = {});

struct SpatialRouteNodeView final {
  std::uint64_t ordinal = 0;
  ::loom::fabric::FabricTransportEndpointRef endpoint;
  std::optional<std::uint64_t> parentOrdinal;
  std::optional<::loom::fabric::FabricPhysicalTraversalRef> incomingTraversal;
  std::vector<SpatialPhysicalRefinementView> refinements;
};

struct SpatialRouteSinkView final {
  ::dataflow::CanonicalGraphConsumerEndpointRef sink;
  std::uint64_t nodeOrdinal = 0;
  std::optional<::loom::fabric::FabricPhysicalTraversalRef> localTraversal;
};

struct SpatialRouteTreeView final {
  ::dataflow::CanonicalGraphProducerEndpointRef logicalNet;
  ::loom::fabric::FabricTransportEndpointRef rootEndpoint;
  std::optional<::loom::fabric::FabricPhysicalTraversalRef> localTraversal;
  std::vector<SpatialRouteNodeView> nodes;
  std::vector<SpatialRouteSinkView> sinks;
};

struct SpatialEventPointView final {
  SpatialActivityEventRef event;
  std::optional<std::vector<std::uint8_t>> guaranteedOffset;
};

struct SpatialRelativeActivationView final {
  SpatialEventPointView trigger;
  std::vector<SpatialEventPointView> release;
};

struct SpatialComputeResourceOwnerRef final {
  std::uint64_t realization = 0;
};

struct SpatialMemoryEngineResourceOwnerRef final {
  std::uint64_t realization = 0;
};

struct SpatialMemoryBindingResourceOwnerRef final {
  std::uint64_t binding = 0;
};

struct SpatialRouteNodeResourceOwnerRef final {
  ::dataflow::CanonicalGraphProducerEndpointRef logicalNet;
  std::uint64_t nodeOrdinal = 0;
};

using SpatialResourceOwnerRef = std::variant<
    SpatialComputeResourceOwnerRef, SpatialMemoryEngineResourceOwnerRef,
    SpatialMemoryBindingResourceOwnerRef, SpatialRouteNodeResourceOwnerRef>;

struct SpatialResourceUseView final {
  SpatialResourceOwnerRef owner;
  ::loom::fabric::FabricUsePatternRef useSite;
  SpatialRelativeActivationView activation;
  std::vector<::fabric::UsePatternValue> parameters;
  std::vector<::fabric::UsePatternValue> sharingAssignments;
};

/// Removable route-continuity index derived while strictly importing the
/// Physical Tag ResourceUse at one maximal tagged segment origin. The tag
/// value remains owned by `resourceUseOrdinal`; this record only joins it to
/// the exact RouteTree nodes that carry the value.
struct SpatialPhysicalTagSegmentView final {
  std::uint64_t routeTreeOrdinal = 0;
  std::uint64_t segmentOrdinal = 0;
  std::vector<std::uint64_t> nodeOrdinals;
  std::uint64_t resourceUseOrdinal = 0;
};

/// Immutable projection of one independently verified mapping.spatial object.
/// Dense PnR indices, search history, selected-edge bitsets, and derived
/// claims are deliberately absent.
class SpatialMappingView final {
public:
  static llvm::Expected<SpatialMappingView> import(
      const ArtifactIdentity &mappingIdentity, ::mapping::SpatialOp root,
      const ::dataflow::CanonicalDataflowProgramView &dataflow,
      const TechMappingView &techMapping,
      const ::loom::fabric::FabricArtifactView &fabric,
      const ::loom::fabric::FabricHandshakeContext *handshakeContext = nullptr);

  const ArtifactIdentity &identity() const { return identity_; }
  const ArtifactIdentity &techMappingIdentity() const {
    return techMappingIdentity_;
  }
  const ArtifactIdentity &dataflowIdentity() const { return dataflowIdentity_; }
  const ArtifactIdentity &fabricIdentity() const { return fabricIdentity_; }
  llvm::ArrayRef<SpatialComputeBindingView> computeBindings() const {
    return computeBindings_;
  }
  llvm::ArrayRef<SpatialMemoryEngineBindingView> memoryEngineBindings() const {
    return memoryEngineBindings_;
  }
  llvm::ArrayRef<SpatialMemoryBindingView> memoryBindings() const {
    return memoryBindings_;
  }
  llvm::ArrayRef<SpatialRouteTreeView> routeTrees() const {
    return routeTrees_;
  }
  llvm::ArrayRef<SpatialRegisterFifoTransferView>
  registerFifoTransfers() const {
    return registerFifoTransfers_;
  }
  llvm::ArrayRef<SpatialResourceUseView> resourceUses() const {
    return resourceUses_;
  }
  llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments() const {
    return physicalTagSegments_;
  }
  const ConfiguredHardwareProjectionView &configuredHardware() const {
    return configuredHardware_;
  }
  const ::loom::fabric::FabricHandshakeSelection &handshakeSelection() const {
    return handshakeSelection_;
  }

private:
  SpatialMappingView(
      ArtifactIdentity identity, ArtifactIdentity techMappingIdentity,
      ArtifactIdentity dataflowIdentity, ArtifactIdentity fabricIdentity,
      std::vector<SpatialComputeBindingView> computeBindings,
      std::vector<SpatialMemoryEngineBindingView> memoryEngineBindings,
      std::vector<SpatialMemoryBindingView> memoryBindings,
      std::vector<SpatialRegisterFifoTransferView> registerFifoTransfers,
      std::vector<SpatialRouteTreeView> routeTrees,
      std::vector<SpatialResourceUseView> resourceUses,
      std::vector<SpatialPhysicalTagSegmentView> physicalTagSegments,
      ConfiguredHardwareProjectionView configuredHardware,
      ::loom::fabric::FabricHandshakeSelection handshakeSelection)
      : identity_(std::move(identity)),
        techMappingIdentity_(std::move(techMappingIdentity)),
        dataflowIdentity_(std::move(dataflowIdentity)),
        fabricIdentity_(std::move(fabricIdentity)),
        computeBindings_(std::move(computeBindings)),
        memoryEngineBindings_(std::move(memoryEngineBindings)),
        memoryBindings_(std::move(memoryBindings)),
        registerFifoTransfers_(std::move(registerFifoTransfers)),
        routeTrees_(std::move(routeTrees)),
        resourceUses_(std::move(resourceUses)),
        physicalTagSegments_(std::move(physicalTagSegments)),
        configuredHardware_(std::move(configuredHardware)),
        handshakeSelection_(std::move(handshakeSelection)) {}

  ArtifactIdentity identity_;
  ArtifactIdentity techMappingIdentity_;
  ArtifactIdentity dataflowIdentity_;
  ArtifactIdentity fabricIdentity_;
  std::vector<SpatialComputeBindingView> computeBindings_;
  std::vector<SpatialMemoryEngineBindingView> memoryEngineBindings_;
  std::vector<SpatialMemoryBindingView> memoryBindings_;
  std::vector<SpatialRegisterFifoTransferView> registerFifoTransfers_;
  std::vector<SpatialRouteTreeView> routeTrees_;
  std::vector<SpatialResourceUseView> resourceUses_;
  std::vector<SpatialPhysicalTagSegmentView> physicalTagSegments_;
  ConfiguredHardwareProjectionView configuredHardware_;
  ::loom::fabric::FabricHandshakeSelection handshakeSelection_;
};

/// Resolves the exact Physical Tag assigned to one RouteTree node. Untagged
/// nodes return the canonical one-bit zero sentinel; callers must inspect the
/// Fabric data path before treating the value as a physical signal.
llvm::Expected<llvm::APInt>
resolveSpatialPhysicalTag(const SpatialMappingView &mapping,
                          const ::loom::fabric::FabricArtifactView &fabric,
                          std::uint64_t routeTreeOrdinal,
                          std::uint64_t nodeOrdinal);

class FinalizedSpatialMapping final {
public:
  const ArtifactRootReference &reference() const { return reference_; }
  const CanonicalSemanticBytes &canonicalBytes() const {
    return canonicalBytes_;
  }
  const SpatialMappingView &view() const { return view_; }

private:
  FinalizedSpatialMapping(ArtifactRootReference reference,
                          CanonicalSemanticBytes canonicalBytes,
                          SpatialMappingView view)
      : reference_(std::move(reference)),
        canonicalBytes_(std::move(canonicalBytes)), view_(std::move(view)) {}

  ArtifactRootReference reference_;
  CanonicalSemanticBytes canonicalBytes_;
  SpatialMappingView view_;

  friend llvm::Expected<FinalizedSpatialMapping> finalizeSpatialMapping(
      ::mapping::SpatialOp source,
      const ::dataflow::CanonicalDataflowProgramView &dataflow,
      const TechMappingView &techMapping,
      const ::loom::fabric::FabricArtifactView &fabric,
      const SpatialMappingConstraintSetView &constraints,
      const ArtifactStore &store,
      const ::loom::fabric::FabricHandshakeContext *handshakeContext);
  friend llvm::Expected<FinalizedSpatialMapping>
  importSpatialMapping(const ArtifactRootReference &reference,
                       const ArtifactStore &store);
};

struct SpatialMappingImportContextStatistics final {
  std::uint64_t constructionNanoseconds = 0;
  std::uint64_t retainedBytes = 0;
  std::uint64_t deterministicWork = 0;
  std::uint64_t mappingCount = 0;
};

/// Bounded immutable import set for one explicit invocation. Every member is
/// strictly imported once; lookups never admit references outside the exact
/// canonical set used to derive the context key.
class SpatialMappingImportContext final {
public:
  SpatialMappingImportContext(const SpatialMappingImportContext &) = delete;
  SpatialMappingImportContext &
  operator=(const SpatialMappingImportContext &) = delete;
  SpatialMappingImportContext(SpatialMappingImportContext &&) noexcept =
      default;
  SpatialMappingImportContext &
  operator=(SpatialMappingImportContext &&) noexcept = default;

  const std::array<std::uint8_t, 32> &key() const { return key_; }
  const SpatialMappingImportContextStatistics &statistics() const {
    return statistics_;
  }
  llvm::ArrayRef<ArtifactRootReference> references() const {
    return references_;
  }
  const FinalizedSpatialMapping *
  find(const ArtifactRootReference &reference) const;

private:
  SpatialMappingImportContext(
      std::array<std::uint8_t, 32> key,
      std::vector<ArtifactRootReference> references,
      std::vector<std::shared_ptr<const FinalizedSpatialMapping>> mappings,
      SpatialMappingImportContextStatistics statistics)
      : key_(key), references_(std::move(references)),
        mappings_(std::move(mappings)), statistics_(statistics) {}

  std::array<std::uint8_t, 32> key_{};
  std::vector<ArtifactRootReference> references_;
  std::vector<std::shared_ptr<const FinalizedSpatialMapping>> mappings_;
  SpatialMappingImportContextStatistics statistics_;

  friend llvm::Expected<SpatialMappingImportContext>
  buildSpatialMappingImportContext(llvm::ArrayRef<ArtifactRootReference>,
                                   const ArtifactStore &);
  friend llvm::Expected<std::shared_ptr<const FinalizedSpatialMapping>>
  resolveSpatialMappingImportHandle(const SpatialMappingImportContext &,
                                    const ArtifactRootReference &);
};

llvm::Expected<SpatialMappingImportContext> buildSpatialMappingImportContext(
    llvm::ArrayRef<ArtifactRootReference> references,
    const ArtifactStore &store);

llvm::Expected<const FinalizedSpatialMapping *>
resolveSpatialMappingImport(const SpatialMappingImportContext &context,
                            const ArtifactRootReference &reference);

llvm::Expected<std::shared_ptr<const FinalizedSpatialMapping>>
resolveSpatialMappingImportHandle(const SpatialMappingImportContext &context,
                                  const ArtifactRootReference &reference);

/// Runs the intrinsic Spatial Mapping base verifier without publishing the
/// draft. Constraint admission is deliberately outside this owner.
llvm::Error verifySpatialMappingBase(
    ::mapping::SpatialOp source,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric);

/// Production Spatial publication gate. The independently finalized K remains
/// outside Mapping identity, but its exact owner tuple and admission must hold
/// before the candidate reaches ArtifactStore::put.
llvm::Expected<FinalizedSpatialMapping> finalizeSpatialMapping(
    ::mapping::SpatialOp source,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const SpatialMappingConstraintSetView &constraints,
    const ArtifactStore &store,
    const ::loom::fabric::FabricHandshakeContext *handshakeContext = nullptr);

llvm::Expected<FinalizedSpatialMapping>
importSpatialMapping(const ArtifactRootReference &reference,
                     const ArtifactStore &store);

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_MAPPINGARTIFACT_H
