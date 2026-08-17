#ifndef LOOM_MAPPING_ARTIFACT_SPATIALPHYSICALDEMANDPROJECTION_H
#define LOOM_MAPPING_ARTIFACT_SPATIALPHYSICALDEMANDPROJECTION_H

#include "Fabric/IR/TemporalOperandBuffer.h"
#include "Fabric/IR/TemporalPeResourceContract.h"
#include "Mapping/Artifact/MappingArtifact.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::mapping {

/// One concrete FU occurrence and its resident instruction-context supply for
/// a Tech compute realization. Contexts are ordered by the Fabric-owned
/// resident-context ordinal.
struct SpatialComputeContextPlacementDomainView final {
  ::loom::fabric::FabricFuOccurrenceRef fu;
  ::loom::fabric::FabricPeOccurrenceRef parentPe;
  ::fabric::Schedule schedule = ::fabric::Schedule::Spatial;
  std::vector<::loom::fabric::InstructionContextRef> contexts;
};

/// The complete unconstrained root-placement supply for one Tech compute
/// realization. Spatial constraints may only filter this domain; they never
/// reconstruct Fabric capability or resident-context ownership.
struct SpatialComputeContextDemandView final {
  std::uint64_t realization = 0;
  ::loom::fabric::FabricFuCapabilityTemplateRef capabilityTemplate;
  std::vector<SpatialComputeContextPlacementDomainView> placements;
  std::uint64_t candidatePlacementCount = 0;
};

/// Rebuilds the complete unconstrained placement domain for one Fabric
/// capability template. Tech cover search and materialized Mapping
/// verification use this function rather than separately interpreting FU and
/// resident-context ownership.
llvm::Expected<std::vector<SpatialComputeContextPlacementDomainView>>
deriveSpatialComputeContextPlacementDomain(
    ::loom::fabric::FabricFuCapabilityTemplateRef capabilityTemplate,
    const ::loom::fabric::FabricArtifactView &fabric);

/// Rebuilds the exact FU, parent-PE, schedule, and resident-context domains for
/// every Tech compute realization. Tech frontier admission and Spatial PnR
/// consume this projection as the single owner of unconstrained root supply.
llvm::Expected<std::vector<SpatialComputeContextDemandView>>
deriveSpatialComputeContextDemands(
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric);

/// Exact all-different closure over an ordinalized compute-context relation.
/// The caller owns ordinalization; this analysis owns matching work and the
/// canonical alternating-tree Hall witness.
struct SpatialComputeContextSupplyAnalysis final {
  std::uint64_t demandCount = 0;
  std::uint64_t valueCount = 0;
  std::uint64_t edgeCount = 0;
  std::uint64_t maximumMatching = 0;
  std::uint64_t hallValueCount = 0;
  std::uint64_t deterministicWork = 0;
  std::vector<std::uint64_t> hallDemands;

  bool admissible() const { return maximumMatching == demandCount; }
};

llvm::Expected<SpatialComputeContextSupplyAnalysis>
analyzeSpatialComputeContextSupply(
    llvm::ArrayRef<std::vector<std::size_t>> domains,
    std::size_t valueCount);

/// One Fabric occurrence that can host a Tech memory realization. Temporal
/// capacity is occurrence-global; Spatial occurrences carry zero here because
/// their operation-port exclusivity is represented separately.
struct SpatialMemoryOccurrenceSupplyView final {
  ::loom::fabric::FabricMemoryOccurrenceRef occurrence;
  std::uint64_t residentCapacity = 0;
};

enum class SpatialMemoryExclusiveResourceKind : std::uint8_t {
  SpatialOperationPort,
  TemporalExternalIngress,
  InternalConnection,
};

llvm::StringRef spatialMemoryExclusiveResourceKindSpelling(
    SpatialMemoryExclusiveResourceKind kind);

/// One occurrence-local resource that cannot be owned by two selected memory
/// realizations. The typed kind prevents unrelated canonical reference domains
/// from becoming accidental aliases.
struct SpatialMemoryExclusiveResourceView final {
  SpatialMemoryExclusiveResourceKind kind =
      SpatialMemoryExclusiveResourceKind::SpatialOperationPort;
  std::vector<std::uint8_t> key;
};

/// Complete unconstrained root-placement demand for one Tech memory
/// realization. This projection is shared by Tech cover search and
/// materialized root-supply revalidation; Spatial constraints may only narrow
/// its occurrence domain.
struct SpatialMemoryOccurrenceDemandView final {
  std::uint64_t realization = 0;
  ::loom::fabric::FabricMemoryEngineTemplateRef engine;
  ::fabric::Schedule schedule = ::fabric::Schedule::Spatial;
  std::vector<SpatialMemoryOccurrenceSupplyView> occurrences;
  std::vector<SpatialMemoryExclusiveResourceView> exclusiveResources;
  std::uint64_t residentDemand = 0;
  std::uint64_t projectionWork = 0;
};

llvm::Expected<SpatialMemoryOccurrenceDemandView>
deriveSpatialMemoryOccurrenceDemand(
    const TechMemoryRealizationView &realization,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricArtifactView &fabric);

llvm::Expected<std::vector<SpatialMemoryOccurrenceDemandView>>
deriveSpatialMemoryOccurrenceDemands(
    const TechMappingView &techMapping,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricArtifactView &fabric);

enum class SpatialMemoryOccurrenceSupplyFailureKind : std::uint8_t {
  None,
  EmptyOccurrenceDomain,
  ExclusiveResourceDeficit,
  ResidentCapacityDeficit,
  JointAssignmentInfeasible,
};

llvm::StringRef spatialMemoryOccurrenceSupplyFailureKindSpelling(
    SpatialMemoryOccurrenceSupplyFailureKind failure);

/// Exact occurrence assignment closure for Memory root supply. The fast
/// relation and capacity checks produce stable witnesses; a deterministic
/// joint search closes interactions between those relations.
struct SpatialMemoryOccurrenceSupplyAnalysis final {
  SpatialMemoryOccurrenceSupplyFailureKind failure =
      SpatialMemoryOccurrenceSupplyFailureKind::None;
  std::uint64_t demandCount = 0;
  std::uint64_t occurrenceValueCount = 0;
  std::uint64_t occurrenceChoiceCount = 0;
  std::uint64_t exclusiveRelationCount = 0;
  std::uint64_t assignmentAttempts = 0;
  std::uint64_t deterministicWork = 0;
  std::optional<SpatialMemoryExclusiveResourceKind> failingResourceKind;
  std::uint64_t failingDemandCount = 0;
  std::uint64_t failingOccurrenceCount = 0;
  std::uint64_t failingResidentDemand = 0;
  std::uint64_t failingResidentCapacity = 0;

  bool admissible() const {
    return failure == SpatialMemoryOccurrenceSupplyFailureKind::None;
  }
};

llvm::Expected<SpatialMemoryOccurrenceSupplyAnalysis>
analyzeSpatialMemoryOccurrenceSupply(
    llvm::ArrayRef<const SpatialMemoryOccurrenceDemandView *> demands);

/// Fabric-typed durable storage reached by one exact selected sink attachment.
/// This is a rebuildable physical-demand fact, not a persistent Mapping
/// reference or an endpoint property.
enum class SpatialDurableProgressBoundaryKind : std::uint8_t {
  None,
  BufferedFifo,
  TemporalPeOperandQueue,
};

struct SpatialDurableProgressBoundaryView final {
  SpatialDurableProgressBoundaryKind kind =
      SpatialDurableProgressBoundaryKind::None;
  ::loom::fabric::FabricPhysicalTraversalRef attachment;
  std::optional<::fabric::LogicalOperandQueueKey> operandQueue;
};

/// One physical Temporal PE operand queue selected by an ingress activation.
/// Every consumer is an FU-local broadcast obligation of the same queue head.
struct SpatialPeOperandQueueMatchView final {
  std::vector<::dataflow::CanonicalGraphConsumerEndpointRef> consumers;
  ::fabric::LogicalOperandQueueKey queue;
  std::uint32_t allocationUnit = 0;
  std::uint32_t entryCapacity = 0;
};

/// One exact Temporal PE ingress activation. Every distinct queue in the group
/// matches the same physical input and Physical Tag and must append on one
/// common fire. A queue appears once even when its FU boundary broadcasts to
/// several logical consumers.
struct SpatialPeOperandQueueMatchGroupView final {
  ::dataflow::CanonicalGraphProducerEndpointRef logicalNet;
  ::loom::fabric::FabricTransportEndpointRef ingress;
  llvm::APInt tag = llvm::APInt(1, 0);
  std::vector<SpatialPeOperandQueueMatchView> matches;
};

/// One active Mapping realization in a Fabric-owned temporal context-
/// evaluation service. Several actors inside the realization share this
/// scheduling opportunity without becoming one macro transition.
struct SpatialTemporalPeDispatchCandidateView final {
  std::uint64_t realization = 0;
  ::loom::fabric::InstructionContextRef context;
  ::loom::fabric::FabricFuOccurrenceRef fu;
  ::loom::fabric::FabricUsePatternRef usePattern;
  ::fabric::RequesterKey requester = ::fabric::RequesterKey(0);
};

/// One independently advancing dispatch cursor. Candidate order is the exact
/// filtered Fabric GrantPolicy cycle, not Mapping insertion order.
struct SpatialTemporalPeDispatchDomainView final {
  ::loom::fabric::FabricPeOccurrenceRef pe;
  std::uint32_t allocationUnit = 0;
  std::uint32_t resetPosition = 0;
  std::vector<SpatialTemporalPeDispatchCandidateView> candidates;
};

/// Rebuilds all active temporal context-dispatch domains from exact compute
/// bindings and the PE ResourceContract. The result is consumed by progress,
/// simulator planning, and RTL/configuration consistency checks.
llvm::Expected<std::vector<SpatialTemporalPeDispatchDomainView>>
deriveSpatialTemporalPeDispatchDomains(
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> computeBindings);

/// One exact eligible disposition of a single-consumer residual edge through
/// a Temporal PE register FIFO. The tag is derived canonically for the
/// initial exclusive-FIFO policy; it is not a second persistent choice.
struct SpatialPeLocalTransferOptionView final {
  ::dataflow::CanonicalGraphProducerEndpointRef producer;
  ::dataflow::CanonicalGraphConsumerEndpointRef sink;
  ::loom::fabric::FabricPeOccurrenceRef pe;
  ::loom::fabric::FabricOrdinal registerFifo = 0;
  ::loom::fabric::FabricPhysicalTraversalRef writeTraversal;
  ::loom::fabric::FabricPhysicalTraversalRef readTraversal;
  llvm::APInt tag = llvm::APInt(1, 0);
};

/// One exact route signature presented to a Temporal switch resident row.
/// The logical segment is identified only to prove canonical reconstruction;
/// the selected crosspoints remain owned by Fabric traversals.
struct SpatialTemporalSwitchRouteSignatureView final {
  ::loom::fabric::FabricSwitchOccurrenceRef occurrence;
  std::uint64_t routeTreeOrdinal = 0;
  std::uint64_t segmentOrdinal = 0;
  ::loom::fabric::FabricOrdinal input = 0;
  std::vector<::loom::fabric::FabricOrdinal> outputs;
  std::vector<::loom::fabric::FabricPhysicalTraversalRef> traversals;
  llvm::APInt tag = llvm::APInt(1, 0);
};

/// One exact resident Temporal switch row. Equal-tag compatible signatures
/// share this row; incompatible signatures with one tag are rejected rather
/// than widened into an unintended crosspoint or broadcast.
struct SpatialTemporalSwitchPackedRowView final {
  ::loom::fabric::FabricSwitchOccurrenceRef occurrence;
  llvm::APInt tag = llvm::APInt(1, 0);
  std::vector<SpatialTemporalSwitchRouteSignatureView> signatures;
  std::vector<::loom::fabric::FabricPhysicalTraversalRef> traversals;
};

/// Rebuilds Temporal switch signatures and resident rows from exact selected
/// RouteTrees and their Physical Tag segments. This projection is shared by
/// strict verification, configuration, handshake, and execution planning.
llvm::Expected<std::vector<SpatialTemporalSwitchPackedRowView>>
deriveSpatialTemporalSwitchPackedRows(
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments);

/// Rebuilds the complete canonical RegFIFO alternative domain for one Tech
/// residual net under exact selected compute placements. Multicast, graph
/// boundary, cross-PE, non-Temporal, and data-path-incompatible edges have an
/// empty domain and therefore retain their external-route disposition.
llvm::Expected<std::vector<SpatialPeLocalTransferOptionView>>
deriveSpatialPeLocalTransferOptions(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> computeBindings,
    const TechResidualLogicalNetView &logicalNet);

/// Equivalent projection when the caller already owns the exact producer and
/// consumer realizations. Frozen PnR indexes use this form to avoid repeatedly
/// resolving actors through the complete TechMapping realization catalog.
llvm::Expected<std::vector<SpatialPeLocalTransferOptionView>>
deriveSpatialPeLocalTransferOptionsForRealizations(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechComputeRealizationView &producerRealization,
    const TechComputeRealizationView &consumerRealization,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> computeBindings,
    const TechResidualLogicalNetView &logicalNet);

/// Exact role-level physical demand for one actor in one selected memory
/// occurrence. Vector positions are ServiceValueRole ordinals. Empty entries
/// denote roles inactive in the actor schema, never an inferred default.
struct SpatialMemoryActorRoleDemandView final {
  ::dataflow::ActorRef actor;
  ::loom::fabric::FabricMemoryOccurrenceRef occurrence;
  std::vector<std::optional<::loom::fabric::FabricMemoryHandshakeRoleSource>>
      sources;
  std::vector<
      std::optional<::loom::fabric::FabricMemoryHandshakeRoleDestination>>
      destinations;
};

/// Resolves one template-relative selected internal connection into the exact
/// occurrence-local connection ordinal. Memory activation, configuration,
/// strict verification, and execution planning share this projection.
llvm::Expected<::loom::fabric::FabricOrdinal>
deriveSpatialMemoryInternalConnectionOrdinal(
    const ::loom::fabric::FabricArtifactView &fabric,
    ::loom::fabric::FabricMemoryOccurrenceRef occurrence,
    const ::loom::fabric::FabricMemoryEngineTemplateInternalConnectionRef
        &selected);

/// Rebuilds role sources and destinations from the Tech realization's exact
/// internal-edge relation and residual logical nets. Configuration,
/// handshake selection, PnR, and execution consume this one projection.
llvm::Expected<std::vector<SpatialMemoryActorRoleDemandView>>
deriveSpatialMemoryActorRoleDemands(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const TechMemoryRealizationView &realization,
    ::loom::fabric::FabricMemoryOccurrenceRef occurrence);

/// Classifies one attachment against its exact concrete FU port. Omitting the
/// FU port admits only traversal-owned boundaries such as a Buffered FIFO.
llvm::Expected<SpatialDurableProgressBoundaryKind>
classifySpatialAttachmentDurableProgressBoundary(
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::fabric::FabricPhysicalTraversalRef &attachment,
    std::optional<::loom::fabric::FabricFuOccurrencePortRef> fuPort);

/// Rebuilds the exact durable boundary selected by a persistent route sink.
/// A Temporal PE operand queue includes its Fabric-owned logical queue key.
llvm::Expected<std::optional<SpatialDurableProgressBoundaryView>>
deriveSpatialSinkDurableProgressBoundary(
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> computeBindings,
    const SpatialRouteSinkView &sink);

llvm::Expected<std::vector<SpatialPeOperandQueueMatchGroupView>>
deriveSpatialPeOperandQueueMatchGroups(
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> computeBindings,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments);

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_SPATIALPHYSICALDEMANDPROJECTION_H
