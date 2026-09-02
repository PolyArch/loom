#ifndef LOOM_MAPPING_ARTIFACT_SPATIALPHYSICALDEMANDPROJECTION_H
#define LOOM_MAPPING_ARTIFACT_SPATIALPHYSICALDEMANDPROJECTION_H

#include "Fabric/IR/TemporalOperandBuffer.h"
#include "Fabric/IR/TemporalPeResourceContract.h"
#include "Common/ComponentViewDigest.h"
#include "Mapping/Artifact/MappingArtifact.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
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
    llvm::ArrayRef<std::vector<std::size_t>> domains, std::size_t valueCount);

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

/// One external actor input participating in a Dataflow-owned firing case.
/// `producer` distinguishes independent ordered token sequences from ordinary
/// SSA fanout of one common sequence.
struct TechComputeOrderedInputMemberView final {
  ::dataflow::ActorTokenOperandRef consumer;
  ::dataflow::CanonicalGraphProducerEndpointRef producer;
};

/// One canonical set of independently produced external inputs that a
/// registered actor firing may consume together. Actor handshake semantics own
/// the firing cases; TechMapping contributes only its selected boundary
/// correspondence. Duplicate cases with the same member set are collapsed.
struct TechComputeOrderedInputGroupView final {
  std::uint64_t realization = 0;
  ::dataflow::ActorRef actor;
  ::dataflow::GraphRef graph;
  std::vector<TechComputeOrderedInputMemberView> members;
};

/// Derives the semantic ordered-input groups shared by early Spatial search
/// and strict Mapping projection. Inputs with one common logical producer are
/// represented once because their physical disposition is atomic fanout, not
/// two independently arriving ordered streams.
llvm::Expected<std::vector<TechComputeOrderedInputGroupView>>
deriveTechComputeOrderedInputGroups(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping);

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
  ::loom::fabric::FabricFuOccurrenceRef fu;
  std::uint32_t allocationUnit = 0;
  std::uint32_t entryCapacity = 0;
};

struct SpatialPeOperandQualifiedPairingKey final {
  ::loom::fabric::InstructionContextRef context;
  ::loom::fabric::FabricFuOccurrenceRef fu;
  llvm::APInt tag = llvm::APInt(1, 0);

  friend bool operator==(const SpatialPeOperandQualifiedPairingKey &lhs,
                         const SpatialPeOperandQualifiedPairingKey &rhs) {
    return lhs.context == rhs.context && lhs.fu == rhs.fu &&
           lhs.tag == rhs.tag;
  }
};

enum class SpatialPeOperandProgressStatus : std::uint8_t {
  Safe,
  LikelyRisk,
  ProofNotEstablished,
  ProvenClosedWait,
};

enum class SpatialPeOperandProgressSupport : std::uint8_t {
  Exact,
  Analytic,
  Unsupported,
};

struct SpatialPeOperandPairingProjection final {
  SpatialPeOperandQualifiedPairingKey key;
  std::vector<std::uint32_t> requiredInputRoles;
  std::vector<::loom::fabric::FabricTransportEndpointRef> ingresses;
  std::vector<std::uint32_t> allocationUnits;
};

struct SpatialPeOperandGraphIngressPressureView final {
  ::dataflow::GraphRef graph;
  std::uint64_t pressure = 0;
};

struct SpatialPeOperandProgressFeedback final {
  SpatialPeOperandProgressStatus status =
      SpatialPeOperandProgressStatus::ProofNotEstablished;
  SpatialPeOperandProgressSupport support =
      SpatialPeOperandProgressSupport::Unsupported;
  std::uint64_t groupCount = 0;
  std::uint64_t potentiallyBlockingGroupCount = 0;
  std::uint64_t distinctIngressCount = 0;
  std::uint64_t sharedIngressCount = 0;
  std::uint64_t sharedIngressPressure = 0;
  std::uint64_t pairingOpportunityCount = 0;
  std::uint64_t pairingKeyCount = 0;
  std::uint64_t distinctPairingKeyCount = 0;
  std::vector<SpatialPeOperandQualifiedPairingKey> pairingKeys;
  std::vector<SpatialPeOperandPairingProjection> pairings;
  std::vector<SpatialPeOperandGraphIngressPressureView> graphPressures;
  std::optional<::loom::ComponentViewDigest> projectionDigest;
};

/// Runtime-owned ordered head observation projected into the Mapping
/// progress domain. It is transient evidence only; queue contents remain
/// simulator state and are never serialized as a Mapping identity.
struct SpatialPeOperandRuntimeHeadView final {
  ::fabric::LogicalOperandQueueKey queue;
  ::loom::fabric::FabricFuOccurrenceRef fu;
  llvm::APInt tag = llvm::APInt(1, 0);
  std::uint32_t allocationUnit = 0;
  std::uint32_t capacity = 0;
  std::uint32_t occupancy = 0;
  std::uint32_t reservations = 0;
  std::uint64_t headBindingOrdinal = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t headOccurrenceOrdinal =
      std::numeric_limits<std::uint64_t>::max();
  std::uint64_t headProducerSequenceOrdinal =
      std::numeric_limits<std::uint64_t>::max();
  bool exactHead = false;
};

enum class SpatialPeOperandRuntimeWitnessStatus : std::uint8_t {
  Exact,
  ProofNotEstablished,
  ProvenClosedWait,
  Unsupported,
};

struct SpatialPeOperandRuntimeWitness final {
  SpatialPeOperandRuntimeWitnessStatus status =
      SpatialPeOperandRuntimeWitnessStatus::ProofNotEstablished;
  SpatialPeOperandProgressSupport support =
      SpatialPeOperandProgressSupport::Unsupported;
  std::uint64_t observedHeadCount = 0;
  std::uint64_t exactHeadCount = 0;
  std::uint64_t matchedPairingKeyCount = 0;
  std::uint64_t unmatchedPairingKeyCount = 0;
  std::uint64_t mismatchedHeadCount = 0;
  std::uint64_t fullQueueCount = 0;
  std::optional<::loom::ComponentViewDigest> projectionDigest;
};

/// Joins simulator queue heads with the Mapping-owned qualified pairing
/// projection. The result is an exact ordered correspondence only when every
/// required role has an exact head and all heads for one PairingKey agree on
/// the producer sequence. It does not infer a closed wait cycle from this
/// join; that proof remains owned by the simulator wait-for projection.
llvm::Expected<SpatialPeOperandRuntimeWitness>
deriveSpatialPeOperandRuntimeWitness(
    const SpatialPeOperandProgressFeedback &projection,
    llvm::ArrayRef<SpatialPeOperandRuntimeHeadView> heads);

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
  ::loom::fabric::FabricFuOccurrencePortRef writer;
  ::loom::fabric::FabricFuOccurrencePortRef reader;
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
/// The terminal attachment may be sink-local or mechanically recovered from
/// the terminal endpoint and the concrete FU port when several logical
/// consumers share one routed PE ingress. A Temporal PE operand queue includes
/// its Fabric-owned logical queue key.
llvm::Expected<std::optional<SpatialDurableProgressBoundaryView>>
deriveSpatialSinkDurableProgressBoundary(
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> computeBindings,
    const SpatialRouteTreeView &route, const SpatialRouteSinkView &sink);

llvm::Expected<std::vector<SpatialPeOperandQueueMatchGroupView>>
deriveSpatialPeOperandQueueMatchGroups(
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> computeBindings,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments);

/// Derives one transient queue/pairing risk result from the same selected
/// match groups consumed by Mapping, simulator, and hardware projections. It
/// never selects an ingress or allocates a queue; likely risk is ranking-only.
llvm::Expected<SpatialPeOperandProgressFeedback>
deriveSpatialPeOperandProgressFeedback(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    llvm::ArrayRef<SpatialPeOperandQueueMatchGroupView> groups);

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_SPATIALPHYSICALDEMANDPROJECTION_H
