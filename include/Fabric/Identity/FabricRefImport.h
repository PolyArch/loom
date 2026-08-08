#ifndef LOOM_FABRIC_IDENTITY_FABRICREFIMPORT_H
#define LOOM_FABRIC_IDENTITY_FABRICREFIMPORT_H

#include "Common/Artifact.h"
#include "Fabric/Artifact/FabricSystemContracts.h"
#include "Fabric/IR/BoundaryDataPath.h"
#include "Fabric/IR/ImplementationFamily.h"
#include "Fabric/IR/MemoryConnectivityContract.h"
#include "Fabric/IR/MemoryOperationPort.h"
#include "Fabric/IR/ResourceContract.h"
#include "Fabric/Identity/FabricFuCapabilityTemplate.h"
#include "Fabric/Identity/FabricPeConfiguration.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace fabric {
class MemoryServiceContractRecord;
} // namespace fabric

namespace loom {
namespace fabric {

using MemoryOperationPortView = ::fabric::MemoryOperationPortRecord;
using MemoryCapabilityAlternativeView =
    ::fabric::MemoryCapabilityAlternativeRecord;

/// The exact Fabric-owned definition shared by equal concrete Memory
/// Operation Engines. Dispatch, services, topology, and dynamic state remain
/// occurrence-owned and are deliberately absent.
struct FabricMemoryEngineTemplateRecord {
  ::fabric::Schedule schedule;
  std::optional<std::uint64_t> residentContextCount;
  std::vector<::fabric::MemoryTransportEndpointDescriptor> tokenEndpoints;
  std::vector<::fabric::MemoryOperationPortRecord> operationPorts;
  std::vector<::fabric::MemoryInternalConnectionDeclaration>
      internalConnections;
};

/// One exact physical port of a concrete operation resource. The canonical
/// transport type is the Fabric-owned encoding used by every consumer; it is
/// not reconstructed from a software type or an implementation family.
struct ResolvedFabricOpPhysicalPortView {
  FabricFuNodePortRef reference;
  std::vector<std::uint8_t> canonicalType;
  std::uint32_t payloadWidthBits = 0;
};

/// The immutable cold projection of one concrete `fabric.op`. Every field is
/// derived from the exact finalized Fabric and the generated operation/HSG
/// registry. It is not persistent state and never becomes another capability
/// authority.
struct ResolvedFabricOpCapabilityView {
  FabricFuTemplateNodeRef occurrence;
  ::fabric::ImplementationFamilyId implementationFamily;
  std::vector<::dataflow::OperationSchemaId> enabledOperationSchemas;
  ::fabric::FamilyCapabilityParams parameterizedCapability;
  std::vector<ResolvedFabricOpPhysicalPortView> physicalPorts;
  std::vector<FabricSemanticConfigFieldRef> configurationFieldSchema;
  ::fabric::ResourceContract resourceStateAndTimingContract;
  std::vector<FabricPhysicalRefinementDomainRef> physicalRefinementDomains;

  /// Checks only the concrete operation-resource capability. Port
  /// correspondence, FU topology, placement, and routing remain Mapping
  /// obligations and are deliberately outside this query. `indexBitWidth` is
  /// the canonical DataLayout resolution owned by the actor's exact program;
  /// it is required even when this actor contains no index type so callers
  /// cannot silently fall back to process configuration.
  llvm::Error admit(const ::dataflow::CanonicalActorSchemaProjection &actor,
                    unsigned indexBitWidth,
                    const ::loom::PointerLayout *pointerLayout = nullptr) const;

  /// Checks one exact ordered software-to-physical port correspondence in
  /// addition to the concrete operation-resource capability. The selected
  /// ordinals are direction-local physical-port ordinals, ordered by the
  /// actor's operand or result ordinal. Mapping persists this relation but
  /// does not reinterpret it.
  llvm::Error admitCorrespondence(
      const ::dataflow::CanonicalActorSchemaProjection &actor,
      unsigned indexBitWidth, llvm::ArrayRef<std::uint64_t> operandPorts,
      llvm::ArrayRef<std::uint64_t> resultPorts,
      const ::loom::PointerLayout *pointerLayout = nullptr) const;

  /// Encodes one admitted actor and its exact TechMapping-owned ordered port
  /// correspondence through this resource's semantic field domain. Facts that
  /// do not change configured hardware behavior are projected out.
  /// ConfigurationABI remains the sole owner of the physical code.
  llvm::Expected<CanonicalSemanticBytes> encodeSemanticConfiguration(
      const FabricSemanticConfigFieldRef &field,
      const ::dataflow::CanonicalActorSchemaProjection &actor,
      unsigned indexBitWidth, llvm::ArrayRef<std::uint64_t> operandPorts,
      llvm::ArrayRef<std::uint64_t> resultPorts,
      const ::loom::PointerLayout *pointerLayout = nullptr) const;

  /// Rebuilds the exact sealed relation that owns this capability's semantic
  /// field. ConfigurationABI 2.0 consumes this relation directly; it does not
  /// reinterpret family parameters or maintain a second behavior domain.
  llvm::Expected<::fabric::FabricOpSemanticFieldRelation>
  resolveSemanticFieldRelation(::mlir::MLIRContext &context) const;
};

/// One boundary's exact role in Physical Tag continuity. The point is a
/// read-only projection of the validated boundary port shape; it carries no
/// configured tag value, lookup-table row, or persistent identity.
enum class FabricBoundaryTagContinuityKind : std::uint8_t {
  TokenWriter,
  ConfigurableWriter,
  Rewriter,
  Remover,
};

struct FabricBoundaryTagContinuityPointView final {
  FabricBoundaryTagContinuityKind kind;
  std::uint32_t inputTagWidthBits = 0;
  std::uint32_t outputTagWidthBits = 0;
};

/// One Fabric-owned local interpretation domain for Physical Tags. Temporal
/// PE and memory ingress match independently per physical input endpoint;
/// temporal switch resident tables and boundary rewrite LUTs match in one
/// owner-wide domain. This is a sealed, rebuildable projection with no
/// persistent identity, configured value, or Mapping-owned namespace.
enum class FabricPhysicalTagMatchDomainKind : std::uint8_t {
  TemporalPeIngress,
  TemporalMemoryIngress,
  TemporalSwitchTable,
  BoundaryLookup,
};

struct FabricPhysicalTagMatchDomainView final {
  FabricPhysicalTagMatchDomainKind kind =
      FabricPhysicalTagMatchDomainKind::TemporalPeIngress;
  FabricInventoryOwnerRef owner;
  std::optional<FabricTransportEndpointRef> ingress;
  std::uint32_t tagWidthBits = 0;

  friend bool operator==(const FabricPhysicalTagMatchDomainView &lhs,
                         const FabricPhysicalTagMatchDomainView &rhs) {
    return lhs.kind == rhs.kind && lhs.owner == rhs.owner &&
           lhs.ingress == rhs.ingress && lhs.tagWidthBits == rhs.tagWidthBits;
  }
};

/// One Fabric-owned position at which Mapping may persist a Physical Tag.
/// Ingress positions cover a tagged root attachment; writer positions cover a
/// PE, memory, or boundary that creates or rewrites a tag. The referenced
/// UsePattern owns the exact sharing-value codec.
enum class FabricPhysicalTagAssignmentPointKind : std::uint8_t {
  Ingress,
  Writer,
};

struct FabricPhysicalTagAssignmentPointView final {
  FabricPhysicalTagAssignmentPointKind kind =
      FabricPhysicalTagAssignmentPointKind::Ingress;
  FabricTransportEndpointRef endpoint;
  FabricUsePatternRef pattern;
  std::uint32_t tagWidthBits = 0;

  friend bool operator==(const FabricPhysicalTagAssignmentPointView &lhs,
                         const FabricPhysicalTagAssignmentPointView &rhs) {
    return lhs.kind == rhs.kind && lhs.endpoint == rhs.endpoint &&
           lhs.pattern == rhs.pattern && lhs.tagWidthBits == rhs.tagWidthBits;
  }
};

/// One exact memory-capability path from a Module signature face to an
/// internal memory endpoint. This relation is rebuilt from canonical SSA and
/// carries no persistent identity or independent connectivity authority.
struct FabricModuleBoundaryMemoryAttachmentView final {
  FabricModuleBoundaryEndpointRef boundary;
  FabricMemoryEndpointRef endpoint;

  friend bool operator==(const FabricModuleBoundaryMemoryAttachmentView &lhs,
                         const FabricModuleBoundaryMemoryAttachmentView &rhs) {
    return lhs.boundary == rhs.boundary && lhs.endpoint == rhs.endpoint;
  }
};

/// The owner-defined domain that makes statically implied traversal uses one
/// atomic activation. Most traversals select one exact UsePattern. Temporal
/// switch broadcast is the sole current exception: every selected egress from
/// one ingress requester belongs to the same atomic activation even though
/// each physical traversal has its own pattern. This key is a sealed,
/// rebuildable view value and has no persistent identity or wire encoding.
enum class FabricTraversalActivationGroupKind : std::uint32_t {
  UsePattern,
  SwitchRequester,
};

struct FabricTraversalActivationGroupView final {
  FabricTraversalActivationGroupKind kind =
      FabricTraversalActivationGroupKind::UsePattern;
  FabricInventoryOwnerRef owner;
  FabricOrdinal ordinal = 0;

  friend bool operator==(const FabricTraversalActivationGroupView &lhs,
                         const FabricTraversalActivationGroupView &rhs) {
    return lhs.kind == rhs.kind && lhs.owner == rhs.owner &&
           lhs.ordinal == rhs.ordinal;
  }
  friend bool operator!=(const FabricTraversalActivationGroupView &lhs,
                         const FabricTraversalActivationGroupView &rhs) {
    return !(lhs == rhs);
  }
};

struct FabricTraversalUseView final {
  FabricUsePatternRef pattern;
  FabricTraversalActivationGroupView activationGroup;
};

/// The exact endpoint relation of one admitted physical traversal. The
/// traversal reference remains the persistent identity; endpoint ranges are a
/// sealed, rebuildable projection of the same Fabric owner. `impliedUses`
/// contains only atomic uses selected mechanically by the traversal itself.
/// Event-relative alternatives, including buffered FIFO enqueue, dequeue, and
/// simultaneous use, remain selected by Mapping ResourceUse and are absent.
struct FabricPhysicalTraversalView final {
  FabricPhysicalTraversalRef reference;
  std::vector<FabricTransportEndpointRef> sources;
  std::vector<FabricTransportEndpointRef> destinations;
  std::vector<FabricResourceStateRef> resourceStates;
  std::vector<FabricTraversalUseView> impliedUses;
};

/// One connected token-plane Module signature endpoint and the exact
/// occurrence-local endpoint reached by canonical Module SSA. The boundary
/// reference remains attachment correspondence rather than a routable
/// endpoint. This view is rebuilt on strict import and is never serialized.
struct FabricModuleBoundaryTransportAttachmentView final {
  FabricModuleBoundaryEndpointRef boundary;
  FabricTransportEndpointRef endpoint;

  friend bool
  operator==(const FabricModuleBoundaryTransportAttachmentView &lhs,
             const FabricModuleBoundaryTransportAttachmentView &rhs) {
    return lhs.boundary == rhs.boundary && lhs.endpoint == rhs.endpoint;
  }
  friend bool
  operator!=(const FabricModuleBoundaryTransportAttachmentView &lhs,
             const FabricModuleBoundaryTransportAttachmentView &rhs) {
    return !(lhs == rhs);
  }
};

/// One direct token-plane connection from a Module input to a Module output.
/// Both references remain boundary correspondence: the relation creates no
/// transport endpoint, traversal, resource, or persistent identity. It is
/// rebuilt mechanically from canonical Module SSA during strict import.
struct FabricModuleBoundaryTransportPassthroughView final {
  FabricModuleBoundaryEndpointRef input;
  FabricModuleBoundaryEndpointRef output;

  friend bool
  operator==(const FabricModuleBoundaryTransportPassthroughView &lhs,
             const FabricModuleBoundaryTransportPassthroughView &rhs) {
    return lhs.input == rhs.input && lhs.output == rhs.output;
  }
  friend bool
  operator!=(const FabricModuleBoundaryTransportPassthroughView &lhs,
             const FabricModuleBoundaryTransportPassthroughView &rhs) {
    return !(lhs == rhs);
  }
};

/// One occurrence-local routing terminal reachable from an exact FU port
/// through a Fabric-owned PE selector traversal. This sealed projection is
/// indexed by the fixed occurrence port and is never persisted separately.
struct FabricFuPortAttachmentView final {
  FabricTransportEndpointRef endpoint;
  FabricPhysicalTraversalRef localTraversal;

  friend bool operator==(const FabricFuPortAttachmentView &lhs,
                         const FabricFuPortAttachmentView &rhs) {
    return lhs.endpoint == rhs.endpoint &&
           lhs.localTraversal == rhs.localTraversal;
  }
};

class FabricArtifactView;
class FabricModuleRootView;
class FabricSystemRootView;

namespace detail {
struct FabricArtifactViewData;
llvm::Expected<FabricArtifactView>
buildFabricArtifactView(FabricArtifactViewData data);
} // namespace detail

/// Narrow read-only hooks into one finalized Fabric Hardware Description: the
/// owner-declared canonical inventories, the elaborated connection relation,
/// the resource-contract traversal relation, and the exact FU
/// template-to-occurrence relation. Answers come from each owner's own data,
/// so importing needs no shadow topology catalog, virtual object graph,
/// property map, or dense persistent index. Freeze may build such caches
/// afterwards; they are derived data and never enter persistent identity.
class FabricArtifactView final {
public:
  FabricArtifactView(const FabricArtifactView &) = default;
  FabricArtifactView(FabricArtifactView &&) noexcept = default;
  FabricArtifactView &operator=(const FabricArtifactView &) = default;
  FabricArtifactView &operator=(FabricArtifactView &&) noexcept = default;
  ~FabricArtifactView();

  const ArtifactIdentity &identity() const;
  FabricRootKind rootKind() const;

  /// The unique canonical Module template owned by a Module root. Other root
  /// kinds return no value.
  std::optional<FabricModuleTemplateRef> moduleRootTemplate() const;

  /// Strictly imported direct Module dependencies of a System root. Module
  /// roots have an empty range. These views retain their own Artifact identity;
  /// owner-local references are never rebound into the System root.
  llvm::ArrayRef<FabricArtifactView> importedModules() const;

  /// Kind of the entity holding `id`, or absent when the artifact declares no
  /// such entity.
  std::optional<FabricEntityKind> entityKind(FabricEntityId id) const;

  /// Canonical occurrence inventories. These ranges are the only supported
  /// way for Mapping to enumerate physical candidates; scanning EntityIds is
  /// not a persistent or native API.
  llvm::ArrayRef<FabricPeOccurrenceRef> peOccurrences() const;
  llvm::ArrayRef<FabricFuOccurrenceRef> fuOccurrences() const;
  llvm::ArrayRef<FabricMemoryOccurrenceRef> memoryOccurrences() const;
  llvm::ArrayRef<FabricSwitchOccurrenceRef> switchOccurrences() const;
  llvm::ArrayRef<FabricFifoOccurrenceRef> fifoOccurrences() const;
  llvm::ArrayRef<FabricBoundaryOccurrenceRef> boundaryOccurrences() const;
  llvm::ArrayRef<HostCoreOccurrenceRef> hostCoreOccurrences() const;
  llvm::ArrayRef<AccCoreOccurrenceRef> accCoreOccurrences() const;
  llvm::ArrayRef<SystemMemoryServiceRef> systemMemoryServices() const;
  llvm::ArrayRef<SystemServiceEndpointRef> systemServiceEndpoints() const;
  llvm::ArrayRef<SystemServiceTransformRef> systemServiceTransforms() const;
  llvm::ArrayRef<ExternalBoundaryRef> externalBoundaries() const;

  /// Derive the exact continuity action of one validated boundary occurrence.
  /// Writers have no incoming tagged domain, removers have no outgoing tagged
  /// domain, and a foreign or malformed reference returns no value.
  std::optional<FabricBoundaryTagContinuityPointView>
  boundaryTagContinuityPoint(FabricBoundaryOccurrenceRef boundary) const;

  /// Complete canonical token-endpoint inventory and its typed physical data
  /// path. The latter is decoded from the endpoint's canonical type bytes and
  /// therefore has no independent encoding authority.
  llvm::ArrayRef<FabricTransportEndpointRef> transportEndpoints() const;
  std::optional<::fabric::DataPathType>
  transportEndpointDataPath(const FabricTransportEndpointRef &endpoint) const;

  /// Complete canonical local Physical Tag interpretation-domain inventory.
  /// The endpoint query is defined only for a tagged input that performs
  /// content matching. Tagged transport-only endpoints return no domain.
  llvm::ArrayRef<FabricPhysicalTagMatchDomainView>
  physicalTagMatchDomains() const;
  std::optional<FabricOrdinal> transportEndpointTagMatchDomain(
      const FabricTransportEndpointRef &endpoint) const;

  /// Complete canonical Physical Tag assignment-point projection and direct
  /// endpoint lookup. Tagged transport outputs that only preserve a tag have
  /// no writer point.
  llvm::ArrayRef<FabricPhysicalTagAssignmentPointView>
  physicalTagAssignmentPoints() const;
  std::optional<FabricPhysicalTagAssignmentPointView>
  physicalTagAssignmentPoint(const FabricTransportEndpointRef &endpoint) const;

  /// Size of the owner's canonical token transport inventory.
  std::uint64_t
  transportEndpointCount(const FabricTransportEndpointOwnerRef &owner) const;

  /// Direction and exact canonical physical type of one token endpoint.
  /// Invalid references return no direction and an empty type range.
  std::optional<FabricPortDirection>
  transportEndpointDirection(const FabricTransportEndpointRef &endpoint) const;
  llvm::ArrayRef<std::uint8_t>
  transportEndpointType(const FabricTransportEndpointRef &endpoint) const;

  /// Size of the owner's canonical memory-service endpoint inventory. It is a
  /// separate plane, so equal ordinals never select the same object.
  std::uint64_t
  memoryEndpointCount(const FabricMemoryEndpointOwnerRef &owner) const;

  /// Size of one other owner-declared canonical inventory. Membership in an
  /// owner union never implies a nonempty inventory.
  std::uint64_t inventorySize(const FabricInventoryOwnerRef &owner,
                              FabricInventoryKind inventory) const;

  /// The complete owner-embedded resource contract, when this owner declares
  /// state, atomic use, or arbitration. ResourceState and UsePattern ranges
  /// are derived from this record and never maintained as parallel counts.
  const ::fabric::ResourceContract *
  resourceContract(const FabricInventoryOwnerRef &owner) const;

  /// Complete canonical inventory of physical owners that embed a resource
  /// contract in a fully elaborated Module root. Definition-only owners are
  /// excluded; FU operation contracts are projected through occurrence-node
  /// owners. Other root kinds have an empty range.
  llvm::ArrayRef<FabricInventoryOwnerRef> moduleResourceOwners() const;

  /// Complete canonical inventory of the Module domain members a fully
  /// elaborated Module root owns: every boundary signature face and every
  /// physical owner, mechanically derived from the same collector that
  /// `moduleResourceOwners` filters by resource contract. Globally sorted and
  /// unique by canonical member bytes. Other root kinds have an empty range.
  llvm::ArrayRef<FabricModuleDomainMemberRef> moduleDomainMembers() const;

  /// The node kind the owner's configured graph declares at `ordinal`, or
  /// absent when the owner declares no node there. One ordinal never carries
  /// more than one node kind.
  std::optional<FabricFuNodeKind>
  fuNodeKind(const FabricInventoryOwnerRef &owner, FabricOrdinal ordinal) const;

  /// Whether the memory occurrence declares its optional Local Memory Service.
  bool declaresLocalMemoryService(FabricMemoryOccurrenceRef memory) const;

  /// The complete occurrence-owned Local Memory Service contract. The
  /// returned record is the exact canonical owner projection imported from
  /// the Fabric artifact; an occurrence without a local service returns null.
  const ::fabric::MemoryServiceContractRecord *
  localMemoryService(FabricMemoryOccurrenceRef memory) const;

  /// The role the owner's inventory declares for this memory endpoint.
  std::optional<FabricMemoryEndpointRole>
  memoryEndpointRole(const FabricMemoryEndpointRef &endpoint) const;
  llvm::ArrayRef<std::uint8_t>
  memoryEndpointType(const FabricMemoryEndpointRef &endpoint) const;

  /// Exact projection of one reusable Module boundary. The signature ordinal
  /// selects the original input or result position; plane and occurrence
  /// ordinal are derived from that signature and are the sole attachment
  /// correspondence consumed by a System root.
  std::uint64_t
  moduleBoundaryEndpointCount(FabricModuleTemplateRef module,
                              FabricPortDirection direction) const;
  std::optional<FabricSpatialAttachmentEndpointRef::Plane>
  moduleBoundaryEndpointPlane(
      const FabricModuleBoundaryEndpointRef &endpoint) const;
  std::optional<FabricOrdinal> moduleBoundaryEndpointOccurrenceOrdinal(
      const FabricModuleBoundaryEndpointRef &endpoint) const;
  llvm::ArrayRef<std::uint8_t> moduleBoundaryEndpointType(
      const FabricModuleBoundaryEndpointRef &endpoint) const;
  /// Decodes the same canonical type for a token-plane endpoint. Memory-plane
  /// and invalid references have no token data path.
  std::optional<::fabric::DataPathType> moduleBoundaryEndpointDataPath(
      const FabricModuleBoundaryEndpointRef &endpoint) const;

  /// Complete canonical relation for connected token-plane Module boundary
  /// endpoints. Unused token endpoints and memory-plane endpoints have no
  /// row. Module boundary references themselves never enter the transport
  /// endpoint or traversal inventories.
  llvm::ArrayRef<FabricModuleBoundaryTransportAttachmentView>
  moduleBoundaryTransportAttachments() const;

  /// Complete canonical output-order relation for direct token-plane Module
  /// input-to-output connections. The relation is disjoint from resource
  /// attachments and has no row for memory-plane endpoints.
  llvm::ArrayRef<FabricModuleBoundaryTransportPassthroughView>
  moduleBoundaryTransportPassthroughs() const;

  /// Complete canonical relation for memory-plane Module boundary paths.
  /// Manager inputs may feed several internal manager endpoints and one
  /// subordinate endpoint may be exported through several Module results.
  llvm::ArrayRef<FabricModuleBoundaryMemoryAttachmentView>
  moduleBoundaryMemoryAttachments() const;

  /// The declared kind of one hardware domain entity.
  std::optional<FabricHardwareDomainKind>
  hardwareDomainKind(HardwareDomainRef domain) const;

  /// The exact PE scheduling domain and FU-to-PE occurrence relation.
  std::optional<::fabric::Schedule>
  peSchedule(FabricPeOccurrenceRef occurrence) const;
  std::uint64_t peResidentContextCount(FabricPeOccurrenceRef occurrence) const;
  std::optional<FabricPeOccurrenceRef>
  parentPeOf(FabricFuOccurrenceRef occurrence) const;

  /// The complete static factorized configuration schema of one Spatial PE.
  /// The view is rebuilt from canonical occurrence, port, and endpoint
  /// inventories and carries no persistent payload of its own.
  llvm::Expected<FabricSpatialPeConfigurationSchemaView>
  spatialPeConfigurationSchema(FabricPeOccurrenceRef occurrence) const;

  /// Project one direction-local FU occurrence port to the owner's canonical
  /// transport inventory, whose ordinals place all inputs before all outputs.
  /// Invalid or out-of-range ports return no endpoint.
  std::optional<FabricTransportEndpointRef>
  fuOccurrenceTransportEndpoint(FabricFuOccurrencePortRef port) const;

  /// Exact factorized terminal domain for one occurrence port. Invalid ports
  /// and ports with no declared local attachment have an empty domain.
  llvm::ArrayRef<FabricFuPortAttachmentView>
  fuOccurrencePortAttachments(FabricFuOccurrencePortRef port) const;

  /// The FU template this occurrence was elaborated from.
  std::optional<FabricFuTemplateRef>
  fuTemplateOf(FabricFuOccurrenceRef occurrence) const;

  /// Canonical owner-local definition inventories consumed by semantic
  /// matching. These are typed references in Fabric entity order; callers do
  /// not probe the global entity namespace or infer definitions from physical
  /// occurrences.
  llvm::ArrayRef<FabricFuTemplateRef> fuTemplates() const;
  llvm::ArrayRef<FabricMemoryEngineTemplateRef> memoryEngineTemplates() const;

  /// The complete canonical capability-template inventory owned by one FU
  /// definition. An invalid owner has an empty range.
  llvm::ArrayRef<FabricFuCapabilityTemplateRecord>
  fuCapabilityTemplates(FabricFuTemplateRef definition) const;

  /// The exact concrete operation capability owned by one FU template node.
  /// An occurrence-node query resolves through its immutable template
  /// relation and therefore returns the same Fabric-owned record.
  const ResolvedFabricOpCapabilityView *
  resolvedFabricOpCapability(const FabricFuTemplateNodeRef &operation) const;
  const ResolvedFabricOpCapabilityView *
  resolvedFabricOpCapability(const FabricFuOccurrenceNodeRef &operation) const;
  llvm::ArrayRef<ResolvedFabricOpCapabilityView>
  resolvedFabricOpCapabilities(FabricFuTemplateRef definition) const;

  /// The canonical Memory Operation Engine template selected by one concrete
  /// memory occurrence. Storage-only memory has no template.
  std::optional<FabricMemoryEngineTemplateRef>
  memoryEngineTemplateOf(FabricMemoryOccurrenceRef occurrence) const;
  const FabricMemoryEngineTemplateRecord *
  memoryEngineTemplate(FabricMemoryEngineTemplateRef definition) const;
  const MemoryOperationPortView *memoryEngineTemplateOperationPort(
      FabricMemoryEngineTemplateOperationPortRef port) const;
  const MemoryCapabilityAlternativeView *
  memoryEngineTemplateCapabilityAlternative(
      FabricMemoryEngineTemplateCapabilityAlternativeRef alternative) const;
  const ::fabric::MemoryTransportEndpointDescriptor *
  memoryEngineTemplateEndpoint(
      FabricMemoryEngineTemplateEndpointRef endpoint) const;
  bool hasMemoryEngineTemplateInternalConnection(
      const FabricMemoryEngineTemplateInternalConnectionRef &connection) const;

  /// The complete canonical physical operation-port inventory of one memory
  /// occurrence, and the exact immutable records selected by those refs.
  llvm::ArrayRef<FabricMemoryOperationPortRef>
  memoryOperationPorts(FabricMemoryOccurrenceRef memory) const;
  const MemoryOperationPortView *
  memoryOperationPort(FabricMemoryOperationPortRef port) const;
  const MemoryCapabilityAlternativeView *memoryCapabilityAlternative(
      FabricMemoryCapabilityAlternativeRef alternative) const;

  std::optional<::fabric::Schedule>
  memorySchedule(FabricMemoryOccurrenceRef memory) const;
  std::uint64_t
  memoryResidentContextCount(FabricMemoryOccurrenceRef memory) const;
  const ::fabric::MemoryConnectivityContractRecord *
  memoryConnectivity(FabricMemoryOccurrenceRef memory) const;

  /// Whether the fully elaborated Fabric contains the one unique directed
  /// fixed connection between exactly these endpoints.
  llvm::ArrayRef<FabricPointConnectionPayload> pointConnections() const;
  bool hasPointConnection(const FabricTransportEndpointRef &source,
                          const FabricTransportEndpointRef &destination) const;

  /// Complete explicit memory-capability identity connections. They are not
  /// transport traversals and carry no independently selectable resource.
  llvm::ArrayRef<FabricMemoryServiceConnectionPayload>
  memoryServiceConnections() const;
  bool
  hasMemoryServiceConnection(const FabricMemoryEndpointRef &manager,
                             const FabricMemoryEndpointRef &subordinate) const;

  /// Whether the owning resource contract admits this traversal.
  llvm::ArrayRef<FabricPhysicalTraversalRef> admittedTraversals() const;
  llvm::ArrayRef<FabricPhysicalTraversalView> physicalTraversals() const;
  bool admitsTraversal(const FabricPhysicalTraversalRef &traversal) const;

private:
  struct Storage;

  explicit FabricArtifactView(std::shared_ptr<const Storage> storage)
      : storage_(std::move(storage)) {}

  std::shared_ptr<const Storage> storage_;

  friend llvm::Expected<FabricArtifactView>
  detail::buildFabricArtifactView(detail::FabricArtifactViewData data);
  friend class FabricModuleRootView;
  friend class FabricSystemRootView;
};

/// The exact upstream Fabric binding a consuming root declares. A compact
/// reference that omits the digest is recovered against this binding; it never
/// permits rebinding or lookup in another Fabric artifact.
struct FabricImportBinding {
  ArtifactIdentity artifact;
  FabricRootKind rootKind;
};

/// Checks the exact artifact scope once per import.
llvm::Error checkFabricBinding(const FabricArtifactView &view,
                               const FabricImportBinding &binding);
llvm::Error checkFabricBinding(const FabricArtifactView &view,
                               const FabricImportBinding &binding,
                               const ArtifactIdentity &encoded);

//===---------------------------------------------------------------------===//
// Typed resolution
//
// Each overload resolves exactly the family its parameter names. A well-formed
// reference whose target cannot support a requested software operation is a
// Mapping feasibility failure and is deliberately never reported here.
//===---------------------------------------------------------------------===//

llvm::Error validateFabricEntity(const FabricArtifactView &view,
                                 FabricEntityKind kind, FabricEntityId id);

template <FabricEntityKind Kind>
llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const FabricTypedEntityRef<Kind> &ref) {
  return validateFabricEntity(view, Kind, ref.id());
}

llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const SpatialCoreOccurrenceRef &ref);
llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const InstructionCoreContextRef &ref);
llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const InstructionContextRef &ref);
llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const FabricModuleBoundaryEndpointRef &ref);
llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const FabricFuTemplateNodeRef &ref);
llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const FabricFuOccurrenceNodeRef &ref);
llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const FabricFuTemplatePortRef &ref);
llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const FabricFuNodePortRef &ref);
llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const FabricFuOccurrencePortRef &ref);
llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const FabricTransportEndpointOwnerRef &ref);
llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const FabricMemoryEndpointOwnerRef &ref);
llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const FabricInventoryOwnerRef &ref);
llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const FabricTransportEndpointRef &ref);
llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const FabricMemoryEndpointRef &ref);
llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const FabricMemoryOperationPortRef &ref);
llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const FabricMemoryCapabilityAlternativeRef &ref);
llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const FabricMemoryOperationContextRef &ref);
llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const FabricMemoryServiceRef &ref);
llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const FabricMemoryServiceRegionRef &ref);
llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const FabricTransferPatternRef &ref);
llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const FabricFuCapabilityTemplateRef &ref);
llvm::Error
validateFabricRef(const FabricArtifactView &view,
                  const FabricMemoryEngineTemplateOperationPortRef &ref);
llvm::Error validateFabricRef(
    const FabricArtifactView &view,
    const FabricMemoryEngineTemplateCapabilityAlternativeRef &ref);
llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const FabricMemoryEngineTemplateEndpointRef &ref);
llvm::Error
validateFabricRef(const FabricArtifactView &view,
                  const FabricMemoryEngineTemplateInternalConnectionRef &ref);
#define LOOM_FABRIC_OWNER_ROLE(Alias, Inventory, Family, Keyword)              \
  llvm::Error validateFabricRef(const FabricArtifactView &view,                \
                                const Family &ref);
#include "Fabric/Identity/FabricRefs.def"

llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const FabricPhysicalTraversalRef &ref);

//===---------------------------------------------------------------------===//
// Typed refinements
//
// A refinement adds no encoding of its own. Validation resolves the underlying
// reference and then checks the fact its owner already declares.
//===---------------------------------------------------------------------===//

llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const LocalMemoryServiceRef &ref);
llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const ManagerEndpointRef &ref);
llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const SubordinateEndpointRef &ref);
llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const MemoryConsistencyDomainRef &ref);
llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const ClockDomainRef &ref);
llvm::Error validateFabricRef(const FabricArtifactView &view,
                              const ResetDomainRef &ref);

/// Resolves one complete cross-artifact reference: exact artifact scope first,
/// then the typed Fabric-local target.
template <typename Ref>
llvm::Error importFabricRef(const FabricArtifactView &view,
                            const FabricImportBinding &binding,
                            const ArtifactReference<Ref> &ref) {
  if (llvm::Error error = checkFabricBinding(view, binding, ref.artifact))
    return error;
  return validateFabricRef(view, ref.entity);
}

/// Derives the occurrence node corresponding to `node` in `occurrence` through
/// the exact template-to-occurrence relation. Unrelated node ordinals cannot
/// be paired and textual order never implies correspondence.
llvm::Expected<FabricFuOccurrenceNodeRef>
deriveFabricFuOccurrenceNode(const FabricArtifactView &view,
                             const FabricFuTemplateNodeRef &node,
                             FabricFuOccurrenceRef occurrence);

} // namespace fabric
} // namespace loom

#endif // LOOM_FABRIC_IDENTITY_FABRICREFIMPORT_H
