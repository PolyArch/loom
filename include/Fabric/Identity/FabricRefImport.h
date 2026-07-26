#ifndef LOOM_FABRIC_IDENTITY_FABRICREFIMPORT_H
#define LOOM_FABRIC_IDENTITY_FABRICREFIMPORT_H

#include "Common/Artifact.h"
#include "Fabric/IR/ResourceContract.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

namespace loom {
namespace fabric {

class FabricArtifactView;

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

  /// Kind of the entity holding `id`, or absent when the artifact declares no
  /// such entity.
  std::optional<FabricEntityKind> entityKind(FabricEntityId id) const;

  /// Size of the owner's canonical token transport inventory.
  std::uint64_t
  transportEndpointCount(const FabricTransportEndpointOwnerRef &owner) const;

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

  /// The node kind the owner's configured graph declares at `ordinal`, or
  /// absent when the owner declares no node there. One ordinal never carries
  /// more than one node kind.
  std::optional<FabricFuNodeKind>
  fuNodeKind(const FabricInventoryOwnerRef &owner, FabricOrdinal ordinal) const;

  /// Whether the memory occurrence declares its optional Local Memory Service.
  bool declaresLocalMemoryService(FabricMemoryOccurrenceRef memory) const;

  /// The role the owner's inventory declares for this memory endpoint.
  std::optional<FabricMemoryEndpointRole>
  memoryEndpointRole(const FabricMemoryEndpointRef &endpoint) const;

  /// The declared kind of one hardware domain entity.
  std::optional<FabricHardwareDomainKind>
  hardwareDomainKind(HardwareDomainRef domain) const;

  /// The FU template this occurrence was elaborated from.
  std::optional<FabricFuTemplateRef>
  fuTemplateOf(FabricFuOccurrenceRef occurrence) const;

  /// Whether the fully elaborated Fabric contains the one unique directed
  /// fixed connection between exactly these endpoints.
  llvm::ArrayRef<FabricPointConnectionPayload> pointConnections() const;
  bool hasPointConnection(const FabricTransportEndpointRef &source,
                          const FabricTransportEndpointRef &destination) const;

  /// Whether the owning resource contract admits this traversal.
  llvm::ArrayRef<FabricPhysicalTraversalRef> admittedTraversals() const;
  bool admitsTraversal(const FabricPhysicalTraversalRef &traversal) const;

private:
  struct Storage;

  explicit FabricArtifactView(std::shared_ptr<const Storage> storage)
      : storage_(std::move(storage)) {}

  std::shared_ptr<const Storage> storage_;

  friend llvm::Expected<FabricArtifactView>
  detail::buildFabricArtifactView(detail::FabricArtifactViewData data);
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
