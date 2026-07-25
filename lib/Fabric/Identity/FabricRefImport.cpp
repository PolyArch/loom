#include "Fabric/Identity/FabricRefImport.h"

#include "Fabric/Identity/FabricRefText.h"

using namespace loom;
using namespace loom::fabric;

FabricArtifactView::~FabricArtifactView() = default;

namespace {

llvm::Error ordinalOutOfRange(llvm::StringRef what, FabricOrdinal ordinal,
                              std::uint64_t bound) {
  return makeFabricRefError(FabricRefErrorKind::OrdinalOutOfRange,
                            llvm::Twine(what) + " ordinal " +
                                llvm::Twine(ordinal) + " is outside [0, " +
                                llvm::Twine(bound) + ")");
}

/// Range-checks one owner-relative ordinal against the canonical inventory the
/// consuming family selects. The owner itself is validated first, so an
/// absent inventory and an invalid owner never blur together.
llvm::Error checkInventory(const FabricArtifactView &view,
                           const FabricInventoryOwnerRef &owner,
                           FabricInventoryKind inventory,
                           FabricOrdinal ordinal) {
  if (llvm::Error error = validateFabricRef(view, owner))
    return error;
  const std::uint64_t bound = view.inventorySize(owner, inventory);
  if (ordinal >= bound)
    return ordinalOutOfRange(fabricRefKeyword(inventory), ordinal,
                             bound);
  return llvm::Error::success();
}

FabricInventoryKind portInventory(FabricPortDirection direction) {
  return direction == FabricPortDirection::Input
             ? FabricInventoryKind::InputPort
             : FabricInventoryKind::OutputPort;
}

} // namespace

llvm::Error loom::fabric::checkFabricBinding(const FabricArtifactView &view,
                                             const FabricImportBinding &binding) {
  if (view.identity() != binding.artifact)
    return makeFabricRefError(FabricRefErrorKind::ForeignArtifact,
                              "the bound Fabric artifact is not the one this "
                              "view resolves");
  if (view.rootKind() != binding.rootKind)
    return makeFabricRefError(
        FabricRefErrorKind::WrongRootKind,
        llvm::Twine("the bound Fabric root is ") +
            fabricRefKeyword(view.rootKind()) + " where " +
            fabricRefKeyword(binding.rootKind) + " is required");
  return llvm::Error::success();
}

llvm::Error loom::fabric::checkFabricBinding(const FabricArtifactView &view,
                                             const FabricImportBinding &binding,
                                             const ArtifactIdentity &encoded) {
  if (encoded != binding.artifact)
    return makeFabricRefError(FabricRefErrorKind::ForeignArtifact,
                              "the reference names a foreign Fabric artifact");
  return checkFabricBinding(view, binding);
}

llvm::Error loom::fabric::validateFabricEntity(const FabricArtifactView &view,
                                               FabricEntityKind kind,
                                               FabricEntityId id) {
  const std::optional<FabricEntityKind> actual = view.entityKind(id);
  if (!actual)
    return makeFabricRefError(FabricRefErrorKind::UnknownEntity,
                              llvm::Twine("no entity ") + llvm::Twine(id) +
                                  " in this Fabric artifact");
  if (*actual != kind)
    return makeFabricRefError(FabricRefErrorKind::WrongEntityKind,
                              llvm::Twine("entity ") + llvm::Twine(id) + " is " +
                                  fabricRefKeyword(*actual) + " where " +
                                  fabricRefKeyword(kind) +
                                  " is required");
  return llvm::Error::success();
}

//===---------------------------------------------------------------------===//
// Closed owner unions
//===---------------------------------------------------------------------===//

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricTransportEndpointOwnerRef &owner) {
  switch (owner.kind) {
#define LOOM_FABRIC_TRANSPORT_OWNER(Name, Member, Type)                        \
  case FabricTransportEndpointOwnerKind::Name:                                 \
    return validateFabricRef(view, owner.payload.Member);
#include "Fabric/Identity/FabricRefs.def"
  }
  return llvm::Error::success();
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricMemoryEndpointOwnerRef &owner) {
  switch (owner.kind) {
#define LOOM_FABRIC_MEMORY_OWNER(Name, Member, Type)                           \
  case FabricMemoryEndpointOwnerKind::Name:                                    \
    return validateFabricRef(view, owner.payload.Member);
#include "Fabric/Identity/FabricRefs.def"
  }
  return llvm::Error::success();
}

llvm::Error loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                            const FabricInventoryOwnerRef &owner) {
  switch (owner.kind) {
#define LOOM_FABRIC_INVENTORY_OWNER(Name, Member, Type)                        \
  case FabricInventoryOwnerKind::Name:                                         \
    return validateFabricRef(view, owner.payload.Member);
#include "Fabric/Identity/FabricRefs.def"
  }
  return llvm::Error::success();
}

//===---------------------------------------------------------------------===//
// Structural families
//===---------------------------------------------------------------------===//

llvm::Error loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                            const SpatialCoreOccurrenceRef &ref) {
  return validateFabricRef(view, ref.core);
}

llvm::Error loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                            const InstructionCoreContextRef &ref) {
  return validateFabricRef(view, ref.core);
}

llvm::Error loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                            const InstructionContextRef &ref) {
  return checkInventory(view, FabricInventoryOwnerRef::of(ref.pe),
                        FabricInventoryKind::InstructionContext, ref.ordinal);
}

llvm::Error loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                            const FabricFuTemplateNodeRef &ref) {
  return checkInventory(view, FabricInventoryOwnerRef::of(ref.fu),
                        FabricInventoryKind::FuNode, ref.ordinal);
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricFuOccurrenceNodeRef &ref) {
  return checkInventory(view, FabricInventoryOwnerRef::of(ref.fu),
                        FabricInventoryKind::FuNode, ref.ordinal);
}

llvm::Error loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                            const FabricFuTemplatePortRef &ref) {
  return checkInventory(view, FabricInventoryOwnerRef::of(ref.fu),
                        portInventory(ref.direction), ref.ordinal);
}

llvm::Error loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                            const FabricFuNodePortRef &ref) {
  if (llvm::Error error = validateFabricRef(view, ref.node))
    return error;
  return checkInventory(view, FabricInventoryOwnerRef::of(ref.node),
                        portInventory(ref.direction), ref.ordinal);
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricFuOccurrencePortRef &ref) {
  return checkInventory(view, FabricInventoryOwnerRef::of(ref.fu),
                        portInventory(ref.direction), ref.ordinal);
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricTransportEndpointRef &ref) {
  if (llvm::Error error = validateFabricRef(view, ref.owner))
    return error;
  const std::uint64_t bound = view.transportEndpointCount(ref.owner);
  if (ref.ordinal >= bound)
    return ordinalOutOfRange("transport endpoint", ref.ordinal, bound);
  return llvm::Error::success();
}

llvm::Error loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                            const FabricMemoryEndpointRef &ref) {
  if (llvm::Error error = validateFabricRef(view, ref.owner))
    return error;
  const std::uint64_t bound = view.memoryEndpointCount(ref.owner);
  if (ref.ordinal >= bound)
    return ordinalOutOfRange("memory endpoint", ref.ordinal, bound);
  return llvm::Error::success();
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricMemoryOperationPortRef &ref) {
  return checkInventory(view, FabricInventoryOwnerRef::of(ref.memory),
                        FabricInventoryKind::MemoryOperationPort, ref.ordinal);
}

llvm::Error loom::fabric::validateFabricRef(
    const FabricArtifactView &view,
    const FabricMemoryCapabilityAlternativeRef &ref) {
  if (llvm::Error error = validateFabricRef(view, ref.port))
    return error;
  return checkInventory(view, FabricInventoryOwnerRef::of(ref.port),
                        FabricInventoryKind::MemoryCapabilityAlternative,
                        ref.ordinal);
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricMemoryOperationContextRef &ref) {
  if (llvm::Error error = validateFabricRef(view, ref.port))
    return error;
  return checkInventory(view, FabricInventoryOwnerRef::of(ref.port),
                        FabricInventoryKind::MemoryOperationContext,
                        ref.ordinal);
}

llvm::Error loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                            const FabricMemoryServiceRef &ref) {
  switch (ref.kind) {
#define LOOM_FABRIC_MEMORY_SERVICE(Name, Keyword, Member, Type)                \
  case FabricMemoryServiceKind::Name:                                          \
    return validateFabricRef(view, ref.payload.Member);
#include "Fabric/Identity/FabricRefs.def"
  }
  return llvm::Error::success();
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricMemoryServiceRegionRef &ref) {
  return checkInventory(view, FabricInventoryOwnerRef::of(ref.service),
                        FabricInventoryKind::MemoryServiceRegion, ref.ordinal);
}

llvm::Error loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                            const FabricTransferPatternRef &ref) {
  return checkInventory(view, FabricInventoryOwnerRef::of(ref.resource),
                        FabricInventoryKind::TransferPattern, ref.ordinal);
}

#define LOOM_FABRIC_OWNER_RELATIVE_VALIDATOR(Type, Inventory)                  \
  llvm::Error loom::fabric::validateFabricRef(const FabricArtifactView &view,  \
                                              const Type &ref) {               \
    return checkInventory(view, ref.owner, FabricInventoryKind::Inventory,     \
                          ref.ordinal);                                        \
  }

LOOM_FABRIC_OWNER_RELATIVE_VALIDATOR(FabricResourceStateRef, ResourceState)
LOOM_FABRIC_OWNER_RELATIVE_VALIDATOR(FabricUsePatternRef, UsePattern)
LOOM_FABRIC_OWNER_RELATIVE_VALIDATOR(FabricSemanticConfigFieldRef,
                                     SemanticConfigField)
LOOM_FABRIC_OWNER_RELATIVE_VALIDATOR(FabricPhysicalRefinementDomainRef,
                                     RefinementDomain)

#undef LOOM_FABRIC_OWNER_RELATIVE_VALIDATOR

//===---------------------------------------------------------------------===//
// Directed physical traversals
//===---------------------------------------------------------------------===//

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricPhysicalTraversalRef &ref) {
  // Every traversal first resolves its own structural fields, so an
  // out-of-range ordinal is never reported as a resource-contract failure.
  switch (ref.kind) {
  case FabricPhysicalTraversalKind::PointConnection: {
    const FabricPointConnectionPayload &payload = ref.payload.pointConnection;
    if (llvm::Error error = validateFabricRef(view, payload.source))
      return error;
    if (llvm::Error error = validateFabricRef(view, payload.destination))
      return error;
    if (!view.hasPointConnection(payload.source, payload.destination))
      return makeFabricRefError(
          FabricRefErrorKind::AbsentPointConnection,
          "no unique directed fixed connection between these endpoints");
    return llvm::Error::success();
  }
  case FabricPhysicalTraversalKind::PeSelectorTraversal: {
    const FabricPeSelectorPayload &payload = ref.payload.peSelector;
    if (llvm::Error error = validateFabricRef(view, payload.owner))
      return error;
    if (llvm::Error error = validateFabricRef(view, payload.source))
      return error;
    if (llvm::Error error = validateFabricRef(view, payload.destination))
      return error;
    break;
  }
  case FabricPhysicalTraversalKind::PeRegisterFifoTraversal: {
    const FabricPeRegisterFifoPayload &payload = ref.payload.peRegisterFifo;
    if (llvm::Error error =
            checkInventory(view, FabricInventoryOwnerRef::of(payload.owner),
                           FabricInventoryKind::RegisterFifo,
                           payload.registerFifo))
      return error;
    break;
  }
  case FabricPhysicalTraversalKind::SwitchTraversal: {
    const FabricSwitchTraversalPayload &payload = ref.payload.switchTraversal;
    const FabricInventoryOwnerRef owner =
        FabricInventoryOwnerRef::of(payload.owner);
    if (llvm::Error error = checkInventory(
            view, owner, FabricInventoryKind::SwitchInput, payload.input))
      return error;
    if (llvm::Error error = checkInventory(
            view, owner, FabricInventoryKind::SwitchOutput, payload.output))
      return error;
    break;
  }
  case FabricPhysicalTraversalKind::FifoTraversal:
    if (llvm::Error error =
            validateFabricRef(view, ref.payload.fifoTraversal.owner))
      return error;
    break;
  case FabricPhysicalTraversalKind::BoundaryTraversal: {
    const FabricBoundaryTraversalPayload &payload =
        ref.payload.boundaryTraversal;
    if (llvm::Error error =
            checkInventory(view, FabricInventoryOwnerRef::of(payload.owner),
                           FabricInventoryKind::BoundaryOutput, payload.output))
      return error;
    break;
  }
  case FabricPhysicalTraversalKind::SystemTransferPatternLeg: {
    const FabricTransferPatternLegPayload &payload =
        ref.payload.transferPatternLeg;
    if (llvm::Error error = validateFabricRef(view, payload.owner))
      return error;
    if (llvm::Error error =
            checkInventory(view, FabricInventoryOwnerRef::of(payload.owner),
                           FabricInventoryKind::TransferPatternEgress,
                           payload.egress))
      return error;
    break;
  }
  }
  // The owning resource contract closes the remaining traversal alternatives:
  // a nonexistent switch turn, a bypass on a non-bypassable FIFO, or a
  // selector pair the PE does not expose.
  if (!view.admitsTraversal(ref))
    return makeFabricRefError(
        FabricRefErrorKind::TraversalNotAdmitted,
        llvm::Twine("the owning resource contract does not admit this ") +
            fabricRefKeyword(ref.kind) + " traversal");
  return llvm::Error::success();
}

llvm::Expected<FabricFuOccurrenceNodeRef>
loom::fabric::deriveFabricFuOccurrenceNode(const FabricArtifactView &view,
                                           const FabricFuTemplateNodeRef &node,
                                           FabricFuOccurrenceRef occurrence) {
  if (llvm::Error error = validateFabricRef(view, node))
    return std::move(error);
  if (llvm::Error error = validateFabricRef(view, occurrence))
    return std::move(error);
  const std::optional<FabricFuTemplateRef> elaborated =
      view.fuTemplateOf(occurrence);
  if (!elaborated || *elaborated != node.fu)
    return makeFabricRefError(
        FabricRefErrorKind::WrongEntityKind,
        llvm::Twine("FU occurrence ") + llvm::Twine(occurrence.id()) +
            " was not elaborated from FU template " +
            llvm::Twine(node.fu.id()));
  return FabricFuOccurrenceNodeRef{node.node, occurrence, node.ordinal};
}
