#include "Fabric/Identity/FabricRefImport.h"

#include "Fabric/Artifact/FabricModuleRootView.h"
#include "Fabric/Artifact/FabricSystemRootView.h"

#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <optional>
#include <variant>

using namespace loom;
using namespace loom::fabric;

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
    return ordinalOutOfRange(fabricRefKeyword(inventory), ordinal, bound);
  return llvm::Error::success();
}

/// An in-range node ordinal still names exactly the node kind its owner's
/// configured graph declares there.
llvm::Error checkNode(const FabricArtifactView &view,
                      const FabricInventoryOwnerRef &owner,
                      FabricFuNodeKind node, FabricOrdinal ordinal) {
  if (llvm::Error error =
          checkInventory(view, owner, FabricInventoryKind::FuNode, ordinal))
    return error;
  const std::optional<FabricFuNodeKind> declared =
      view.fuNodeKind(owner, ordinal);
  if (!declared || *declared != node)
    return makeFabricRefError(FabricRefErrorKind::WrongEntityKind,
                              llvm::Twine("node ordinal ") +
                                  llvm::Twine(ordinal) + " is not a " +
                                  fabricRefKeyword(node) + " node");
  return llvm::Error::success();
}

FabricInventoryKind portInventory(FabricPortDirection direction) {
  return direction == FabricPortDirection::Input
             ? FabricInventoryKind::InputPort
             : FabricInventoryKind::OutputPort;
}

} // namespace

llvm::Error
loom::fabric::checkFabricBinding(const FabricArtifactView &view,
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
                              llvm::Twine("entity ") + llvm::Twine(id) +
                                  " is " + fabricRefKeyword(*actual) +
                                  " where " + fabricRefKeyword(kind) +
                                  " is required");
  return llvm::Error::success();
}

//===---------------------------------------------------------------------===//
// Closed owner unions
//===---------------------------------------------------------------------===//

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricTransportEndpointOwnerRef &owner) {
  switch (owner.kind()) {
#define LOOM_FABRIC_TRANSPORT_OWNER(Ordinal, Name, Type)                       \
  case FabricTransportEndpointOwnerKind::Name:                                 \
    return validateFabricRef(view, std::get<Type>(owner.payload));
#include "Fabric/Identity/FabricRefs.def"
  }
  return llvm::Error::success();
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricMemoryEndpointOwnerRef &owner) {
  switch (owner.kind()) {
#define LOOM_FABRIC_MEMORY_OWNER(Ordinal, Name, Type)                          \
  case FabricMemoryEndpointOwnerKind::Name:                                    \
    return validateFabricRef(view, std::get<Type>(owner.payload));
#include "Fabric/Identity/FabricRefs.def"
  }
  return llvm::Error::success();
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricInventoryOwnerRef &owner) {
  switch (owner.kind()) {
#define LOOM_FABRIC_INVENTORY_OWNER(Name, Type)                                \
  case FabricInventoryOwnerKind::Name:                                         \
    return validateFabricRef(view, std::get<Type>(owner.payload));
#include "Fabric/Identity/FabricRefs.def"
  }
  return llvm::Error::success();
}

//===---------------------------------------------------------------------===//
// Structural families
//===---------------------------------------------------------------------===//

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const SpatialCoreOccurrenceRef &ref) {
  return validateFabricRef(view, ref.core);
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const InstructionCoreContextRef &ref) {
  return validateFabricRef(view, ref.core);
}

llvm::Error loom::fabric::validateFabricRef(
    const FabricArtifactView &view,
    const SpatialCoreDomainSlotOccurrenceRef &ref) {
  if (llvm::Error error = validateFabricRef(view, ref.spatialCore))
    return error;
  if (ref.kind != FabricClockResetKind::Clock &&
      ref.kind != FabricClockResetKind::Reset)
    return makeFabricRefError(FabricRefErrorKind::WrongEntityKind,
                              "hardware-domain slot has an invalid kind");
  auto system = requireSystemRoot(view);
  if (!system)
    return system.takeError();
  auto target = system->spatialCoreTarget(ref.spatialCore.core);
  if (!target || target->dependencyOrdinal >= view.importedModules().size())
    return makeFabricRefError(FabricRefErrorKind::UnknownEntity,
                              "hardware-domain slot has no imported Module");
  auto module = requireModuleRoot(
      view.importedModules()[target->dependencyOrdinal]);
  if (!module)
    return module.takeError();
  const auto moduleTemplate = module->artifact().moduleRootTemplate();
  if (!moduleTemplate || *moduleTemplate != target->target)
    return makeFabricRefError(
        FabricRefErrorKind::InvalidOwnerFamily,
        "hardware-domain slot names a different Module template");
  const FabricModuleDomainSlotRef expected{*moduleTemplate, ref.kind,
                                           ref.ordinal};
  bool declared = false;
  for (const FabricModuleDomainSlotRef &slot : module->domainSlots())
    declared |= slot == expected;
  if (!declared)
    return makeFabricRefError(FabricRefErrorKind::UnknownEntity,
                              "hardware-domain slot is not declared by its Module");
  return llvm::Error::success();
}

llvm::Error loom::fabric::validateFabricRef(
    const FabricArtifactView &view,
    const FabricHardwareDomainMemberRef &ref) {
  return std::visit([&](const auto &member) {
    return validateFabricRef(view, member);
  }, ref.payload());
}

llvm::Error loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                            const InstructionContextRef &ref) {
  return checkInventory(view, FabricInventoryOwnerRef::of(ref.pe),
                        FabricInventoryKind::InstructionContext, ref.ordinal);
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricModuleBoundaryEndpointRef &ref) {
  return checkInventory(view, FabricInventoryOwnerRef::of(ref.module),
                        portInventory(ref.direction), ref.ordinal);
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricFuTemplateNodeRef &ref) {
  return checkNode(view, FabricInventoryOwnerRef::of(ref.fu), ref.node,
                   ref.ordinal);
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricFuOccurrenceNodeRef &ref) {
  return checkNode(view, FabricInventoryOwnerRef::of(ref.fu), ref.node,
                   ref.ordinal);
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
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

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
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
  switch (ref.kind()) {
  case FabricMemoryServiceKind::Local: {
    // The Local variant exists only where the memory occurrence declares its
    // optional Local Memory Service. This is the one place that rule lives,
    // so every nested region, owner, and refined use inherits it.
    const FabricMemoryOccurrenceRef memory =
        std::get<FabricMemoryOccurrenceRef>(ref.payload);
    if (llvm::Error error = validateFabricRef(view, memory))
      return error;
    if (!view.declaresLocalMemoryService(memory))
      return makeFabricRefError(FabricRefErrorKind::WrongEntityKind,
                                llvm::Twine("memory occurrence ") +
                                    llvm::Twine(memory.id()) +
                                    " declares no Local Memory Service");
    return llvm::Error::success();
  }
  case FabricMemoryServiceKind::System:
    return validateFabricRef(view,
                             std::get<SystemMemoryServiceRef>(ref.payload));
  }
  return llvm::Error::success();
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricMemoryServiceRegionRef &ref) {
  return checkInventory(view, FabricInventoryOwnerRef::of(ref.service),
                        FabricInventoryKind::MemoryServiceRegion, ref.ordinal);
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricTransferPatternRef &ref) {
  return checkInventory(view, FabricInventoryOwnerRef::of(ref.resource),
                        FabricInventoryKind::TransferPattern, ref.ordinal);
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricFuCapabilityTemplateRef &ref) {
  if (llvm::Error error = validateFabricRef(view, ref.fu))
    return error;
  const std::uint64_t bound = view.fuCapabilityTemplates(ref.fu).size();
  if (ref.ordinal >= bound)
    return ordinalOutOfRange("FU capability template", ref.ordinal, bound);
  return llvm::Error::success();
}

llvm::Error loom::fabric::validateFabricRef(
    const FabricArtifactView &view,
    const FabricMemoryEngineTemplateOperationPortRef &ref) {
  if (llvm::Error error = validateFabricRef(view, ref.engine))
    return error;
  const FabricMemoryEngineTemplateRecord *engine =
      view.memoryEngineTemplate(ref.engine);
  const std::uint64_t bound = engine ? engine->operationPorts.size() : 0;
  if (ref.ordinal >= bound)
    return ordinalOutOfRange("memory engine operation port", ref.ordinal,
                             bound);
  return llvm::Error::success();
}

llvm::Error loom::fabric::validateFabricRef(
    const FabricArtifactView &view,
    const FabricMemoryEngineTemplateCapabilityAlternativeRef &ref) {
  if (llvm::Error error = validateFabricRef(view, ref.port))
    return error;
  const MemoryOperationPortView *port =
      view.memoryEngineTemplateOperationPort(ref.port);
  const std::uint64_t bound = port ? port->capabilityAlternatives().size() : 0;
  if (ref.ordinal >= bound)
    return ordinalOutOfRange("memory engine capability alternative",
                             ref.ordinal, bound);
  return llvm::Error::success();
}

llvm::Error loom::fabric::validateFabricRef(
    const FabricArtifactView &view,
    const FabricMemoryEngineTemplateEndpointRef &ref) {
  if (llvm::Error error = validateFabricRef(view, ref.engine))
    return error;
  const FabricMemoryEngineTemplateRecord *engine =
      view.memoryEngineTemplate(ref.engine);
  const std::uint64_t bound = engine ? engine->tokenEndpoints.size() : 0;
  if (ref.ordinal >= bound)
    return ordinalOutOfRange("memory engine endpoint", ref.ordinal, bound);
  return llvm::Error::success();
}

llvm::Error loom::fabric::validateFabricRef(
    const FabricArtifactView &view,
    const FabricMemoryEngineTemplateInternalConnectionRef &ref) {
  if (llvm::Error error = validateFabricRef(view, ref.engine))
    return error;
  if (ref.source.engine != ref.engine || ref.sink.engine != ref.engine)
    return makeFabricRefError(
        FabricRefErrorKind::WrongOwner,
        "memory engine internal connection endpoints have the wrong owner");
  if (llvm::Error error = validateFabricRef(view, ref.source))
    return error;
  if (llvm::Error error = validateFabricRef(view, ref.sink))
    return error;
  if (!view.hasMemoryEngineTemplateInternalConnection(ref))
    return makeFabricRefError(
        FabricRefErrorKind::TraversalNotAdmitted,
        "memory engine template does not admit this internal connection");
  return llvm::Error::success();
}

#define LOOM_FABRIC_OWNER_ROLE(Alias, Inventory, Family, Keyword)              \
  llvm::Error loom::fabric::validateFabricRef(const FabricArtifactView &view,  \
                                              const Family &ref) {             \
    return checkInventory(view, ref.owner.catalog(),                           \
                          FabricInventoryKind::Inventory, ref.ordinal);        \
  }
#include "Fabric/Identity/FabricRefs.def"

//===---------------------------------------------------------------------===//
// Directed physical traversals
//===---------------------------------------------------------------------===//

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricPhysicalTraversalRef &ref) {
  // Every traversal first resolves its own structural fields, so an
  // out-of-range ordinal is never reported as a resource-contract failure.
  switch (ref.kind()) {
  case FabricPhysicalTraversalKind::PointConnection: {
    const FabricPointConnectionPayload &payload =
        std::get<FabricPointConnectionPayload>(ref.payload);
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
    const FabricPeSelectorPayload &payload =
        std::get<FabricPeSelectorPayload>(ref.payload);
    if (llvm::Error error = validateFabricRef(view, payload.owner))
      return error;
    if (llvm::Error error = validateFabricRef(view, payload.source))
      return error;
    if (llvm::Error error = validateFabricRef(view, payload.destination))
      return error;
    break;
  }
  case FabricPhysicalTraversalKind::PeRegisterFifoTraversal: {
    const FabricPeRegisterFifoPayload &payload =
        std::get<FabricPeRegisterFifoPayload>(ref.payload);
    if (llvm::Error error = checkInventory(
            view, FabricInventoryOwnerRef::of(payload.owner),
            FabricInventoryKind::RegisterFifo, payload.registerFifo))
      return error;
    break;
  }
  case FabricPhysicalTraversalKind::SwitchTraversal: {
    const FabricSwitchTraversalPayload &payload =
        std::get<FabricSwitchTraversalPayload>(ref.payload);
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
    if (llvm::Error error = validateFabricRef(
            view, std::get<FabricFifoTraversalPayload>(ref.payload).owner))
      return error;
    break;
  case FabricPhysicalTraversalKind::BoundaryTraversal: {
    const FabricBoundaryTraversalPayload &payload =
        std::get<FabricBoundaryTraversalPayload>(ref.payload);
    if (llvm::Error error =
            checkInventory(view, FabricInventoryOwnerRef::of(payload.owner),
                           FabricInventoryKind::BoundaryOutput, payload.output))
      return error;
    break;
  }
  case FabricPhysicalTraversalKind::SystemTransferPatternLeg: {
    const FabricTransferPatternLegPayload &payload =
        std::get<FabricTransferPatternLegPayload>(ref.payload);
    if (llvm::Error error = validateFabricRef(view, payload.owner))
      return error;
    if (llvm::Error error = checkInventory(
            view, FabricInventoryOwnerRef::of(payload.owner),
            FabricInventoryKind::TransferPatternEgress, payload.egress))
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
            fabricRefKeyword(ref.kind()) + " traversal");
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
    return makeFabricRefError(FabricRefErrorKind::WrongOwner,
                              llvm::Twine("FU occurrence ") +
                                  llvm::Twine(occurrence.id()) +
                                  " was not elaborated from FU template " +
                                  llvm::Twine(node.fu.id()));
  return FabricFuOccurrenceNodeRef{node.node, occurrence, node.ordinal};
}

//===---------------------------------------------------------------------===//
// Typed refinements
//===---------------------------------------------------------------------===//

llvm::Error loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                            const LocalMemoryServiceRef &ref) {
  // The refined name only narrows the accepted variant; presence of the
  // service remains the generic reference's rule.
  if (ref.underlying().kind() != FabricMemoryServiceKind::Local)
    return makeFabricRefError(FabricRefErrorKind::WrongEntityKind,
                              "a local memory service reference selects the "
                              "System variant");
  return validateFabricRef(view, ref.underlying());
}

/// The owner inventory decides which refined endpoint name applies; the
/// reference never carries a copied role field.
static llvm::Error checkEndpointRole(const FabricArtifactView &view,
                                     const FabricMemoryEndpointRef &endpoint,
                                     FabricMemoryEndpointRole required) {
  if (llvm::Error error = validateFabricRef(view, endpoint))
    return error;
  const std::optional<FabricMemoryEndpointRole> declared =
      view.memoryEndpointRole(endpoint);
  if (!declared || *declared != required)
    return makeFabricRefError(FabricRefErrorKind::WrongEntityKind,
                              llvm::Twine("the owner inventory does not "
                                          "declare this endpoint ") +
                                  fabricRefKeyword(required));
  return llvm::Error::success();
}

llvm::Error loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                            const ManagerEndpointRef &ref) {
  return checkEndpointRole(view, ref.underlying(),
                           FabricMemoryEndpointRole::Manager);
}

llvm::Error loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                            const SubordinateEndpointRef &ref) {
  return checkEndpointRole(view, ref.underlying(),
                           FabricMemoryEndpointRole::Subordinate);
}

static llvm::Error checkHardwareDomainKind(const FabricArtifactView &view,
                                           HardwareDomainRef domain,
                                           FabricHardwareDomainKind required) {
  if (llvm::Error error = validateFabricRef(view, domain))
    return error;
  const std::optional<FabricHardwareDomainKind> declared =
      view.hardwareDomainKind(domain);
  if (!declared || *declared != required)
    return makeFabricRefError(FabricRefErrorKind::WrongEntityKind,
                              llvm::Twine("hardware domain ") +
                                  llvm::Twine(domain.id()) + " is not a " +
                                  fabricRefKeyword(required) + " domain");
  return llvm::Error::success();
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const MemoryConsistencyDomainRef &ref) {
  return checkHardwareDomainKind(view, ref.underlying(),
                                 FabricHardwareDomainKind::MemoryConsistency);
}

llvm::Error loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                            const ClockDomainRef &ref) {
  return checkHardwareDomainKind(view, ref.underlying(),
                                 FabricHardwareDomainKind::Clock);
}

llvm::Error loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                            const ResetDomainRef &ref) {
  return checkHardwareDomainKind(view, ref.underlying(),
                                 FabricHardwareDomainKind::Reset);
}
