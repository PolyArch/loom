#include "FabricModuleDomainMaterialization.h"

#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/ModuleDomain.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "FabricCanonicalLabeling.h"
#include "FabricModuleDomainNormalization.h"

#include "mlir/IR/Operation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"

#include <system_error>

using namespace mlir;

namespace loom::fabric::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_artifact_invalid: " + message);
}

FabricFuNodeKind fuNodeKind(Operation *operation) {
  if (isa<::fabric::MuxOp>(operation))
    return FabricFuNodeKind::Mux;
  if (isa<::fabric::DemuxOp>(operation))
    return FabricFuNodeKind::Demux;
  return FabricFuNodeKind::Op;
}

template <typename Ref>
llvm::Expected<FabricModuleDomainMemberRef> domainMember(const Ref &reference) {
  auto physical = FabricModulePhysicalOwnerRef::create(reference);
  if (!physical)
    return physical.takeError();
  return FabricModuleDomainMemberRef::of(*physical);
}

} // namespace

llvm::Error materializeFabricModuleDomainRelation(
    ::fabric::ModuleOp root, const NormalizedModuleDomainRelation &relation,
    const FabricCanonicalLabeling &labeling) {
  llvm::DenseMap<Operation *, const FabricEntityCarrier *> carrierByOperation;
  for (const FabricEntityCarrier &carrier : labeling.carriers)
    if (carrier.op)
      carrierByOperation[carrier.op] = &carrier;

  auto rootCarrier = carrierByOperation.find(root.getOperation());
  if (rootCarrier == carrierByOperation.end() ||
      rootCarrier->second->kind != FabricEntityKind::FabricModuleTemplate)
    return invalid("domain-authored Module has no canonical template");
  const FabricModuleTemplateRef module(rootCarrier->second->id);

  const auto canonicalSlot =
      [&](FabricClockResetKind kind,
          FabricOrdinal provisional) -> llvm::Expected<FabricOrdinal> {
    for (const FabricModuleDomainSlotCarrier &slot : labeling.moduleDomainSlots)
      if (slot.kind == kind && slot.provisionalOrdinal == provisional)
        return slot.canonicalOrdinal;
    return invalid("Module domain slot has no canonical ordinal");
  };

  const auto resolveInternal = [&](const NormalizedModuleDomainMember &member)
      -> llvm::Expected<FabricModuleDomainMemberRef> {
    using Role = ::fabric::ModuleDomainAuthoringRelation::InternalMemberRole;
    if (member.role == Role::FuNode) {
      auto fu = member.owner ? member.owner->getParentOfType<::fabric::FuOp>()
                             : ::fabric::FuOp();
      auto found = fu ? carrierByOperation.find(fu.getOperation())
                      : carrierByOperation.end();
      if (found == carrierByOperation.end() ||
          found->second->kind != FabricEntityKind::FabricFuOccurrence)
        return invalid("domain FU node has no canonical occurrence owner");
      auto ordinal =
          labeling.definitionFuNodeOrdinalByOperation.find(member.owner);
      if (ordinal == labeling.definitionFuNodeOrdinalByOperation.end())
        return invalid("domain FU node has no canonical node ordinal");
      return domainMember(FabricFuOccurrenceNodeRef{
          fuNodeKind(member.owner), FabricFuOccurrenceRef(found->second->id),
          ordinal->second});
    }

    auto found = carrierByOperation.find(member.owner);
    if (found == carrierByOperation.end())
      return invalid("domain member has no canonical entity carrier");
    const FabricEntityCarrier &carrier = *found->second;
    switch (member.role) {
    case Role::Occurrence:
      switch (carrier.kind) {
      case FabricEntityKind::FabricPeOccurrence:
        return domainMember(FabricPeOccurrenceRef(carrier.id));
      case FabricEntityKind::FabricFuOccurrence:
        return domainMember(FabricFuOccurrenceRef(carrier.id));
      case FabricEntityKind::FabricMemoryOccurrence:
        return domainMember(FabricMemoryOccurrenceRef(carrier.id));
      case FabricEntityKind::FabricSwitchOccurrence:
        return domainMember(FabricSwitchOccurrenceRef(carrier.id));
      case FabricEntityKind::FabricFifoOccurrence:
        return domainMember(FabricFifoOccurrenceRef(carrier.id));
      case FabricEntityKind::FabricBoundaryOccurrence:
        return domainMember(FabricBoundaryOccurrenceRef(carrier.id));
      default:
        break;
      }
      break;
    case Role::InstructionContext:
      if (carrier.kind == FabricEntityKind::FabricPeOccurrence)
        return domainMember(InstructionContextRef{
            FabricPeOccurrenceRef(carrier.id), member.ordinal});
      break;
    case Role::MemoryOperationPort:
      if (carrier.kind == FabricEntityKind::FabricMemoryOccurrence)
        return domainMember(FabricMemoryOperationPortRef{
            FabricMemoryOccurrenceRef(carrier.id), member.ordinal});
      break;
    case Role::LocalMemoryService:
      if (carrier.kind == FabricEntityKind::FabricMemoryOccurrence)
        return domainMember(LocalMemoryServiceRef(FabricMemoryServiceRef::local(
            FabricMemoryOccurrenceRef(carrier.id))));
      break;
    case Role::FuNode:
      llvm_unreachable("FU node role was resolved above");
    }
    return invalid("domain member role does not match its canonical owner");
  };

  ::fabric::CanonicalModuleDomainRelation canonical;
  canonical.slots.reserve(relation.slots.size());
  for (const NormalizedModuleDomainSlot &slot : relation.slots) {
    auto ordinal = canonicalSlot(slot.kind, slot.provisionalOrdinal);
    if (!ordinal)
      return ordinal.takeError();
    canonical.slots.push_back({module, slot.kind, *ordinal});
  }
  canonical.assignments.reserve(relation.assignments.size());
  for (const NormalizedModuleDomainAssignment &assignment :
       relation.assignments) {
    if (assignment.member >= relation.members.size() ||
        assignment.slot >= relation.slots.size())
      return invalid("Module domain relation index is out of range");
    const NormalizedModuleDomainMember &member =
        relation.members[assignment.member];
    FabricModuleDomainMemberRef resolved;
    if (member.boundary) {
      resolved =
          FabricModuleDomainMemberRef::of(FabricModuleBoundaryEndpointRef{
              module, member.direction, member.ordinal});
    } else {
      auto internal = resolveInternal(member);
      if (!internal)
        return internal.takeError();
      resolved = std::move(*internal);
    }
    const NormalizedModuleDomainSlot &slot = relation.slots[assignment.slot];
    auto ordinal = canonicalSlot(slot.kind, slot.provisionalOrdinal);
    if (!ordinal)
      return ordinal.takeError();
    canonical.assignments.push_back(
        {std::move(resolved), {module, slot.kind, *ordinal}});
  }
  llvm::sort(canonical.slots, [](const auto &left, const auto &right) {
    return loom::fabric::canonicalFabricBytes(left) <
           loom::fabric::canonicalFabricBytes(right);
  });
  llvm::sort(canonical.assignments, [](const auto &left, const auto &right) {
    return loom::fabric::canonicalFabricBytes(left) <
           loom::fabric::canonicalFabricBytes(right);
  });
  root.setDomainSlotsAttr(
      ::fabric::encodeModuleDomainSlots(root.getContext(), canonical.slots));
  root.setDomainAssignmentsAttr(::fabric::encodeModuleDomainAssignments(
      root.getContext(), canonical.assignments));
  return llvm::Error::success();
}

} // namespace loom::fabric::detail
