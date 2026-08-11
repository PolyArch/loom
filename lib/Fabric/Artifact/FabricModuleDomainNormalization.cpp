#include "FabricModuleDomainNormalization.h"

#include "Fabric/IR/FabricCanonicalEntity.h"
#include "Fabric/IR/FabricOps.h"
#include "FabricMemoryEngineTemplate.h"
#include "FabricModuleDomainMaterialization.h"
#include "FabricModuleViewBuilding.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <map>
#include <optional>
#include <set>
#include <system_error>
#include <type_traits>
#include <utility>

using namespace mlir;

namespace loom::fabric::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_artifact_invalid: " + message);
}

std::size_t findOrAddMember(NormalizedModuleDomainRelation &relation,
                            NormalizedModuleDomainMember member) {
  auto found = llvm::find(relation.members, member);
  if (found != relation.members.end())
    return static_cast<std::size_t>(found - relation.members.begin());
  relation.members.push_back(member);
  return relation.members.size() - 1;
}

llvm::Expected<std::size_t>
findSlot(const NormalizedModuleDomainRelation &relation,
         FabricClockResetKind kind, FabricOrdinal ordinal) {
  auto found = llvm::find_if(relation.slots, [&](const auto &slot) {
    return slot.kind == kind && slot.provisionalOrdinal == ordinal;
  });
  if (found == relation.slots.end())
    return invalid("Module assignment selects an undeclared slot");
  return static_cast<std::size_t>(found - relation.slots.begin());
}

struct StoredEntity final {
  FabricEntityKind kind = FabricEntityKind::FabricModuleTemplate;
  Operation *operation = nullptr;
};

std::optional<FabricEntityKind> storedEntityKind(Operation *operation,
                                                 Operation *root) {
  if (operation == root)
    return FabricEntityKind::FabricModuleTemplate;
  if (auto symbol = dyn_cast<SymbolOpInterface>(operation))
    if (symbol.getNameAttr())
      return std::nullopt;
  if (isa<::fabric::PeOp>(operation))
    return FabricEntityKind::FabricPeOccurrence;
  if (isa<::fabric::FuOp>(operation))
    return FabricEntityKind::FabricFuOccurrence;
  if (isa<::fabric::MemOp>(operation))
    return FabricEntityKind::FabricMemoryOccurrence;
  if (isa<::fabric::SwitchOp>(operation))
    return FabricEntityKind::FabricSwitchOccurrence;
  if (isa<::fabric::FifoOp>(operation))
    return FabricEntityKind::FabricFifoOccurrence;
  if (isa<::fabric::BoundaryOp>(operation))
    return FabricEntityKind::FabricBoundaryOccurrence;
  return std::nullopt;
}

llvm::Error validateStoredDerivedIdentifiers(::fabric::ModuleOp root) {
  bool invalidCarrier = false;
  root->walk([&](Operation *operation) {
    const bool entityCarrier =
        storedEntityKind(operation, root.getOperation()).has_value();
    const bool hasEntity = operation->hasAttr(::fabric::kEntityIdAttrName);
    const bool hasFuTemplate =
        operation->hasAttr(::fabric::kFuTemplateIdAttrName);
    const bool hasMemoryTemplate =
        operation->hasAttr(::fabric::kMemoryEngineTemplateIdAttrName);
    if ((hasEntity && !entityCarrier) ||
        (hasFuTemplate && !isa<::fabric::FuOp>(operation)) ||
        (hasMemoryTemplate && !isa<::fabric::MemOp>(operation))) {
      invalidCarrier = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return invalidCarrier
             ? invalid("derived identifier is attached to a non-carrier")
             : llvm::Error::success();
}

llvm::Expected<std::map<FabricEntityId, StoredEntity>>
collectStoredEntities(::fabric::ModuleOp root) {
  std::map<FabricEntityId, StoredEntity> entities;
  llvm::Error error = llvm::Error::success();
  root->walk([&](Operation *operation) {
    if (error)
      return WalkResult::interrupt();
    std::optional<FabricEntityKind> kind =
        storedEntityKind(operation, root.getOperation());
    if (!kind)
      return WalkResult::advance();
    auto id = operation->getAttrOfType<::fabric::EntityIdAttr>(
        ::fabric::kEntityIdAttrName);
    if (!id) {
      error = invalid("stored Module entity has no EntityId lookup key");
      return WalkResult::interrupt();
    }
    if (!entities.emplace(id.getId(), StoredEntity{*kind, operation}).second) {
      error = invalid("stored Module entity IDs are not unique");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (error)
    return std::move(error);
  return entities;
}

using CanonicalFuNodesByOccurrence =
    llvm::DenseMap<Operation *, std::vector<Operation *>>;

llvm::Expected<CanonicalFuNodesByOccurrence> collectCanonicalFuNodes(
    const std::map<FabricEntityId, StoredEntity> &entities) {
  CanonicalFuNodesByOccurrence nodesByOccurrence;
  for (const auto &entry : entities) {
    const StoredEntity &entity = entry.second;
    if (entity.kind != FabricEntityKind::FabricFuOccurrence)
      continue;
    auto fu = dyn_cast_or_null<::fabric::FuOp>(entity.operation);
    if (!fu)
      return invalid("stored FU occurrence has no operation carrier");
    auto definition = computeCanonicalFabricFuDefinition(fu);
    if (!definition)
      return definition.takeError();
    nodesByOccurrence[entity.operation] =
        std::move(definition->canonicalNodeOrder);
  }
  return nodesByOccurrence;
}

llvm::Expected<Operation *>
resolveStoredEntity(const std::map<FabricEntityId, StoredEntity> &entities,
                    FabricEntityId id, FabricEntityKind kind) {
  auto found = entities.find(id);
  if (found == entities.end())
    return invalid("Module domain member names an unknown entity");
  if (found->second.kind != kind)
    return invalid("Module domain member names an entity of the wrong kind");
  return found->second.operation;
}

NormalizedModuleDomainMember
internalMember(Operation *owner,
               ::fabric::ModuleDomainAuthoringRelation::InternalMemberRole role,
               FabricOrdinal ordinal = 0) {
  NormalizedModuleDomainMember member;
  member.owner = owner;
  member.role = role;
  member.ordinal = ordinal;
  return member;
}

llvm::Expected<Operation *>
resolveStoredFuNode(Operation *owner, FabricFuNodeKind kind,
                    FabricOrdinal ordinal,
                    const CanonicalFuNodesByOccurrence &nodesByOccurrence) {
  auto fu = dyn_cast_or_null<::fabric::FuOp>(owner);
  if (!fu)
    return invalid("Module domain FU node has no FU occurrence owner");
  auto nodes = nodesByOccurrence.find(owner);
  if (nodes == nodesByOccurrence.end())
    return invalid("Module domain FU node has no canonical definition");
  if (ordinal >= nodes->second.size())
    return invalid("Module domain FU node ordinal is out of range");
  Operation *node = nodes->second[ordinal];
  if (classifyFabricFuNode(node) != kind)
    return invalid("Module domain FU node kind does not match its lookup key");
  return node;
}

llvm::Expected<NormalizedModuleDomainMember> normalizeStoredInternalMember(
    const FabricModulePhysicalOwnerRef &physical,
    const std::map<FabricEntityId, StoredEntity> &entities,
    const CanonicalFuNodesByOccurrence &nodesByOccurrence) {
  using Role = ::fabric::ModuleDomainAuthoringRelation::InternalMemberRole;
  return std::visit(
      [&](const auto &reference)
          -> llvm::Expected<NormalizedModuleDomainMember> {
        using Reference = std::decay_t<decltype(reference)>;
        Operation *owner = nullptr;
        if constexpr (std::is_same_v<Reference, FabricPeOccurrenceRef>) {
          auto resolved = resolveStoredEntity(
              entities, reference.id(), FabricEntityKind::FabricPeOccurrence);
          if (!resolved)
            return resolved.takeError();
          owner = *resolved;
          return internalMember(owner, Role::Occurrence);
        } else if constexpr (std::is_same_v<Reference, FabricFuOccurrenceRef>) {
          auto resolved = resolveStoredEntity(
              entities, reference.id(), FabricEntityKind::FabricFuOccurrence);
          if (!resolved)
            return resolved.takeError();
          owner = *resolved;
          return internalMember(owner, Role::Occurrence);
        } else if constexpr (std::is_same_v<Reference,
                                            FabricFuOccurrenceNodeRef>) {
          auto resolved =
              resolveStoredEntity(entities, reference.fu.id(),
                                  FabricEntityKind::FabricFuOccurrence);
          if (!resolved)
            return resolved.takeError();
          auto node = resolveStoredFuNode(*resolved, reference.node,
                                          reference.ordinal, nodesByOccurrence);
          if (!node)
            return node.takeError();
          return internalMember(*node, Role::FuNode);
        } else if constexpr (std::is_same_v<Reference,
                                            FabricMemoryOccurrenceRef>) {
          auto resolved =
              resolveStoredEntity(entities, reference.id(),
                                  FabricEntityKind::FabricMemoryOccurrence);
          if (!resolved)
            return resolved.takeError();
          owner = *resolved;
          return internalMember(owner, Role::Occurrence);
        } else if constexpr (std::is_same_v<Reference,
                                            FabricMemoryOperationPortRef>) {
          auto resolved =
              resolveStoredEntity(entities, reference.memory.id(),
                                  FabricEntityKind::FabricMemoryOccurrence);
          if (!resolved)
            return resolved.takeError();
          owner = *resolved;
          return internalMember(owner, Role::MemoryOperationPort,
                                reference.ordinal);
        } else if constexpr (std::is_same_v<Reference, LocalMemoryServiceRef>) {
          const FabricMemoryServiceRef &service = reference.underlying();
          if (service.kind() != FabricMemoryServiceKind::Local)
            return invalid("Module domain local service names a System owner");
          const auto &memory =
              std::get<FabricMemoryOccurrenceRef>(service.payload);
          auto resolved = resolveStoredEntity(
              entities, memory.id(), FabricEntityKind::FabricMemoryOccurrence);
          if (!resolved)
            return resolved.takeError();
          owner = *resolved;
          return internalMember(owner, Role::LocalMemoryService);
        } else if constexpr (std::is_same_v<Reference,
                                            FabricSwitchOccurrenceRef>) {
          auto resolved =
              resolveStoredEntity(entities, reference.id(),
                                  FabricEntityKind::FabricSwitchOccurrence);
          if (!resolved)
            return resolved.takeError();
          owner = *resolved;
          return internalMember(owner, Role::Occurrence);
        } else if constexpr (std::is_same_v<Reference,
                                            FabricFifoOccurrenceRef>) {
          auto resolved = resolveStoredEntity(
              entities, reference.id(), FabricEntityKind::FabricFifoOccurrence);
          if (!resolved)
            return resolved.takeError();
          owner = *resolved;
          return internalMember(owner, Role::Occurrence);
        } else if constexpr (std::is_same_v<Reference,
                                            FabricBoundaryOccurrenceRef>) {
          auto resolved =
              resolveStoredEntity(entities, reference.id(),
                                  FabricEntityKind::FabricBoundaryOccurrence);
          if (!resolved)
            return resolved.takeError();
          owner = *resolved;
          return internalMember(owner, Role::Occurrence);
        } else if constexpr (std::is_same_v<Reference, InstructionContextRef>) {
          auto resolved =
              resolveStoredEntity(entities, reference.pe.id(),
                                  FabricEntityKind::FabricPeOccurrence);
          if (!resolved)
            return resolved.takeError();
          owner = *resolved;
          return internalMember(owner, Role::InstructionContext,
                                reference.ordinal);
        }
        llvm_unreachable("closed Module physical owner kind");
      },
      physical.payload());
}

llvm::Expected<NormalizedModuleDomainMember>
normalizeStoredMember(::fabric::ModuleOp root, FabricModuleTemplateRef module,
                      const FabricModuleDomainMemberRef &stored,
                      const std::map<FabricEntityId, StoredEntity> &entities,
                      const CanonicalFuNodesByOccurrence &nodesByOccurrence) {
  if (stored.kind() == FabricModuleDomainMemberKind::Boundary) {
    const auto &boundary =
        std::get<FabricModuleBoundaryEndpointRef>(stored.payload);
    if (boundary.module != module)
      return invalid("Module domain boundary names a foreign Module");
    if (static_cast<std::uint32_t>(boundary.direction) >=
        fabricClosedBound(FabricPortDirection{}))
      return invalid("Module domain boundary has an unknown direction");
    const FabricOrdinal bound = boundary.direction == FabricPortDirection::Input
                                    ? root.getFunctionType().getNumInputs()
                                    : root.getFunctionType().getNumResults();
    if (boundary.ordinal >= bound)
      return invalid("Module domain boundary ordinal is out of range");
    NormalizedModuleDomainMember member;
    member.boundary = true;
    member.direction = boundary.direction;
    member.ordinal = boundary.ordinal;
    return member;
  }
  return normalizeStoredInternalMember(
      std::get<FabricModulePhysicalOwnerRef>(stored.payload), entities,
      nodesByOccurrence);
}

llvm::Expected<NormalizedModuleDomainRelation>
reconstructStoredFabricModuleDomain(::fabric::ModuleOp root) {
  ArrayAttr slotsAttribute = root.getDomainSlotsAttr();
  ArrayAttr assignmentsAttribute = root.getDomainAssignmentsAttr();
  if (!slotsAttribute || !assignmentsAttribute)
    return invalid("canonical Module has no complete domain carrier");
  auto slots = ::fabric::decodeModuleDomainSlots(slotsAttribute);
  if (!slots)
    return slots.takeError();
  auto assignments =
      ::fabric::decodeModuleDomainAssignments(assignmentsAttribute);
  if (!assignments)
    return assignments.takeError();

  auto entities = collectStoredEntities(root);
  if (!entities)
    return entities.takeError();
  auto canonicalFuNodes = collectCanonicalFuNodes(*entities);
  if (!canonicalFuNodes)
    return canonicalFuNodes.takeError();
  auto moduleId =
      root->getAttrOfType<::fabric::EntityIdAttr>(::fabric::kEntityIdAttrName);
  if (!moduleId)
    return invalid("stored Module has no EntityId lookup key");
  const FabricModuleTemplateRef module(moduleId.getId());

  NormalizedModuleDomainRelation relation;
  std::set<std::pair<std::uint32_t, FabricOrdinal>> slotKeys;
  relation.slots.reserve(slots->size());
  for (const FabricModuleDomainSlotRef &slot : *slots) {
    if (slot.module != module)
      return invalid("Module domain slot names a foreign Module");
    if (static_cast<std::uint32_t>(slot.kind) >=
        fabricClosedBound(FabricClockResetKind{}))
      return invalid("Module domain slot has an unknown kind");
    const auto key =
        std::make_pair(static_cast<std::uint32_t>(slot.kind), slot.ordinal);
    if (!slotKeys.insert(key).second)
      return invalid("Module domain slot lookup keys are not unique");
    relation.slots.push_back({slot.kind, slot.ordinal});
  }

  relation.assignments.reserve(assignments->size());
  for (const ModuleDomainAssignment &assignment : *assignments) {
    if (assignment.slot.module != module)
      return invalid("Module domain assignment selects a foreign Module");
    auto slot =
        findSlot(relation, assignment.slot.kind, assignment.slot.ordinal);
    if (!slot)
      return slot.takeError();
    auto member = normalizeStoredMember(root, module, assignment.member,
                                        *entities, *canonicalFuNodes);
    if (!member)
      return member.takeError();
    relation.assignments.push_back(
        {findOrAddMember(relation, std::move(*member)), *slot});
  }
  return relation;
}

llvm::Error validateStoredGraphOrder(::fabric::ModuleOp root,
                                     const FabricCanonicalLabeling &labeling) {
  llvm::DenseMap<Operation *, std::uint64_t> rank;
  for (auto [ordinal, operation] :
       llvm::enumerate(labeling.canonicalOperationOrder))
    rank[operation] = ordinal;

  llvm::Error error = llvm::Error::success();
  root->walk([&](Operation *container) {
    if (error)
      return WalkResult::interrupt();
    if (!isa<::fabric::ModuleOp, ::fabric::PeOp, ::fabric::FuOp>(container))
      return WalkResult::advance();
    const bool fuDefinition = isa<::fabric::FuOp>(container);
    for (Region &region : container->getRegions())
      for (Block &block : region) {
        std::optional<std::uint64_t> previous;
        for (Operation &operation : block) {
          if (operation.hasTrait<OpTrait::IsTerminator>())
            continue;
          std::optional<std::uint64_t> current;
          if (fuDefinition) {
            auto found =
                labeling.definitionFuNodeOrdinalByOperation.find(&operation);
            if (found != labeling.definitionFuNodeOrdinalByOperation.end())
              current = found->second;
          } else {
            auto found = rank.find(&operation);
            if (found != rank.end())
              current = found->second;
          }
          if (!current || (previous && *previous >= *current)) {
            error = invalid(
                "canonical Module graph operation order is not canonical");
            return WalkResult::interrupt();
          }
          previous = *current;
        }
      }
    return WalkResult::advance();
  });
  return error;
}

} // namespace

llvm::Expected<NormalizedModuleDomainRelation> normalizeFabricModuleDomain(
    ::fabric::ModuleOp root,
    const ::fabric::ModuleDomainAuthoringRelation &authoring) {
  if (llvm::Error error =
          authoring.validateTotality(root.getFunctionType().getNumInputs(),
                                     root.getFunctionType().getNumResults()))
    return std::move(error);

  NormalizedModuleDomainRelation relation;
  for (FabricClockResetKind kind :
       {FabricClockResetKind::Clock, FabricClockResetKind::Reset})
    for (FabricOrdinal ordinal = 0; ordinal < authoring.declaredSlotCount(kind);
         ++ordinal)
      relation.slots.push_back({kind, ordinal});

  const auto append = [&](NormalizedModuleDomainMember member,
                          FabricClockResetKind kind,
                          FabricOrdinal ordinal) -> llvm::Error {
    auto slot = findSlot(relation, kind, ordinal);
    if (!slot)
      return slot.takeError();
    relation.assignments.push_back({findOrAddMember(relation, member), *slot});
    return llvm::Error::success();
  };
  if (llvm::Error error = authoring.visitAssignments(
          [&](FabricPortDirection direction, FabricOrdinal memberOrdinal,
              FabricClockResetKind kind,
              FabricOrdinal slotOrdinal) -> llvm::Error {
            NormalizedModuleDomainMember member;
            member.boundary = true;
            member.direction = direction;
            member.ordinal = memberOrdinal;
            return append(member, kind, slotOrdinal);
          },
          [&](Operation *owner,
              ::fabric::ModuleDomainAuthoringRelation::InternalMemberRole role,
              FabricOrdinal memberOrdinal, FabricClockResetKind kind,
              FabricOrdinal slotOrdinal) -> llvm::Error {
            NormalizedModuleDomainMember member;
            member.owner = owner;
            member.role = role;
            member.ordinal = memberOrdinal;
            return append(member, kind, slotOrdinal);
          }))
    return std::move(error);
  return relation;
}

llvm::Expected<NormalizedModuleDomainRelation>
buildDefaultFabricModuleDomain(::fabric::ModuleOp root) {
  ::fabric::ModuleDomainAuthoringRelation authoring;
  using Role = ::fabric::ModuleDomainAuthoringRelation::InternalMemberRole;
  const auto note = [&](Operation *owner, Role role,
                        FabricOrdinal ordinal = 0) -> llvm::Error {
    return authoring.noteInternalMember(owner, role, ordinal);
  };

  llvm::Error walkError = llvm::Error::success();
  root->walk([&](Operation *operation) {
    if (walkError)
      return WalkResult::interrupt();
    if (operation != root.getOperation())
      if (auto symbol = dyn_cast<SymbolOpInterface>(operation))
        if (symbol.getNameAttr())
          return WalkResult::skip();
    if (auto pe = dyn_cast<::fabric::PeOp>(operation)) {
      if ((walkError = note(operation, Role::Occurrence)))
        return WalkResult::interrupt();
      std::uint64_t contextCount = 1;
      if (pe.getSchedule() == ::fabric::Schedule::Temporal) {
        auto count = pe.getNumInstruction();
        if (!count || *count <= 0) {
          walkError = invalid("a temporal PE occurrence has no contexts");
          return WalkResult::interrupt();
        }
        contextCount = static_cast<std::uint64_t>(*count);
      }
      for (FabricOrdinal ordinal = 0; ordinal < contextCount; ++ordinal)
        if ((walkError = note(operation, Role::InstructionContext, ordinal)))
          return WalkResult::interrupt();
      return WalkResult::advance();
    }
    if (isa<::fabric::FuOp>(operation)) {
      if ((walkError = note(operation, Role::Occurrence)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    }
    if (isa<::fabric::OpOp, ::fabric::MuxOp, ::fabric::DemuxOp>(operation)) {
      if ((walkError = note(operation, Role::FuNode)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    }
    if (auto memory = dyn_cast<::fabric::MemOp>(operation)) {
      if ((walkError = note(operation, Role::Occurrence)))
        return WalkResult::interrupt();
      auto engine = deriveFabricMemoryEngineTemplate(memory);
      if (!engine) {
        walkError = engine.takeError();
        return WalkResult::interrupt();
      }
      if (*engine)
        for (FabricOrdinal ordinal = 0;
             ordinal < (**engine).record.operationPorts.size(); ++ordinal)
          if ((walkError = note(operation, Role::MemoryOperationPort, ordinal)))
            return WalkResult::interrupt();
      if (memory.getMemoryContract().getLocalService())
        if ((walkError = note(operation, Role::LocalMemoryService)))
          return WalkResult::interrupt();
      return WalkResult::advance();
    }
    if (isa<::fabric::SwitchOp, ::fabric::FifoOp, ::fabric::BoundaryOp>(
            operation))
      if ((walkError = note(operation, Role::Occurrence)))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (walkError)
    return std::move(walkError);
  if (llvm::Error error = authoring.ensureDefaultAssignments(
          root.getFunctionType().getNumInputs(),
          root.getFunctionType().getNumResults()))
    return std::move(error);
  return normalizeFabricModuleDomain(root, authoring);
}

llvm::Expected<::fabric::ModuleDomainAuthoringRelation>
recoverFabricModuleDomainAuthoring(::fabric::ModuleOp root) {
  auto normalized = reconstructStoredFabricModuleDomain(root);
  if (!normalized)
    return normalized.takeError();

  ::fabric::ModuleDomainAuthoringRelation relation;
  for (const NormalizedModuleDomainSlot &slot : normalized->slots) {
    auto ordinal = relation.declareSlot(slot.kind);
    if (!ordinal)
      return ordinal.takeError();
    if (*ordinal != slot.provisionalOrdinal)
      return invalid("stored Module domain slots are not dense");
  }
  for (const NormalizedModuleDomainMember &member : normalized->members) {
    if (member.boundary)
      continue;
    if (llvm::Error error = relation.noteInternalMember(
            member.owner, member.role, member.ordinal))
      return std::move(error);
  }
  for (const NormalizedModuleDomainAssignment &assignment :
       normalized->assignments) {
    if (assignment.member >= normalized->members.size() ||
        assignment.slot >= normalized->slots.size())
      return invalid("stored Module domain relation has an invalid ordinal");
    const NormalizedModuleDomainMember &member =
        normalized->members[assignment.member];
    const NormalizedModuleDomainSlot &slot = normalized->slots[assignment.slot];
    llvm::Error error =
        member.boundary
            ? relation.assignBoundary(member.direction, member.ordinal,
                                      slot.kind, slot.provisionalOrdinal)
            : relation.assignInternal(member.owner, member.role, member.ordinal,
                                      slot.kind, slot.provisionalOrdinal);
    if (error)
      return std::move(error);
  }
  return relation;
}

llvm::Expected<FabricCanonicalLabeling>
validateStoredFabricModuleDomain(::fabric::ModuleOp root) {
  if (llvm::Error error = validateStoredDerivedIdentifiers(root))
    return std::move(error);
  ArrayAttr storedSlots = root.getDomainSlotsAttr();
  ArrayAttr storedAssignments = root.getDomainAssignmentsAttr();
  auto relation = reconstructStoredFabricModuleDomain(root);
  if (!relation)
    return relation.takeError();
  auto labeling = computeCanonicalFabricModulePayloadLabeling(root, *relation);
  if (!labeling)
    return labeling.takeError();
  if (llvm::Error error = validateStoredGraphOrder(root, *labeling))
    return std::move(error);
  if (llvm::Error error =
          materializeFabricModuleDomainRelation(root, *relation, *labeling))
    return std::move(error);
  if (root.getDomainSlotsAttr() != storedSlots ||
      root.getDomainAssignmentsAttr() != storedAssignments)
    return invalid("canonical Module domain carrier is stale");
  return labeling;
}

} // namespace loom::fabric::detail
