#include "PnR/SpatialMappingSelectionProjection.h"

#include "Fabric/IR/PhysicalTag.h"
#include "Fabric/IR/UsePatternValue.h"
#include "PnR/SpatialCandidateState.h"
#include "PnR/SpatialTagContinuity.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::pnr {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::errc::invalid_argument,
      "invalid Spatial Mapping selection projection: %s",
      message.str().c_str());
}

bool rangeContains(PnrIndex offset, PnrIndex count, PnrIndex value) {
  return value >= offset && value - offset < count;
}

bool exactTagBitsEqual(const llvm::APInt &left, const llvm::APInt &right) {
  return left.getBitWidth() == right.getBitWidth() && left == right;
}

const ::loom::mapping::SpatialComputeBindingView *findComputeBinding(
    const ::loom::mapping::SpatialMappingView &mapping,
    std::uint64_t realization) {
  const ::loom::mapping::SpatialComputeBindingView *result = nullptr;
  for (const auto &binding : mapping.computeBindings())
    if (binding.realization == realization) {
      if (result)
        return nullptr;
      result = &binding;
    }
  return result;
}

const ::loom::mapping::SpatialMemoryEngineBindingView *findMemoryEngine(
    const ::loom::mapping::SpatialMappingView &mapping,
    std::uint64_t realization) {
  const ::loom::mapping::SpatialMemoryEngineBindingView *result = nullptr;
  for (const auto &binding : mapping.memoryEngineBindings())
    if (binding.realization == realization) {
      if (result)
        return nullptr;
      result = &binding;
    }
  return result;
}

const ::loom::mapping::SpatialMemoryBindingView *findMemoryBinding(
    const ::loom::mapping::SpatialMappingView &mapping,
    std::uint64_t entity) {
  const ::loom::mapping::SpatialMemoryBindingView *result = nullptr;
  for (const auto &binding : mapping.memoryBindings())
    if (binding.entityId == entity) {
      if (result)
        return nullptr;
      result = &binding;
    }
  return result;
}

const ::loom::mapping::SpatialRouteTreeView *findRouteTree(
    const ::loom::mapping::SpatialMappingView &mapping,
    const ::dataflow::CanonicalGraphProducerEndpointRef &producer,
    std::size_t *ordinal = nullptr) {
  const ::loom::mapping::SpatialRouteTreeView *result = nullptr;
  for (auto indexed : llvm::enumerate(mapping.routeTrees()))
    if (indexed.value().logicalNet == producer) {
      if (result)
        return nullptr;
      result = &indexed.value();
      if (ordinal)
        *ordinal = indexed.index();
    }
  return result;
}

const ::loom::mapping::SpatialRegisterFifoTransferView *findRegisterTransfer(
    const ::loom::mapping::SpatialMappingView &mapping,
    const ::dataflow::CanonicalGraphProducerEndpointRef &producer) {
  const ::loom::mapping::SpatialRegisterFifoTransferView *result = nullptr;
  for (const auto &transfer : mapping.registerFifoTransfers())
    if (transfer.logicalNet == producer) {
      if (result)
        return nullptr;
      result = &transfer;
    }
  return result;
}

llvm::Expected<const FrozenSpatialAttachmentOption *>
selectedAttachment(const SpatialCandidateState &candidate,
                   FrozenSpatialTerminalBinding binding) {
  const auto &problem = candidate.problem();
  PnrIndex selected = getInvalidPnrIndex();
  if (binding.kind == FrozenSpatialTerminalBindingKind::PortDemand) {
    if (binding.index >= problem.ports().portDemands().size() ||
        binding.index >= candidate.portAttachmentSelections().size())
      return invalid("terminal names an absent PortDemand");
    selected = candidate.portAttachment(binding.index);
  } else {
    if (binding.index >= problem.ports().graphBoundaries().size() ||
        binding.index >= candidate.graphBoundaryAttachmentSelections().size())
      return invalid("terminal names an absent graph boundary");
    selected = candidate.graphBoundaryAttachment(binding.index);
  }
  if (selected >= problem.ports().attachmentOptions().size())
    return invalid("terminal selects an absent attachment option");
  return &problem.ports().attachmentOptions()[selected];
}

llvm::Expected<bool>
compareComputeSelections(const ::loom::mapping::SpatialMappingView &mapping,
                         const SpatialCandidateState &candidate) {
  const auto &problem = candidate.problem();
  const auto realizations = problem.realizations().computeRealizations();
  const auto placements = problem.realizations().computePlacements();
  const auto contexts = problem.realizations().computeInstructionContexts();
  if (mapping.computeBindings().size() != realizations.size())
    return invalid("compute binding inventory is incomplete");
  for (PnrIndex ordinal = 0; ordinal < realizations.size(); ++ordinal) {
    const auto &realization = realizations[ordinal];
    const auto *binding = findComputeBinding(mapping, realization.reference.entity);
    if (!binding)
      return invalid("compute realization does not resolve uniquely");
    const auto &selection = candidate.computeBinding(ordinal);
    if (selection.placement >= placements.size() ||
        selection.instructionContext >= contexts.size())
      return invalid("candidate compute selection is out of range");
    const auto &placement = placements[selection.placement];
    if (placement.realization != ordinal ||
        !rangeContains(realization.placementOffset,
                       realization.placementCount, selection.placement) ||
        !rangeContains(placement.contextOffset, placement.contextCount,
                       selection.instructionContext))
      return invalid("candidate compute selection is outside its domain");
    if (binding->occurrence != placement.fu ||
        binding->context != contexts[selection.instructionContext] ||
        !binding->refinements.empty())
      return false;
  }
  return true;
}

bool memoryDispatchEqual(
    const ::loom::mapping::SpatialMemoryDispatchTargetView &mapping,
    const FrozenSpatialMemoryDispatchTarget &candidate) {
  return std::visit(
      [&](const auto &selected) {
        using Selected = std::decay_t<decltype(selected)>;
        if constexpr (std::is_same_v<
                          Selected,
                          ::loom::fabric::MemoryConsistencyDomainRef>) {
          return false;
        } else {
          if (const auto *value = std::get_if<Selected>(&mapping))
            return *value == selected;
          return false;
        }
      },
      candidate);
}

bool memoryConsistencyEqual(
    const ::loom::mapping::SpatialMemoryConsistencyTargetView &mapping,
    const FrozenSpatialMemoryDispatchTarget &candidate) {
  return std::visit(
      [&](const auto &selected) {
        using Selected = std::decay_t<decltype(selected)>;
        if constexpr (
            std::is_same_v<Selected,
                           ::loom::fabric::LocalMemoryServiceRef>) {
          return false;
        } else {
          if (const auto *value = std::get_if<Selected>(&mapping))
            return *value == selected;
          return false;
        }
      },
      candidate);
}

bool exposureDispatchEqual(
    const ::loom::mapping::SpatialMemoryDispatchTargetView &mapping,
    const FrozenSpatialMemoryExposureDispatchTarget &candidate) {
  return std::visit(
      [&](const auto &selected) {
        using Selected = std::decay_t<decltype(selected)>;
        if (const auto *value = std::get_if<Selected>(&mapping))
          return *value == selected;
        return false;
      },
      candidate);
}

bool memoryBindingTargetEqual(
    const ::loom::mapping::SpatialMemoryBindingTargetView &mapping,
    const FrozenSpatialMemoryBindingTargetOption &target,
    const SpatialLogicalMemoryBindingSelection &selection) {
  if (const auto *region =
          std::get_if<::loom::fabric::FabricMemoryServiceRegionRef>(
              &target.target)) {
    const auto *local =
        std::get_if<::loom::mapping::SpatialMemoryLocalRegionView>(&mapping);
    return local && local->serviceRegion == *region &&
           local->physicalOffsetBytes == selection.physicalOffsetBytes;
  }
  return selection.physicalOffsetBytes == 0 &&
         std::holds_alternative<
             ::loom::mapping::SpatialMemoryBoundaryProxyView>(mapping);
}

const ::loom::mapping::SpatialExposureEntryView *findExposure(
    const ::loom::mapping::SpatialMemoryBindingView &binding,
    const ::dataflow::MemoryExposureRef &exposure) {
  const ::loom::mapping::SpatialExposureEntryView *result = nullptr;
  for (const auto &entry : binding.exposures)
    if (entry.exposure == exposure) {
      if (result)
        return nullptr;
      result = &entry;
    }
  return result;
}

llvm::Expected<bool>
compareLogicalMemoryBindings(const ::loom::mapping::SpatialMappingView &mapping,
                             const SpatialCandidateState &candidate) {
  const auto &memory = candidate.problem().memory();
  if (mapping.memoryBindings().size() != memory.logicalBindings().size())
    return invalid("logical memory binding inventory is incomplete");
  for (PnrIndex ordinal = 0; ordinal < memory.logicalBindings().size();
       ++ordinal) {
    const auto *binding = findMemoryBinding(mapping, ordinal);
    if (!binding)
      return invalid("logical memory binding does not resolve uniquely");
    const auto &logical = memory.logicalBindings()[ordinal];
    const auto &selection = candidate.logicalMemoryBinding(ordinal);
    if (selection.target >= memory.bindingTargets().size())
      return invalid("candidate logical memory target is out of range");
    if (binding->logicalMemory != logical.logicalMemory ||
        !std::holds_alternative<
            ::loom::mapping::SpatialMemoryWholeIntervalView>(
            binding->interval) ||
        !memoryBindingTargetEqual(binding->target,
                                  memory.bindingTargets()[selection.target],
                                  selection))
      return false;

    if (ordinal + 1 >= memory.bindingExposureOffsets().size())
      return invalid("logical memory exposure offsets are incomplete");
    const PnrIndex begin = memory.bindingExposureOffsets()[ordinal];
    const PnrIndex end = memory.bindingExposureOffsets()[ordinal + 1];
    if (begin > end || end > memory.bindingExposures().size() ||
        binding->exposures.size() != end - begin)
      return invalid("logical memory exposure inventory is malformed");
    for (PnrIndex incidence = begin; incidence < end; ++incidence) {
      const PnrIndex exposureOrdinal = memory.bindingExposures()[incidence];
      if (exposureOrdinal >= memory.exposures().size())
        return invalid("logical memory binding names an absent exposure");
      const auto &exposure = memory.exposures()[exposureOrdinal];
      if (exposure.logicalBinding != ordinal)
        return invalid("logical memory exposure owner is inconsistent");
      const auto *entry = findExposure(*binding, exposure.exposure);
      if (!entry)
        return invalid("memory exposure does not resolve uniquely");
      const PnrIndex optionOrdinal =
          candidate.memoryExposureSelection(exposureOrdinal);
      if (optionOrdinal >= memory.exposureOptions().size())
        return invalid("candidate memory exposure option is out of range");
      const auto &option = memory.exposureOptions()[optionOrdinal];
      if (option.provider >= memory.exposureProviders().size())
        return invalid("candidate memory exposure provider is out of range");
      if (entry->terminal !=
              memory.exposureProviders()[option.provider].terminal ||
          !exposureDispatchEqual(entry->dispatch, option.target))
        return false;
    }
  }
  return true;
}

bool activityEventEqual(
    const ::loom::mapping::SpatialActivityEventRef &left,
    const ::loom::mapping::SpatialActivityEventRef &right) {
  return left == right;
}

bool resourceOwnerEqual(
    const ::loom::mapping::SpatialResourceOwnerRef &left,
    const ::loom::mapping::SpatialResourceOwnerRef &right) {
  if (left.index() != right.index())
    return false;
  switch (left.index()) {
  case 0:
    return std::get<::loom::mapping::SpatialComputeResourceOwnerRef>(left)
               .realization ==
           std::get<::loom::mapping::SpatialComputeResourceOwnerRef>(right)
               .realization;
  case 1:
    return std::get<::loom::mapping::SpatialMemoryEngineResourceOwnerRef>(left)
               .realization ==
           std::get<::loom::mapping::SpatialMemoryEngineResourceOwnerRef>(
               right)
               .realization;
  case 2:
    return std::get<::loom::mapping::SpatialMemoryBindingResourceOwnerRef>(left)
               .binding ==
           std::get<::loom::mapping::SpatialMemoryBindingResourceOwnerRef>(
               right)
               .binding;
  case 3: {
    const auto &leftRoute =
        std::get<::loom::mapping::SpatialRouteNodeResourceOwnerRef>(left);
    const auto &rightRoute =
        std::get<::loom::mapping::SpatialRouteNodeResourceOwnerRef>(right);
    return leftRoute.logicalNet == rightRoute.logicalNet &&
           leftRoute.nodeOrdinal == rightRoute.nodeOrdinal;
  }
  default:
    return false;
  }
}

llvm::Expected<const ::loom::mapping::SpatialResourceUseView *>
findMemoryResourceUse(
    const ::loom::mapping::SpatialMappingView &mapping,
    const ::loom::mapping::SpatialResourceOwnerRef &owner,
    const ::loom::mapping::SpatialActivityEventRef &trigger) {
  const ::loom::mapping::SpatialResourceUseView *result = nullptr;
  for (const auto &use : mapping.resourceUses()) {
    if (!resourceOwnerEqual(use.owner, owner) ||
        !activityEventEqual(use.activation.trigger.event, trigger))
      continue;
    if (result)
      return invalid("memory ResourceUse does not resolve uniquely");
    result = &use;
  }
  if (!result)
    return invalid("memory ResourceUse is absent");
  if (result->activation.trigger.guaranteedOffset ||
      !result->activation.release.empty() || !result->parameters.empty() ||
      !result->sharingAssignments.empty())
    return invalid("memory ResourceUse has non-intrinsic fields");
  return result;
}

llvm::Expected<const FrozenSpatialMemoryDispatchDomain *>
memoryDispatchDomain(const SpatialCandidateState &candidate,
                     PnrIndex useOrdinal) {
  const auto &problem = candidate.problem();
  const auto &memory = problem.memory();
  const auto &realizations = problem.realizations();
  if (useOrdinal >= memory.rootedUses().size())
    return invalid("rooted memory use is out of range");
  const auto &use = memory.rootedUses()[useOrdinal];
  if (use.actor >= realizations.memoryActors().size() ||
      use.actor >= realizations.memoryActorRealizations().size())
    return invalid("rooted memory use has an absent actor");
  const PnrIndex owner = realizations.memoryActorRealizations()[use.actor];
  if (owner >= realizations.memoryRealizations().size())
    return invalid("rooted memory use has an absent realization");
  const auto &realization = realizations.memoryRealizations()[owner];
  const PnrIndex placement = candidate.memoryBinding(owner).placement;
  const auto offsets = memory.memoryPlacementDomainOffsets();
  if (placement >= realizations.memoryPlacements().size() ||
      placement + 1 >= offsets.size() || use.actor < realization.actorOffset)
    return invalid("rooted memory use has no selected placement domain");
  const PnrIndex localActor = use.actor - realization.actorOffset;
  if (localActor >= realization.actorCount)
    return invalid("rooted memory use is outside its realization");
  const PnrIndex domainOrdinal = offsets[placement] + localActor;
  if (domainOrdinal >= offsets[placement + 1] ||
      domainOrdinal >= memory.dispatchDomains().size())
    return invalid("rooted memory use has no dispatch domain");
  const auto &domain = memory.dispatchDomains()[domainOrdinal];
  if (domain.placement != placement || domain.actor != use.actor)
    return invalid("rooted memory dispatch domain is inconsistent");
  return &domain;
}

llvm::Expected<const FrozenSpatialMemoryOperationHandshakePlan *>
memoryOperationPlan(const SpatialCandidateState &candidate, PnrIndex actor,
                    PnrIndex placement) {
  const auto &problem = candidate.problem();
  const auto &realizations = problem.realizations();
  if (actor >= realizations.memoryActors().size() ||
      actor >= realizations.memoryActorRealizations().size())
    return invalid("memory operation actor is out of range");
  const PnrIndex owner = realizations.memoryActorRealizations()[actor];
  if (owner >= realizations.memoryRealizations().size())
    return invalid("memory operation owner is out of range");
  const auto &realization = realizations.memoryRealizations()[owner];
  const auto offsets = problem.handshake().memoryPlacementDomainOffsets();
  if (placement + 1 >= offsets.size() || actor < realization.actorOffset)
    return invalid("memory operation has no selected placement domain");
  const PnrIndex localActor = actor - realization.actorOffset;
  if (localActor >= realization.actorCount)
    return invalid("memory operation is outside its realization");
  const PnrIndex domainOrdinal = offsets[placement] + localActor;
  if (domainOrdinal >= offsets[placement + 1] ||
      domainOrdinal >= problem.handshake().memoryOperationDomains().size())
    return invalid("memory operation plan domain is absent");
  const auto &domain =
      problem.handshake().memoryOperationDomains()[domainOrdinal];
  const PnrIndex selected = candidate.memoryOperationPlan(actor);
  if (domain.placement != placement || domain.actor != actor ||
      !rangeContains(domain.planOffset, domain.planCount, selected) ||
      selected >= problem.handshake().memoryOperationPlans().size())
    return invalid("memory operation plan is outside its domain");
  return &problem.handshake().memoryOperationPlans()[selected];
}

llvm::Expected<::loom::mapping::SpatialActivityEventRef>
memoryPlanTrigger(const SpatialCandidateState &candidate, PnrIndex plan) {
  const auto &problem = candidate.problem();
  const auto envelopes = problem.capacity().memoryOperationPlanEnvelopes();
  const auto timeEnvelopes = problem.capacity().resourceTimeEnvelopes();
  const auto events = problem.capacity().resourceEvents();
  if (plan >= envelopes.size() || envelopes[plan] >= timeEnvelopes.size())
    return invalid("memory operation plan has no resource envelope");
  const PnrIndex event = timeEnvelopes[envelopes[plan]].event;
  if (event >= events.size())
    return invalid("memory operation plan has no resource event");
  return events[event].reference;
}

struct MemoryOccurrenceCursor final {
  ::loom::fabric::FabricMemoryOccurrenceRef occurrence;
  std::uint64_t next = 0;
};

std::uint64_t &residentCursor(
    std::vector<MemoryOccurrenceCursor> &cursors,
    const ::loom::fabric::FabricMemoryOccurrenceRef &occurrence) {
  for (auto &cursor : cursors)
    if (cursor.occurrence == occurrence)
      return cursor.next;
  cursors.push_back({occurrence, 0});
  return cursors.back().next;
}

const ::loom::mapping::SpatialMemoryOperationView *findMemoryOperation(
    const ::loom::mapping::SpatialMemoryEngineBindingView &engine,
    const ::dataflow::ActorRef &actor) {
  const ::loom::mapping::SpatialMemoryOperationView *result = nullptr;
  for (const auto &operation : engine.operations) {
    const ::dataflow::ActorRef &candidate = std::visit(
        [](const auto &record) -> const ::dataflow::ActorRef & {
          return record.actor;
        },
        operation);
    if (candidate == actor) {
      if (result)
        return nullptr;
      result = &operation;
    }
  }
  return result;
}

bool memoryPlacementEqual(
    const ::loom::mapping::SpatialMemoryOperationPlacementView &mapping,
    const ::loom::fabric::FabricMemoryOperationPortRef &port,
    bool temporalResident, std::uint64_t contextOrdinal) {
  if (!temporalResident) {
    const auto *selected =
        std::get_if<::loom::fabric::FabricMemoryOperationPortRef>(&mapping);
    return selected && *selected == port;
  }
  const auto *selected =
      std::get_if<::loom::fabric::FabricMemoryOperationContextRef>(&mapping);
  return selected && selected->port == port &&
         selected->ordinal == contextOrdinal;
}

llvm::Expected<bool> compareMemoryOperationUses(
    const ::loom::mapping::SpatialMemoryOperationView &operation,
    PnrIndex actorOrdinal, const SpatialCandidateState &candidate) {
  const auto &memory = candidate.problem().memory();
  if (actorOrdinal + 1 >= memory.actorUseOffsets().size())
    return invalid("memory actor rooted-use offsets are incomplete");
  const PnrIndex begin = memory.actorUseOffsets()[actorOrdinal];
  const PnrIndex end = memory.actorUseOffsets()[actorOrdinal + 1];
  if (begin >= end || end > memory.rootedUses().size())
    return invalid("memory actor rooted-use inventory is incomplete");
  const bool addressed = memory.rootedUses()[begin].logicalBinding.has_value();
  if (addressed != std::holds_alternative<
                         ::loom::mapping::SpatialAddressedMemoryOperationView>(
                         operation))
    return false;

  if (addressed) {
    const auto &mapped =
        std::get<::loom::mapping::SpatialAddressedMemoryOperationView>(
            operation);
    if (mapped.uses.size() != end - begin)
      return invalid("addressed memory-use inventory is incomplete");
    for (PnrIndex useOrdinal = begin; useOrdinal < end; ++useOrdinal) {
      const auto &use = memory.rootedUses()[useOrdinal];
      if (!use.logicalBinding)
        return invalid("one memory actor mixes addressed and fence uses");
      auto domain = memoryDispatchDomain(candidate, useOrdinal);
      if (!domain)
        return domain.takeError();
      const PnrIndex selected = candidate.memoryUseDispatch(useOrdinal);
      if (!rangeContains((*domain)->optionOffset, (*domain)->optionCount,
                         selected) ||
          selected >= memory.dispatchOptions().size())
        return invalid("candidate addressed dispatch is outside its domain");
      const auto &option = memory.dispatchOptions()[selected];
      const auto found = llvm::find_if(mapped.uses, [&](const auto &entry) {
        return entry.launch == use.launch &&
               entry.binding == *use.logicalBinding;
      });
      if (found == mapped.uses.end())
        return false;
      if (!memoryDispatchEqual(found->dispatch, option.target))
        return false;
    }
    return true;
  }

  const auto &mapped =
      std::get<::loom::mapping::SpatialFenceMemoryOperationView>(operation);
  if (mapped.uses.size() != end - begin)
    return invalid("fence memory-use inventory is incomplete");
  for (PnrIndex useOrdinal = begin; useOrdinal < end; ++useOrdinal) {
    const auto &use = memory.rootedUses()[useOrdinal];
    if (use.logicalBinding)
      return invalid("one memory actor mixes addressed and fence uses");
    auto domain = memoryDispatchDomain(candidate, useOrdinal);
    if (!domain)
      return domain.takeError();
    const PnrIndex selected = candidate.memoryUseDispatch(useOrdinal);
    if (!rangeContains((*domain)->optionOffset, (*domain)->optionCount,
                       selected) ||
        selected >= memory.dispatchOptions().size())
      return invalid("candidate fence dispatch is outside its domain");
    const auto &option = memory.dispatchOptions()[selected];
    const auto found = llvm::find_if(mapped.uses, [&](const auto &entry) {
      return entry.launch == use.launch;
    });
    if (found == mapped.uses.end())
      return false;
    if (!memoryConsistencyEqual(found->consistency, option.target))
      return false;
  }
  return true;
}

llvm::Expected<bool> compareMemorySelections(
    const ::loom::mapping::SpatialMappingView &mapping,
    const SpatialCandidateState &candidate) {
  auto logical = compareLogicalMemoryBindings(mapping, candidate);
  if (!logical || !*logical)
    return logical;

  const auto &problem = candidate.problem();
  const auto &realizations = problem.realizations();
  const auto placements = realizations.memoryPlacements();
  const auto actors = realizations.memoryActors();
  const auto patterns = problem.resources().usePatterns();
  if (mapping.memoryEngineBindings().size() !=
      realizations.memoryRealizations().size())
    return invalid("memory engine binding inventory is incomplete");
  std::vector<MemoryOccurrenceCursor> residentContexts;
  for (PnrIndex realizationOrdinal = 0;
       realizationOrdinal < realizations.memoryRealizations().size();
       ++realizationOrdinal) {
    const auto &realization =
        realizations.memoryRealizations()[realizationOrdinal];
    const auto *engine = findMemoryEngine(mapping, realization.reference.entity);
    if (!engine)
      return invalid("memory realization does not resolve uniquely");
    const PnrIndex placementOrdinal =
        candidate.memoryBinding(realizationOrdinal).placement;
    if (!rangeContains(realization.placementOffset,
                       realization.placementCount, placementOrdinal) ||
        placementOrdinal >= placements.size())
      return invalid("candidate memory placement is outside its domain");
    const auto &placement = placements[placementOrdinal];
    if (placement.realization != realizationOrdinal)
      return invalid("candidate memory placement has a foreign owner");
    if (engine->occurrence != placement.memory ||
        engine->operations.size() != realization.actorCount)
      return false;

    for (PnrIndex localActor = 0; localActor < realization.actorCount;
         ++localActor) {
      const PnrIndex actorOrdinal = realization.actorOffset + localActor;
      if (actorOrdinal >= actors.size())
        return invalid("memory realization actor slice is out of range");
      const auto &actor = actors[actorOrdinal];
      const auto *operation = findMemoryOperation(*engine, actor.actor);
      if (!operation)
        return invalid("memory operation does not resolve uniquely");
      auto plan = memoryOperationPlan(candidate, actorOrdinal, placementOrdinal);
      if (!plan)
        return plan.takeError();
      const PnrIndex selectedPlan = candidate.memoryOperationPlan(actorOrdinal);
      if ((*plan)->usePattern >= patterns.size())
        return invalid("memory operation plan has an absent UsePattern");
      auto trigger = memoryPlanTrigger(candidate, selectedPlan);
      if (!trigger)
        return trigger.takeError();
      auto use = findMemoryResourceUse(
          mapping,
          ::loom::mapping::SpatialResourceOwnerRef(
              ::loom::mapping::SpatialMemoryEngineResourceOwnerRef{
                  realization.reference.entity}),
          *trigger);
      if (!use)
        return use.takeError();
      if ((*use)->useSite != patterns[(*plan)->usePattern].reference)
        return false;

      const ::loom::fabric::FabricMemoryOperationPortRef port{
          placement.memory, actor.operationPort.ordinal};
      std::uint64_t contextOrdinal = 0;
      if ((*plan)->temporalResident) {
        contextOrdinal = residentCursor(residentContexts, placement.memory)++;
        if (!placement.residentContextCount ||
            contextOrdinal >= *placement.residentContextCount)
          return invalid("Temporal memory context capacity is exceeded");
      }
      const auto &mappedPlacement = std::visit(
          [](const auto &record)
              -> const ::loom::mapping::SpatialMemoryOperationPlacementView & {
            return record.placement;
          },
          *operation);
      if (!memoryPlacementEqual(mappedPlacement, port,
                                (*plan)->temporalResident, contextOrdinal))
        return false;
      auto uses = compareMemoryOperationUses(*operation, actorOrdinal, candidate);
      if (!uses || !*uses)
        return uses;
    }
  }

  // A local-service dispatch selects a ResourceContract UsePattern in
  // addition to its operation-use target. That choice is independent when a
  // target admits multiple patterns, so it is compared explicitly here.
  const auto &memory = problem.memory();
  for (PnrIndex useOrdinal = 0; useOrdinal < memory.rootedUses().size();
       ++useOrdinal) {
    const PnrIndex selected = candidate.memoryUseDispatch(useOrdinal);
    if (selected >= memory.dispatchOptions().size())
      return invalid("candidate memory dispatch is out of range");
    const auto &option = memory.dispatchOptions()[selected];
    if (!option.serviceUsePattern)
      continue;
    const auto &rooted = memory.rootedUses()[useOrdinal];
    if (!rooted.logicalBinding ||
        *rooted.logicalBinding >= memory.logicalBindings().size())
      return invalid("local-service dispatch has no logical binding");
    const PnrIndex planOrdinal = candidate.memoryOperationPlan(rooted.actor);
    auto trigger = memoryPlanTrigger(candidate, planOrdinal);
    if (!trigger)
      return trigger.takeError();
    auto resourceUse = findMemoryResourceUse(
        mapping,
        ::loom::mapping::SpatialResourceOwnerRef(
            ::loom::mapping::SpatialMemoryBindingResourceOwnerRef{
                *rooted.logicalBinding}),
        *trigger);
    if (!resourceUse)
      return resourceUse.takeError();
    if ((*resourceUse)->useSite != *option.serviceUsePattern)
      return false;
  }
  return true;
}

llvm::Expected<const ::loom::mapping::SpatialRouteNodeView *>
mappingNodeByOrdinal(const ::loom::mapping::SpatialRouteTreeView &route,
                     std::uint64_t ordinal) {
  const ::loom::mapping::SpatialRouteNodeView *result = nullptr;
  for (const auto &node : route.nodes)
    if (node.ordinal == ordinal) {
      if (result)
        return invalid("RouteTree repeats a node ordinal");
      result = &node;
    }
  if (!result)
    return invalid("RouteTree node ordinal is absent");
  return result;
}

const ::loom::mapping::SpatialRouteNodeView *mappingNodeByEndpoint(
    const ::loom::mapping::SpatialRouteTreeView &route,
    const ::loom::fabric::FabricTransportEndpointRef &endpoint) {
  const ::loom::mapping::SpatialRouteNodeView *result = nullptr;
  for (const auto &node : route.nodes)
    if (node.endpoint == endpoint) {
      if (result)
        return nullptr;
      result = &node;
    }
  return result;
}

llvm::Expected<bool> compareRouteNodes(
    const ::loom::mapping::SpatialRouteTreeView &mapping,
    const RouteTreeState &candidate) {
  const auto &routing = candidate.routingGraph();
  const auto endpoints = routing.routingEndpoints();
  const auto arcs = routing.routingArcs();
  const auto arcSources = routing.arcSources();
  const auto traversals = routing.traversals();
  if (mapping.nodes.size() != candidate.activeNodeCount())
    return false;
  for (const auto &node : mapping.nodes)
    if (!node.refinements.empty())
      return false;

  for (const auto &node : candidate.nodeStorage()) {
    if (!node.isActive())
      continue;
    if (node.endpoint >= endpoints.size())
      return invalid("candidate RouteTree node endpoint is out of range");
    const auto *mapped =
        mappingNodeByEndpoint(mapping, endpoints[node.endpoint].reference);
    if (!mapped)
      return false;
    if (node.parentArc == getInvalidPnrIndex()) {
      if (mapped->parentOrdinal || mapped->incomingTraversal)
        return false;
      continue;
    }
    if (node.parentArc >= arcs.size() || node.parentArc >= arcSources.size())
      return invalid("candidate RouteTree node arc is out of range");
    const auto &arc = arcs[node.parentArc];
    if (arc.target != node.endpoint || arc.traversal >= traversals.size() ||
        arcSources[node.parentArc] >= endpoints.size())
      return invalid("candidate RouteTree node arc is inconsistent");
    if (!mapped->parentOrdinal || !mapped->incomingTraversal ||
        *mapped->incomingTraversal != traversals[arc.traversal].reference)
      return false;
    auto mappedParent = mappingNodeByOrdinal(mapping, *mapped->parentOrdinal);
    if (!mappedParent)
      return mappedParent.takeError();
    if ((*mappedParent)->endpoint !=
        endpoints[arcSources[node.parentArc]].reference)
      return false;
  }
  return true;
}

const ::loom::mapping::SpatialRouteSinkView *findRouteSink(
    const ::loom::mapping::SpatialRouteTreeView &route,
    const ::dataflow::CanonicalGraphConsumerEndpointRef &sink) {
  const ::loom::mapping::SpatialRouteSinkView *result = nullptr;
  for (const auto &entry : route.sinks)
    if (entry.sink == sink) {
      if (result)
        return nullptr;
      result = &entry;
    }
  return result;
}

llvm::Expected<bool> compareRouteTerminals(
    const ::loom::mapping::SpatialRouteTreeView &mapping,
    const SpatialCandidateState &candidate, PnrIndex logicalNet,
    const RouteTreeState &route) {
  const auto &problem = candidate.problem();
  const auto &net = problem.transfers().logicalNets()[logicalNet];
  auto sourceAttachment = selectedAttachment(
      candidate,
      problem.transfers().logicalNetSourceBindings()[logicalNet]);
  if (!sourceAttachment)
    return sourceAttachment.takeError();
  const auto source = route.sourceEndpoint();
  if (!source || *source != (*sourceAttachment)->endpoint ||
      mapping.rootEndpoint != problem.routing()
                                  .routingEndpoints()[(*sourceAttachment)->endpoint]
                                  .reference ||
      mapping.localTraversal !=
          ((*sourceAttachment)->localTraversal
               ? std::optional<::loom::fabric::FabricPhysicalTraversalRef>(
                     problem.routing()
                         .traversals()[*(*sourceAttachment)->localTraversal]
                         .reference)
               : std::nullopt))
    return false;
  if (mapping.sinks.size() != net.sinkCount)
    return false;
  for (PnrIndex localSink = 0; localSink < net.sinkCount; ++localSink) {
    const PnrIndex sinkOrdinal = net.sinkOffset + localSink;
    if (sinkOrdinal >= problem.transfers().logicalNetSinks().size() ||
        sinkOrdinal >= problem.transfers().logicalNetSinkBindings().size())
      return invalid("logical-net sink slice is out of range");
    auto attachment = selectedAttachment(
        candidate,
        problem.transfers().logicalNetSinkBindings()[sinkOrdinal]);
    if (!attachment)
      return attachment.takeError();
    const auto endpoint = route.sinkEndpoint(localSink);
    if (!endpoint || *endpoint != (*attachment)->endpoint)
      return false;
    const auto *mapped = findRouteSink(
        mapping, problem.transfers().logicalNetSinks()[sinkOrdinal]);
    if (!mapped)
      return invalid("RouteTree sink does not resolve uniquely");
    auto mappedNode = mappingNodeByOrdinal(mapping, mapped->nodeOrdinal);
    if (!mappedNode)
      return mappedNode.takeError();
    if ((*mappedNode)->endpoint != problem.routing()
                                      .routingEndpoints()[*endpoint]
                                      .reference)
      return false;
    const auto expectedLocal =
        (*attachment)->localTraversal
            ? std::optional<::loom::fabric::FabricPhysicalTraversalRef>(
                  problem.routing()
                      .traversals()[*(*attachment)->localTraversal]
                      .reference)
            : std::nullopt;
    if (mapped->localTraversal != expectedLocal)
      return false;
  }
  return true;
}

llvm::Expected<bool> sameSegmentNodes(
    const ::loom::mapping::SpatialRouteTreeView &mappingRoute,
    const ::loom::mapping::SpatialPhysicalTagSegmentView &mappingSegment,
    const RouteTreeState &candidateRoute,
    llvm::ArrayRef<PnrIndex> nodeSegments, PnrIndex segment) {
  std::vector<::loom::fabric::FabricTransportEndpointRef> mappedEndpoints;
  mappedEndpoints.reserve(mappingSegment.nodeOrdinals.size());
  for (std::uint64_t ordinal : mappingSegment.nodeOrdinals) {
    auto node = mappingNodeByOrdinal(mappingRoute, ordinal);
    if (!node)
      return node.takeError();
    if (llvm::is_contained(mappedEndpoints, (*node)->endpoint))
      return invalid("Physical Tag segment repeats a Mapping endpoint");
    mappedEndpoints.push_back((*node)->endpoint);
  }
  std::vector<::loom::fabric::FabricTransportEndpointRef> candidateEndpoints;
  const auto endpoints = candidateRoute.routingGraph().routingEndpoints();
  const auto nodes = candidateRoute.nodeStorage();
  if (nodeSegments.size() != nodes.size())
    return invalid("Physical Tag node projection has the wrong size");
  for (auto indexed : llvm::enumerate(nodes)) {
    if (!indexed.value().isActive() ||
        nodeSegments[indexed.index()] != segment)
      continue;
    if (indexed.value().endpoint >= endpoints.size())
      return invalid("Physical Tag segment endpoint is out of range");
    candidateEndpoints.push_back(endpoints[indexed.value().endpoint].reference);
  }
  if (mappedEndpoints.size() != candidateEndpoints.size())
    return false;
  for (const auto &endpoint : mappedEndpoints)
    if (!llvm::is_contained(candidateEndpoints, endpoint))
      return false;
  return true;
}

llvm::Expected<llvm::APInt> mappingSegmentTag(
    const ::loom::mapping::SpatialMappingView &mapping,
    const ::loom::mapping::SpatialPhysicalTagSegmentView &segment) {
  if (segment.resourceUseOrdinal >= mapping.resourceUses().size())
    return invalid("Physical Tag segment ResourceUse is out of range");
  const auto &use = mapping.resourceUses()[segment.resourceUseOrdinal];
  if (use.parameters.size() != 0 || use.sharingAssignments.size() != 1)
    return invalid("Physical Tag ResourceUse has the wrong value shape");
  const auto *tag = std::get_if<::fabric::PhysicalTagPatternValue>(
      &use.sharingAssignments.front());
  if (!tag)
    return invalid("Physical Tag ResourceUse has a non-tag value");
  return tag->value;
}

llvm::Expected<bool> compareRouteTags(
    const ::loom::mapping::SpatialMappingView &mapping,
    std::size_t mappingRouteOrdinal,
    const ::loom::mapping::SpatialRouteTreeView &mappingRoute,
    const RouteTreeState &candidateRoute,
    llvm::ArrayRef<std::optional<llvm::APInt>> candidateValues) {
  SpatialTagContinuityProjection continuity;
  SpatialTagContinuityScratch continuityScratch;
  if (llvm::Error error = detail::rebuildSpatialTagContinuityUnchecked(
          candidateRoute, continuity, continuityScratch))
    return std::move(error);
  if (candidateValues.size() != continuity.segments().size())
    return invalid("provisional Physical Tag value inventory is incomplete");
  std::size_t mappingSegmentCount = 0;
  for (const auto &segment : mapping.physicalTagSegments())
    if (segment.routeTreeOrdinal == mappingRouteOrdinal)
      ++mappingSegmentCount;
  if (mappingSegmentCount != continuity.segments().size())
    return false;
  for (PnrIndex ordinal = 0; ordinal < continuity.segments().size();
       ++ordinal) {
    const ::loom::mapping::SpatialPhysicalTagSegmentView *mapped = nullptr;
    for (const auto &segment : mapping.physicalTagSegments())
      if (segment.routeTreeOrdinal == mappingRouteOrdinal &&
          segment.segmentOrdinal == ordinal) {
        if (mapped)
          return invalid("Physical Tag segment ordinal is repeated");
        mapped = &segment;
      }
    if (!mapped)
      return invalid("Physical Tag segment ordinal is absent");
    // TagUnassigned is a supported temporary Candidate state. It cannot equal
    // a sealed Mapping, but it is not a malformed projection and must not turn
    // an ordinary reroute/recolor probe into an invocation error.
    if (!candidateValues[ordinal])
      return false;
    const auto &descriptor = continuity.segments()[ordinal];
    if (!::fabric::isRepresentablePhysicalTagValue(
            descriptor.tagWidthBits, *candidateValues[ordinal]))
      return invalid("provisional Physical Tag value is not representable");
    auto mappedTag = mappingSegmentTag(mapping, *mapped);
    if (!mappedTag)
      return mappedTag.takeError();
    const llvm::APInt expected =
        candidateValues[ordinal]->zextOrTrunc(descriptor.tagWidthBits);
    if (!exactTagBitsEqual(*mappedTag, expected))
      return false;
    auto sameNodes = sameSegmentNodes(mappingRoute, *mapped, candidateRoute,
                                      continuity.nodeSegments(), ordinal);
    if (!sameNodes || !*sameNodes)
      return sameNodes;
  }
  return true;
}

llvm::Expected<bool> compareTransportSelections(
    const ::loom::mapping::SpatialMappingView &mapping,
    const SpatialCandidateState &candidate,
    llvm::ArrayRef<const RouteTreeState *> provisionalRoutes,
    llvm::ArrayRef<llvm::ArrayRef<std::optional<llvm::APInt>>>
        provisionalTagValues) {
  const auto &problem = candidate.problem();
  const auto nets = problem.transfers().logicalNets();
  if (provisionalRoutes.size() != nets.size() ||
      provisionalTagValues.size() != nets.size())
    return invalid("provisional route/tag inventory is incomplete");
  std::size_t expectedRoutes = 0;
  std::size_t expectedRegisterTransfers = 0;
  for (PnrIndex logicalNet = 0; logicalNet < nets.size(); ++logicalNet) {
    const RouteTreeState *route = provisionalRoutes[logicalNet];
    if (!route || &route->routingGraph() != &problem.routing())
      return invalid("provisional RouteTree has a foreign routing graph");
    const PnrIndex local = candidate.registerFifoTransfer(logicalNet);
    if (local != getInvalidPnrIndex()) {
      ++expectedRegisterTransfers;
      if (!route->isUnrouted() || !provisionalTagValues[logicalNet].empty())
        return invalid(
            "register-FIFO disposition retains an external route or tag");
      if (logicalNet >= problem.localTransfers().domains().size())
        return invalid("register-FIFO disposition has no domain");
      const auto &domain = problem.localTransfers().domains()[logicalNet];
      if (!rangeContains(domain.optionOffset, domain.optionCount, local) ||
          local >= problem.localTransfers().options().size())
        return invalid("register-FIFO option is outside its net domain");
      const auto &option = problem.localTransfers().options()[local];
      const auto *mapped = findRegisterTransfer(mapping, nets[logicalNet].producer);
      if (!mapped)
        return false;
      if (option.logicalNet != logicalNet || nets[logicalNet].sinkCount != 1 ||
          nets[logicalNet].sinkOffset >=
              problem.transfers().logicalNetSinks().size() ||
          option.writeTraversal >= problem.routing().traversals().size() ||
          option.readTraversal >= problem.routing().traversals().size())
        return invalid("register-FIFO option is malformed");
      if (mapped->sink != problem.transfers().logicalNetSinks()[
                              nets[logicalNet].sinkOffset] ||
          mapped->pe != option.pe ||
          mapped->registerFifo != option.registerFifo ||
          mapped->writeTraversal !=
              problem.routing().traversals()[option.writeTraversal].reference ||
          mapped->readTraversal !=
              problem.routing().traversals()[option.readTraversal].reference ||
          !exactTagBitsEqual(mapped->tag, option.tag))
        return false;
      if (findRouteTree(mapping, nets[logicalNet].producer))
        return false;
      continue;
    }

    ++expectedRoutes;
    if (route->isUnrouted())
      return false;
    std::size_t mappingRouteOrdinal = 0;
    const auto *mapped = findRouteTree(mapping, nets[logicalNet].producer,
                                       &mappingRouteOrdinal);
    if (!mapped || findRegisterTransfer(mapping, nets[logicalNet].producer))
      return false;
    auto terminals = compareRouteTerminals(*mapped, candidate, logicalNet,
                                           *route);
    if (!terminals || !*terminals)
      return terminals;
    auto nodes = compareRouteNodes(*mapped, *route);
    if (!nodes || !*nodes)
      return nodes;
    auto tags = compareRouteTags(mapping, mappingRouteOrdinal, *mapped, *route,
                                 provisionalTagValues[logicalNet]);
    if (!tags || !*tags)
      return tags;
  }
  if (mapping.routeTrees().size() != expectedRoutes ||
      mapping.registerFifoTransfers().size() != expectedRegisterTransfers)
    return invalid("transport disposition inventory is incomplete");
  return true;
}

llvm::Error requireRepresentableResourceUseValues(
    const ::loom::mapping::SpatialMappingView &mapping) {
  std::vector<std::uint8_t> physicalTagUses(mapping.resourceUses().size(), 0);
  for (const auto &segment : mapping.physicalTagSegments()) {
    if (segment.resourceUseOrdinal >= physicalTagUses.size())
      return invalid("Physical Tag segment ResourceUse is out of range");
    if (physicalTagUses[segment.resourceUseOrdinal])
      return invalid("Physical Tag segments repeat one ResourceUse");
    physicalTagUses[segment.resourceUseOrdinal] = 1;
  }
  for (auto indexed : llvm::enumerate(mapping.resourceUses())) {
    if (physicalTagUses[indexed.index()])
      continue;
    if (!std::holds_alternative<
            ::loom::mapping::SpatialComputeResourceOwnerRef>(
            indexed.value().owner))
      continue;
    // SpatialMappingMaterializer derives compute-use ownership, activation,
    // release, and UsePattern from the selected compute bindings and local
    // dispositions, and its Candidate domain has no parameter or sharing
    // decision for these uses. A strict-imported authored Mapping may still
    // carry pattern-valid values, so fail closed instead of treating such a
    // parent as equal to an unrepresentable Candidate.
    if (!indexed.value().parameters.empty() ||
        !indexed.value().sharingAssignments.empty())
      return invalid(
          "compute ResourceUse carries values outside the Candidate domain");
  }
  return llvm::Error::success();
}

} // namespace

llvm::Expected<bool> spatialMappingSelectionEqualsCandidate(
    const ::loom::mapping::SpatialMappingView &mapping,
    const SpatialCandidateState &candidate,
    llvm::ArrayRef<const RouteTreeState *> provisionalRoutes,
    llvm::ArrayRef<llvm::ArrayRef<std::optional<llvm::APInt>>>
        provisionalTagValues) {
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  if (mapping.dataflowIdentity() != problem.dataflowIdentity() ||
      mapping.techMappingIdentity() != problem.techMappingIdentity() ||
      mapping.fabricIdentity() != problem.fabricIdentity())
    return false;

  auto compute = compareComputeSelections(mapping, candidate);
  if (!compute || !*compute)
    return compute;
  auto memory = compareMemorySelections(mapping, candidate);
  if (!memory || !*memory)
    return memory;
  auto transport = compareTransportSelections(
      mapping, candidate, provisionalRoutes, provisionalTagValues);
  if (!transport || !*transport)
    return transport;
  if (llvm::Error error = requireRepresentableResourceUseValues(mapping))
    return std::move(error);

  // SpatialMappingView::import strictly consumes every required compute,
  // memory, service, and Physical Tag ResourceUse from the exact selections
  // compared above and rejects extras. ResourceUses therefore have no further
  // independent selection except memory UsePatterns and tag values, both
  // compared explicitly by this projection.
  //
  // The same strict importer derives configuredHardware and
  // FabricHandshakeSelection from bindings, dispositions, terminals, routes,
  // uses, and tag segments. Those projections are validated caches rather
  // than authored Mapping choices, so comparing them again would create a
  // second semantic owner. Search-only fragment, dense-tag, and route-cache
  // state is intentionally absent for the same reason.
  return true;
}

} // namespace loom::pnr
