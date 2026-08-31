#include "PnR/SpatialMappingWarmSeed.h"

#include "Fabric/IR/UsePatternValue.h"
#include "PnR/SpatialMappingSelectionProjection.h"

#include "InitializerRelationSolver.h"
#include "SpatialBindingRelationModel.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Errc.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::pnr {

char SpatialMappingWarmSeedFailure::ID;

void SpatialMappingWarmSeedFailure::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code SpatialMappingWarmSeedFailure::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

namespace {

template <typename T>
llvm::Expected<T> failure(SpatialMappingWarmSeedFailureKind kind,
                          const llvm::Twine &message) {
  return llvm::make_error<SpatialMappingWarmSeedFailure>(kind, message.str());
}

template <typename Range, typename Predicate>
llvm::Expected<PnrIndex> uniqueOrdinal(const Range &range, Predicate predicate,
                                       const llvm::Twine &description) {
  std::optional<PnrIndex> result;
  for (auto indexed : llvm::enumerate(range)) {
    if (!predicate(indexed.value()))
      continue;
    if (indexed.index() > std::numeric_limits<PnrIndex>::max())
      return failure<PnrIndex>(
          SpatialMappingWarmSeedFailureKind::SelectionAbsent,
          description + " exceeds the dense index domain");
    if (result)
      return failure<PnrIndex>(
          SpatialMappingWarmSeedFailureKind::SelectionAmbiguous,
          description + " resolves to more than one frozen selection");
    result = static_cast<PnrIndex>(indexed.index());
  }
  if (!result)
    return failure<PnrIndex>(SpatialMappingWarmSeedFailureKind::SelectionAbsent,
                             description + " has no frozen selection");
  return *result;
}

const ::loom::mapping::SpatialComputeBindingView *
findComputeBinding(const ::loom::mapping::SpatialMappingView &mapping,
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

const ::loom::mapping::SpatialMemoryEngineBindingView *
findMemoryEngine(const ::loom::mapping::SpatialMappingView &mapping,
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

const ::loom::mapping::SpatialMemoryBindingView *
findMemoryBinding(const ::loom::mapping::SpatialMappingView &mapping,
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

const ::loom::mapping::SpatialMemoryOperationView *findMemoryOperation(
    const ::loom::mapping::SpatialMemoryEngineBindingView &engine,
    const ::dataflow::ActorRef &actor) {
  const ::loom::mapping::SpatialMemoryOperationView *result = nullptr;
  for (const auto &operation : engine.operations) {
    const auto &candidate = std::visit(
        [](const auto &record) -> const ::dataflow::ActorRef & {
          return record.actor;
        },
        operation);
    if (candidate != actor)
      continue;
    if (result)
      return nullptr;
    result = &operation;
  }
  return result;
}

const ::loom::mapping::SpatialRouteTreeView *
findRoute(const ::loom::mapping::SpatialMappingView &mapping,
          const ::dataflow::CanonicalGraphProducerEndpointRef &producer,
          std::size_t *mappingOrdinal = nullptr) {
  const ::loom::mapping::SpatialRouteTreeView *result = nullptr;
  for (auto indexed : llvm::enumerate(mapping.routeTrees()))
    if (indexed.value().logicalNet == producer) {
      if (result)
        return nullptr;
      result = &indexed.value();
      if (mappingOrdinal)
        *mappingOrdinal = indexed.index();
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

const ::loom::mapping::SpatialRouteNodeView *
findRouteNode(const ::loom::mapping::SpatialRouteTreeView &route,
              std::uint64_t ordinal) {
  const ::loom::mapping::SpatialRouteNodeView *result = nullptr;
  for (const auto &node : route.nodes)
    if (node.ordinal == ordinal) {
      if (result)
        return nullptr;
      result = &node;
    }
  return result;
}

const ::loom::mapping::SpatialRouteSinkView *
findRouteSink(const ::loom::mapping::SpatialRouteTreeView &route,
              const ::dataflow::CanonicalGraphConsumerEndpointRef &sink) {
  const ::loom::mapping::SpatialRouteSinkView *result = nullptr;
  for (const auto &candidate : route.sinks)
    if (candidate.sink == sink) {
      if (result)
        return nullptr;
      result = &candidate;
    }
  return result;
}

bool memoryDispatchEqual(
    const ::loom::mapping::SpatialMemoryDispatchTargetView &mapped,
    const FrozenSpatialMemoryDispatchTarget &candidate) {
  return std::visit(
      [&](const auto &selected) {
        using Selected = std::decay_t<decltype(selected)>;
        if constexpr (std::is_same_v<
                          Selected,
                          ::loom::fabric::MemoryConsistencyDomainRef>) {
          return false;
        } else if (const auto *value = std::get_if<Selected>(&mapped)) {
          return *value == selected;
        }
        return false;
      },
      candidate);
}

bool memoryConsistencyEqual(
    const ::loom::mapping::SpatialMemoryConsistencyTargetView &mapped,
    const FrozenSpatialMemoryDispatchTarget &candidate) {
  return std::visit(
      [&](const auto &selected) {
        using Selected = std::decay_t<decltype(selected)>;
        if constexpr (std::is_same_v<Selected,
                                     ::loom::fabric::LocalMemoryServiceRef>) {
          return false;
        } else if (const auto *value = std::get_if<Selected>(&mapped)) {
          return *value == selected;
        }
        return false;
      },
      candidate);
}

bool exposureDispatchEqual(
    const ::loom::mapping::SpatialMemoryDispatchTargetView &mapped,
    const FrozenSpatialMemoryExposureDispatchTarget &candidate) {
  return std::visit(
      [&](const auto &selected) {
        if (const auto *value =
                std::get_if<std::decay_t<decltype(selected)>>(&mapped))
          return *value == selected;
        return false;
      },
      candidate);
}

llvm::Expected<::loom::mapping::SpatialActivityEventRef>
memoryPlanTrigger(const FrozenSpatialPnrProblem &problem, PnrIndex plan) {
  const auto planEnvelopes = problem.capacity().memoryOperationPlanEnvelopes();
  const auto envelopes = problem.capacity().resourceTimeEnvelopes();
  const auto events = problem.capacity().resourceEvents();
  if (plan >= planEnvelopes.size() || planEnvelopes[plan] >= envelopes.size() ||
      envelopes[planEnvelopes[plan]].event >= events.size())
    return failure<::loom::mapping::SpatialActivityEventRef>(
        SpatialMappingWarmSeedFailureKind::SelectionAbsent,
        "memory operation plan has no frozen activation event");
  return events[envelopes[planEnvelopes[plan]].event].reference;
}

const ::loom::mapping::SpatialResourceUseView *
findResourceUse(const ::loom::mapping::SpatialMappingView &mapping,
                const ::loom::mapping::SpatialResourceOwnerRef &owner,
                const ::loom::mapping::SpatialActivityEventRef &trigger,
                const ::loom::fabric::FabricUsePatternRef &useSite) {
  const auto ownerEqual = [&](const auto &left) {
    using Owner = std::decay_t<decltype(left)>;
    const auto *right = std::get_if<Owner>(&owner);
    if (!right)
      return false;
    if constexpr (std::is_same_v<
                      Owner, ::loom::mapping::SpatialComputeResourceOwnerRef>)
      return left.realization == right->realization;
    else if constexpr (std::is_same_v<Owner,
                                      ::loom::mapping::
                                          SpatialMemoryEngineResourceOwnerRef>)
      return left.realization == right->realization;
    else if constexpr (std::is_same_v<Owner,
                                      ::loom::mapping::
                                          SpatialMemoryBindingResourceOwnerRef>)
      return left.binding == right->binding;
    else
      return left.logicalNet == right->logicalNet &&
             left.nodeOrdinal == right->nodeOrdinal;
  };
  const auto eventEqual = [](const auto &left, const auto &right) {
    if (left.index() != right.index())
      return false;
    switch (left.index()) {
    case 0:
      return std::get<::loom::mapping::SpatialActorTransitionEventRef>(left) ==
             std::get<::loom::mapping::SpatialActorTransitionEventRef>(right);
    case 1:
      return std::get<::dataflow::CanonicalGraphProducerEndpointRef>(left) ==
             std::get<::dataflow::CanonicalGraphProducerEndpointRef>(right);
    case 2:
      return std::get<::dataflow::CanonicalGraphConsumerEndpointRef>(left) ==
             std::get<::dataflow::CanonicalGraphConsumerEndpointRef>(right);
    default:
      llvm_unreachable("closed Spatial activity event variant");
    }
  };
  const ::loom::mapping::SpatialResourceUseView *result = nullptr;
  for (const auto &use : mapping.resourceUses()) {
    if (!std::visit(ownerEqual, use.owner) ||
        !eventEqual(use.activation.trigger.event, trigger) ||
        use.useSite != useSite)
      continue;
    if (result)
      return nullptr;
    result = &use;
  }
  return result;
}

llvm::Expected<PnrIndex>
memoryOperationDomain(const FrozenSpatialPnrProblem &problem, PnrIndex actor,
                      PnrIndex placement) {
  const auto &realizations = problem.realizations();
  if (actor >= realizations.memoryActorRealizations().size())
    return failure<PnrIndex>(SpatialMappingWarmSeedFailureKind::SelectionAbsent,
                             "memory actor has no realization");
  const PnrIndex realization = realizations.memoryActorRealizations()[actor];
  if (realization >= realizations.memoryRealizations().size())
    return failure<PnrIndex>(SpatialMappingWarmSeedFailureKind::SelectionAbsent,
                             "memory actor realization is out of range");
  const auto &record = realizations.memoryRealizations()[realization];
  const auto offsets = problem.handshake().memoryPlacementDomainOffsets();
  if (placement + 1 >= offsets.size() || actor < record.actorOffset ||
      actor - record.actorOffset >= record.actorCount)
    return failure<PnrIndex>(SpatialMappingWarmSeedFailureKind::SelectionAbsent,
                             "memory actor has no placement plan domain");
  const PnrIndex domain = offsets[placement] + actor - record.actorOffset;
  if (domain >= offsets[placement + 1] ||
      domain >= problem.handshake().memoryOperationDomains().size())
    return failure<PnrIndex>(SpatialMappingWarmSeedFailureKind::SelectionAbsent,
                             "memory actor plan domain is absent");
  const auto &frozen = problem.handshake().memoryOperationDomains()[domain];
  if (frozen.actor != actor || frozen.placement != placement)
    return failure<PnrIndex>(SpatialMappingWarmSeedFailureKind::SelectionAbsent,
                             "memory actor plan domain has foreign ownership");
  return domain;
}

struct MemoryResidentCursor final {
  ::loom::fabric::FabricMemoryOccurrenceRef occurrence;
  std::uint64_t next = 0;
};

std::uint64_t &
residentCursor(std::vector<MemoryResidentCursor> &cursors,
               const ::loom::fabric::FabricMemoryOccurrenceRef &occurrence) {
  for (auto &cursor : cursors)
    if (cursor.occurrence == occurrence)
      return cursor.next;
  cursors.push_back({occurrence, 0});
  return cursors.back().next;
}

bool mappedPlacementMatches(
    const ::loom::mapping::SpatialMemoryOperationPlacementView &mapped,
    const FrozenSpatialMemoryPlacement &placement,
    const FrozenSpatialMemoryActorBinding &actor, bool temporalResident,
    std::uint64_t contextOrdinal) {
  const ::loom::fabric::FabricMemoryOperationPortRef port{
      placement.memory, actor.operationPort.ordinal};
  if (!temporalResident) {
    const auto *value =
        std::get_if<::loom::fabric::FabricMemoryOperationPortRef>(&mapped);
    return value && *value == port;
  }
  const auto *value =
      std::get_if<::loom::fabric::FabricMemoryOperationContextRef>(&mapped);
  return value && value->port == port && value->ordinal == contextOrdinal;
}

llvm::Expected<PnrIndex> selectMemoryPlan(
    const ::loom::mapping::SpatialMappingView &mapping,
    const FrozenSpatialPnrProblem &problem, PnrIndex realization,
    PnrIndex actorOrdinal, PnrIndex placementOrdinal,
    const ::loom::mapping::SpatialMemoryOperationView &mappedOperation,
    std::vector<MemoryResidentCursor> &residentCursors) {
  auto domainOrdinal =
      memoryOperationDomain(problem, actorOrdinal, placementOrdinal);
  if (!domainOrdinal)
    return domainOrdinal.takeError();
  const auto &domain =
      problem.handshake().memoryOperationDomains()[*domainOrdinal];
  const auto plans = problem.handshake().memoryOperationPlans();
  const auto patterns = problem.resources().usePatterns();
  const auto &realizations = problem.realizations();
  if (placementOrdinal >= realizations.memoryPlacements().size() ||
      actorOrdinal >= realizations.memoryActors().size())
    return failure<PnrIndex>(SpatialMappingWarmSeedFailureKind::SelectionAbsent,
                             "memory operation frozen owner is absent");
  const auto &placement = realizations.memoryPlacements()[placementOrdinal];
  const auto &actor = realizations.memoryActors()[actorOrdinal];
  const auto &mappedPlacement = std::visit(
      [](const auto &operation)
          -> const ::loom::mapping::SpatialMemoryOperationPlacementView & {
        return operation.placement;
      },
      mappedOperation);
  const bool mappedResident =
      std::holds_alternative<::loom::fabric::FabricMemoryOperationContextRef>(
          mappedPlacement);
  std::uint64_t *contextCursor =
      mappedResident ? &residentCursor(residentCursors, placement.memory)
                     : nullptr;
  const std::uint64_t contextOrdinal = contextCursor ? *contextCursor : 0;

  std::optional<PnrIndex> selected;
  for (PnrIndex local = 0; local < domain.planCount; ++local) {
    const PnrIndex ordinal = domain.planOffset + local;
    if (ordinal >= plans.size() ||
        plans[ordinal].temporalResident != mappedResident ||
        plans[ordinal].usePattern >= patterns.size() ||
        !mappedPlacementMatches(mappedPlacement, placement, actor,
                                mappedResident, contextOrdinal))
      continue;
    auto trigger = memoryPlanTrigger(problem, ordinal);
    if (!trigger)
      return trigger.takeError();
    const auto &useSite = patterns[plans[ordinal].usePattern].reference;
    const auto *use = findResourceUse(
        mapping,
        ::loom::mapping::SpatialResourceOwnerRef(
            ::loom::mapping::SpatialMemoryEngineResourceOwnerRef{
                problem.realizations()
                    .memoryRealizations()[realization]
                    .reference.entity}),
        *trigger, useSite);
    if (!use)
      continue;
    if (selected)
      return failure<PnrIndex>(
          SpatialMappingWarmSeedFailureKind::SelectionAmbiguous,
          "memory operation resolves to multiple frozen handshake plans");
    selected = ordinal;
  }
  if (!selected)
    return failure<PnrIndex>(SpatialMappingWarmSeedFailureKind::SelectionAbsent,
                             "memory operation has no frozen handshake plan");
  if (mappedResident && (!placement.residentContextCount ||
                         contextOrdinal >= *placement.residentContextCount))
    return failure<PnrIndex>(SpatialMappingWarmSeedFailureKind::SelectionAbsent,
                             "memory operation context exceeds its occurrence");
  if (contextCursor)
    ++*contextCursor;
  return *selected;
}

llvm::Expected<std::vector<SpatialComputeBindingSelection>>
projectComputeBindings(const ::loom::mapping::SpatialMappingView &mapping,
                       const FrozenSpatialPnrProblem &problem) {
  const auto &realizations = problem.realizations();
  std::vector<SpatialComputeBindingSelection> selections;
  selections.reserve(realizations.computeRealizations().size());
  for (PnrIndex realization = 0;
       realization < realizations.computeRealizations().size(); ++realization) {
    const auto &frozen = realizations.computeRealizations()[realization];
    const auto *mapped = findComputeBinding(mapping, frozen.reference.entity);
    if (!mapped)
      return failure<std::vector<SpatialComputeBindingSelection>>(
          SpatialMappingWarmSeedFailureKind::SelectionAbsent,
          "compute realization does not resolve uniquely in the parent");
    if (!mapped->refinements.empty())
      return failure<std::vector<SpatialComputeBindingSelection>>(
          SpatialMappingWarmSeedFailureKind::SelectionAbsent,
          "compute refinements are outside the Candidate selection domain");
    auto placement = uniqueOrdinal(
        realizations.computePlacements(),
        [&](const FrozenSpatialComputePlacement &candidate) {
          return candidate.realization == realization &&
                 candidate.fu == mapped->occurrence;
        },
        "compute occurrence");
    if (!placement)
      return placement.takeError();
    const auto &placed = realizations.computePlacements()[*placement];
    auto context = uniqueOrdinal(
        realizations.computeInstructionContexts(),
        [&](const ::loom::fabric::InstructionContextRef &candidate) {
          const auto index = static_cast<PnrIndex>(
              &candidate - realizations.computeInstructionContexts().data());
          return index >= placed.contextOffset &&
                 index - placed.contextOffset < placed.contextCount &&
                 candidate == mapped->context;
        },
        "compute InstructionContext");
    if (!context)
      return context.takeError();
    selections.push_back({*placement, *context});
  }
  return selections;
}

llvm::Expected<std::vector<SpatialMemoryBindingSelection>>
projectMemoryPlacements(const ::loom::mapping::SpatialMappingView &mapping,
                        const FrozenSpatialPnrProblem &problem) {
  const auto &realizations = problem.realizations();
  std::vector<SpatialMemoryBindingSelection> selections;
  selections.reserve(realizations.memoryRealizations().size());
  for (PnrIndex realization = 0;
       realization < realizations.memoryRealizations().size(); ++realization) {
    const auto &frozen = realizations.memoryRealizations()[realization];
    const auto *mapped = findMemoryEngine(mapping, frozen.reference.entity);
    if (!mapped)
      return failure<std::vector<SpatialMemoryBindingSelection>>(
          SpatialMappingWarmSeedFailureKind::SelectionAbsent,
          "memory realization does not resolve uniquely in the parent");
    auto placement = uniqueOrdinal(
        realizations.memoryPlacements(),
        [&](const FrozenSpatialMemoryPlacement &candidate) {
          return candidate.realization == realization &&
                 candidate.memory == mapped->occurrence;
        },
        "memory occurrence");
    if (!placement)
      return placement.takeError();
    selections.push_back({*placement});
  }
  return selections;
}

struct MemorySelections final {
  std::vector<PnrIndex> operationPlans;
  std::vector<SpatialLogicalMemoryBindingSelection> logicalBindings;
  std::vector<PnrIndex> useDispatches;
  std::vector<PnrIndex> exposureSelections;
};

bool bindingTargetMatches(
    const ::loom::mapping::SpatialMemoryBindingTargetView &mapped,
    const FrozenSpatialMemoryBindingTargetOption &candidate) {
  if (const auto *region =
          std::get_if<::loom::fabric::FabricMemoryServiceRegionRef>(
              &candidate.target)) {
    const auto *local =
        std::get_if<::loom::mapping::SpatialMemoryLocalRegionView>(&mapped);
    return local && local->serviceRegion == *region;
  }
  return std::holds_alternative<
      ::loom::mapping::SpatialMemoryBoundaryProxyView>(mapped);
}

llvm::Expected<MemorySelections> projectMemorySelections(
    const ::loom::mapping::SpatialMappingView &mapping,
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<SpatialMemoryBindingSelection> placements) {
  const auto &realizations = problem.realizations();
  const auto &memory = problem.memory();
  MemorySelections result;
  result.operationPlans.assign(realizations.memoryActors().size(),
                               getInvalidPnrIndex());
  std::vector<MemoryResidentCursor> residentCursors;
  for (PnrIndex realization = 0;
       realization < realizations.memoryRealizations().size(); ++realization) {
    if (realization >= placements.size())
      return failure<MemorySelections>(
          SpatialMappingWarmSeedFailureKind::SelectionAbsent,
          "memory placement projection is incomplete");
    const auto &frozen = realizations.memoryRealizations()[realization];
    const auto *engine = findMemoryEngine(mapping, frozen.reference.entity);
    if (!engine || engine->operations.size() != frozen.actorCount)
      return failure<MemorySelections>(
          SpatialMappingWarmSeedFailureKind::SelectionAbsent,
          "memory engine operation inventory is incomplete");
    for (PnrIndex local = 0; local < frozen.actorCount; ++local) {
      const PnrIndex actorOrdinal = frozen.actorOffset + local;
      if (actorOrdinal >= realizations.memoryActors().size())
        return failure<MemorySelections>(
            SpatialMappingWarmSeedFailureKind::SelectionAbsent,
            "memory actor slice is out of range");
      const auto *operation = findMemoryOperation(
          *engine, realizations.memoryActors()[actorOrdinal].actor);
      if (!operation)
        return failure<MemorySelections>(
            SpatialMappingWarmSeedFailureKind::SelectionAbsent,
            "memory operation does not resolve uniquely in the parent");
      auto plan = selectMemoryPlan(mapping, problem, realization, actorOrdinal,
                                   placements[realization].placement,
                                   *operation, residentCursors);
      if (!plan)
        return plan.takeError();
      result.operationPlans[actorOrdinal] = *plan;
    }
  }

  result.logicalBindings.reserve(memory.logicalBindings().size());
  result.exposureSelections.assign(memory.exposures().size(),
                                   getInvalidPnrIndex());
  for (PnrIndex bindingOrdinal = 0;
       bindingOrdinal < memory.logicalBindings().size(); ++bindingOrdinal) {
    const auto *mapped = findMemoryBinding(mapping, bindingOrdinal);
    if (!mapped ||
        mapped->logicalMemory !=
            memory.logicalBindings()[bindingOrdinal].logicalMemory ||
        !std::holds_alternative<
            ::loom::mapping::SpatialMemoryWholeIntervalView>(mapped->interval))
      return failure<MemorySelections>(
          SpatialMappingWarmSeedFailureKind::SelectionAbsent,
          "logical memory binding does not resolve uniquely in the parent");
    auto target = uniqueOrdinal(
        memory.bindingTargets(),
        [&](const FrozenSpatialMemoryBindingTargetOption &candidate) {
          return bindingTargetMatches(mapped->target, candidate);
        },
        "logical memory target");
    if (!target)
      return target.takeError();
    std::uint64_t offset = 0;
    if (const auto *local =
            std::get_if<::loom::mapping::SpatialMemoryLocalRegionView>(
                &mapped->target))
      offset = local->physicalOffsetBytes;
    result.logicalBindings.push_back({*target, offset});

    if (bindingOrdinal + 1 >= memory.bindingExposureOffsets().size())
      return failure<MemorySelections>(
          SpatialMappingWarmSeedFailureKind::SelectionAbsent,
          "logical memory exposure CSR is incomplete");
    const PnrIndex begin = memory.bindingExposureOffsets()[bindingOrdinal];
    const PnrIndex end = memory.bindingExposureOffsets()[bindingOrdinal + 1];
    if (begin > end || end > memory.bindingExposures().size() ||
        mapped->exposures.size() != end - begin)
      return failure<MemorySelections>(
          SpatialMappingWarmSeedFailureKind::SelectionAbsent,
          "logical memory exposure inventory is incomplete");
    for (PnrIndex incidence = begin; incidence < end; ++incidence) {
      const PnrIndex exposureOrdinal = memory.bindingExposures()[incidence];
      if (exposureOrdinal >= memory.exposures().size())
        return failure<MemorySelections>(
            SpatialMappingWarmSeedFailureKind::SelectionAbsent,
            "logical memory exposure ordinal is out of range");
      const auto &exposure = memory.exposures()[exposureOrdinal];
      const ::loom::mapping::SpatialExposureEntryView *mappedExposure = nullptr;
      for (const auto &entry : mapped->exposures)
        if (entry.exposure == exposure.exposure) {
          if (mappedExposure)
            return failure<MemorySelections>(
                SpatialMappingWarmSeedFailureKind::SelectionAmbiguous,
                "memory exposure is repeated in the parent");
          mappedExposure = &entry;
        }
      if (!mappedExposure)
        return failure<MemorySelections>(
            SpatialMappingWarmSeedFailureKind::SelectionAbsent,
            "memory exposure is absent from the parent");
      auto option = uniqueOrdinal(
          memory.exposureOptions(),
          [&](const FrozenSpatialMemoryExposureOption &candidate) {
            return candidate.provider < memory.exposureProviders().size() &&
                   memory.exposureProviders()[candidate.provider].terminal ==
                       mappedExposure->terminal &&
                   exposureDispatchEqual(mappedExposure->dispatch,
                                         candidate.target);
          },
          "memory exposure option");
      if (!option)
        return option.takeError();
      result.exposureSelections[exposureOrdinal] = *option;
    }
  }

  result.useDispatches.assign(memory.rootedUses().size(), getInvalidPnrIndex());
  for (PnrIndex useOrdinal = 0; useOrdinal < memory.rootedUses().size();
       ++useOrdinal) {
    const auto &use = memory.rootedUses()[useOrdinal];
    if (use.actor >= realizations.memoryActorRealizations().size())
      return failure<MemorySelections>(
          SpatialMappingWarmSeedFailureKind::SelectionAbsent,
          "rooted memory use has no actor realization");
    const PnrIndex realization =
        realizations.memoryActorRealizations()[use.actor];
    if (realization >= realizations.memoryRealizations().size() ||
        realization >= placements.size())
      return failure<MemorySelections>(
          SpatialMappingWarmSeedFailureKind::SelectionAbsent,
          "rooted memory use has no placement");
    const auto *engine = findMemoryEngine(
        mapping,
        realizations.memoryRealizations()[realization].reference.entity);
    const auto *operation =
        engine ? findMemoryOperation(
                     *engine, realizations.memoryActors()[use.actor].actor)
               : nullptr;
    if (!operation)
      return failure<MemorySelections>(
          SpatialMappingWarmSeedFailureKind::SelectionAbsent,
          "rooted memory use has no parent operation");

    auto domainOrdinal = memoryOperationDomain(
        problem, use.actor, placements[realization].placement);
    if (!domainOrdinal)
      return domainOrdinal.takeError();
    const auto placementOffsets = memory.memoryPlacementDomainOffsets();
    const PnrIndex dispatchDomain =
        placementOffsets[placements[realization].placement] + use.actor -
        realizations.memoryRealizations()[realization].actorOffset;
    if (dispatchDomain >= memory.dispatchDomains().size())
      return failure<MemorySelections>(
          SpatialMappingWarmSeedFailureKind::SelectionAbsent,
          "rooted memory dispatch domain is absent");
    const auto &domain = memory.dispatchDomains()[dispatchDomain];

    const ::loom::mapping::SpatialMemoryDispatchTargetView *addressedTarget =
        nullptr;
    const ::loom::mapping::SpatialMemoryConsistencyTargetView *fenceTarget =
        nullptr;
    if (const auto *addressed =
            std::get_if<::loom::mapping::SpatialAddressedMemoryOperationView>(
                operation)) {
      for (const auto &entry : addressed->uses)
        if (entry.launch == use.launch && use.logicalBinding &&
            entry.binding == *use.logicalBinding) {
          if (addressedTarget)
            return failure<MemorySelections>(
                SpatialMappingWarmSeedFailureKind::SelectionAmbiguous,
                "addressed memory use is repeated in the parent");
          addressedTarget = &entry.dispatch;
        }
    } else {
      const auto &fence =
          std::get<::loom::mapping::SpatialFenceMemoryOperationView>(
              *operation);
      for (const auto &entry : fence.uses)
        if (entry.launch == use.launch) {
          if (fenceTarget)
            return failure<MemorySelections>(
                SpatialMappingWarmSeedFailureKind::SelectionAmbiguous,
                "fence memory use is repeated in the parent");
          fenceTarget = &entry.consistency;
        }
    }
    if ((!addressedTarget && !fenceTarget) || (addressedTarget && fenceTarget))
      return failure<MemorySelections>(
          SpatialMappingWarmSeedFailureKind::SelectionAbsent,
          "rooted memory use is absent from the parent operation");

    std::optional<PnrIndex> selected;
    for (PnrIndex local = 0; local < domain.optionCount; ++local) {
      const PnrIndex optionOrdinal = domain.optionOffset + local;
      if (optionOrdinal >= memory.dispatchOptions().size())
        return failure<MemorySelections>(
            SpatialMappingWarmSeedFailureKind::SelectionAbsent,
            "memory dispatch option is out of range");
      const auto &option = memory.dispatchOptions()[optionOrdinal];
      if ((addressedTarget &&
           !memoryDispatchEqual(*addressedTarget, option.target)) ||
          (fenceTarget && !memoryConsistencyEqual(*fenceTarget, option.target)))
        continue;
      if (option.serviceUsePattern) {
        if (!use.logicalBinding ||
            *use.logicalBinding >= memory.logicalBindings().size())
          continue;
        auto trigger =
            memoryPlanTrigger(problem, result.operationPlans[use.actor]);
        if (!trigger)
          return trigger.takeError();
        const auto *resourceUse = findResourceUse(
            mapping,
            ::loom::mapping::SpatialResourceOwnerRef(
                ::loom::mapping::SpatialMemoryBindingResourceOwnerRef{
                    *use.logicalBinding}),
            *trigger, *option.serviceUsePattern);
        if (!resourceUse)
          continue;
      }
      if (selected)
        return failure<MemorySelections>(
            SpatialMappingWarmSeedFailureKind::SelectionAmbiguous,
            "rooted memory use resolves to multiple dispatch options");
      selected = optionOrdinal;
    }
    if (!selected)
      return failure<MemorySelections>(
          SpatialMappingWarmSeedFailureKind::SelectionAbsent,
          "rooted memory use has no frozen dispatch option");
    result.useDispatches[useOrdinal] = *selected;
  }
  return result;
}

bool exactTagEqual(const llvm::APInt &left, const llvm::APInt &right) {
  return left.getBitWidth() == right.getBitWidth() && left == right;
}

llvm::Expected<std::vector<PnrIndex>>
projectLocalDispositions(const ::loom::mapping::SpatialMappingView &mapping,
                         const FrozenSpatialPnrProblem &problem) {
  const auto nets = problem.transfers().logicalNets();
  const auto sinks = problem.transfers().logicalNetSinks();
  const auto options = problem.localTransfers().options();
  const auto domains = problem.localTransfers().domains();
  const auto traversals = problem.routing().traversals();
  std::vector<PnrIndex> selections(nets.size(), getInvalidPnrIndex());
  for (PnrIndex logicalNet = 0; logicalNet < nets.size(); ++logicalNet) {
    const auto *mapped =
        findRegisterTransfer(mapping, nets[logicalNet].producer);
    if (!mapped)
      continue;
    if (logicalNet >= domains.size() || nets[logicalNet].sinkCount != 1 ||
        nets[logicalNet].sinkOffset >= sinks.size())
      return failure<std::vector<PnrIndex>>(
          SpatialMappingWarmSeedFailureKind::SelectionAbsent,
          "register-FIFO transfer has no frozen logical-net domain");
    const auto &domain = domains[logicalNet];
    std::optional<PnrIndex> selected;
    for (PnrIndex local = 0; local < domain.optionCount; ++local) {
      const PnrIndex ordinal = domain.optionOffset + local;
      if (ordinal >= options.size())
        return failure<std::vector<PnrIndex>>(
            SpatialMappingWarmSeedFailureKind::SelectionAbsent,
            "register-FIFO option is out of range");
      const auto &option = options[ordinal];
      if (option.writeTraversal >= traversals.size() ||
          option.readTraversal >= traversals.size() ||
          option.logicalNet != logicalNet ||
          mapped->sink != sinks[nets[logicalNet].sinkOffset] ||
          mapped->pe != option.pe ||
          mapped->registerFifo != option.registerFifo ||
          mapped->writeTraversal !=
              traversals[option.writeTraversal].reference ||
          mapped->readTraversal != traversals[option.readTraversal].reference ||
          !exactTagEqual(mapped->tag, option.tag))
        continue;
      if (selected)
        return failure<std::vector<PnrIndex>>(
            SpatialMappingWarmSeedFailureKind::SelectionAmbiguous,
            "register-FIFO transfer resolves to multiple frozen options");
      selected = ordinal;
    }
    if (!selected)
      return failure<std::vector<PnrIndex>>(
          SpatialMappingWarmSeedFailureKind::SelectionAbsent,
          "register-FIFO transfer has no frozen option");
    selections[logicalNet] = *selected;
  }
  return selections;
}

std::optional<::loom::fabric::FabricPhysicalTraversalRef>
attachmentLocalTraversal(const FrozenSpatialPnrProblem &problem,
                         const FrozenSpatialAttachmentOption &option) {
  if (!option.localTraversal)
    return std::nullopt;
  if (*option.localTraversal >= problem.routing().traversals().size())
    return std::nullopt;
  return problem.routing().traversals()[*option.localTraversal].reference;
}

llvm::Expected<PnrIndex> attachmentForTerminal(
    const FrozenSpatialPnrProblem &problem,
    FrozenSpatialTerminalBinding binding, PnrIndex endpoint,
    std::optional<::loom::fabric::FabricPhysicalTraversalRef> localTraversal) {
  const auto &relations = problem.bindingRelations();
  const auto options = problem.ports().attachmentOptions();
  const auto choices =
      binding.kind == FrozenSpatialTerminalBindingKind::PortDemand
          ? relations.portAttachmentChoices(binding.index)
          : relations.graphBoundaryAttachmentChoices(binding.index);
  std::optional<PnrIndex> selected;
  for (PnrIndex optionOrdinal : choices) {
    if (optionOrdinal >= options.size())
      return failure<PnrIndex>(
          SpatialMappingWarmSeedFailureKind::SelectionAbsent,
          "terminal attachment domain contains an absent option");
    const auto &option = options[optionOrdinal];
    if (option.endpoint != endpoint ||
        attachmentLocalTraversal(problem, option) != localTraversal)
      continue;
    if (selected)
      return failure<PnrIndex>(
          SpatialMappingWarmSeedFailureKind::SelectionAmbiguous,
          "terminal resolves to multiple frozen attachment options");
    selected = optionOrdinal;
  }
  if (!selected)
    return failure<PnrIndex>(SpatialMappingWarmSeedFailureKind::SelectionAbsent,
                             "terminal has no frozen attachment option");
  return *selected;
}

struct RootSelections final {
  std::vector<PnrIndex> portAttachments;
  std::vector<PnrIndex> graphBoundaryAttachments;
  std::uint64_t privatePorts = 0;
  std::uint64_t privateBoundaries = 0;
  std::uint64_t relationAssignmentAttempts = 0;
};

llvm::Expected<RootSelections> projectAttachments(
    const ::loom::mapping::SpatialMappingView &mapping,
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<SpatialComputeBindingSelection> computeBindings,
    llvm::ArrayRef<SpatialMemoryBindingSelection> memoryBindings,
    llvm::ArrayRef<PnrIndex> registerFifoTransfers) {
  const auto &relations = problem.bindingRelations();
  std::vector<PnrIndex> fixed(relations.decisionCount(), 0);
  std::vector<PnrIndex> released;
  for (PnrIndex realization = 0; realization < computeBindings.size();
       ++realization) {
    auto choice = relations.computeChoiceOrdinal(
        realization, computeBindings[realization].placement,
        computeBindings[realization].instructionContext);
    if (!choice)
      return failure<RootSelections>(
          SpatialMappingWarmSeedFailureKind::SelectionAbsent,
          "compute binding has no frozen relation choice");
    fixed[realization] = *choice;
  }
  for (PnrIndex realization = 0; realization < memoryBindings.size();
       ++realization) {
    auto choice = relations.memoryChoiceOrdinal(
        realization, memoryBindings[realization].placement);
    if (!choice)
      return failure<RootSelections>(
          SpatialMappingWarmSeedFailureKind::SelectionAbsent,
          "memory binding has no frozen relation choice");
    fixed[relations.computeDecisionCount() + realization] = *choice;
  }

  RootSelections result;
  result.portAttachments.assign(problem.ports().portDemands().size(),
                                getInvalidPnrIndex());
  result.graphBoundaryAttachments.assign(
      problem.ports().graphBoundaries().size(), getInvalidPnrIndex());
  const auto nets = problem.transfers().logicalNets();
  const auto sinks = problem.transfers().logicalNetSinks();
  for (PnrIndex logicalNet = 0; logicalNet < nets.size(); ++logicalNet) {
    const bool privateDisposition =
        logicalNet < registerFifoTransfers.size() &&
        registerFifoTransfers[logicalNet] != getInvalidPnrIndex();
    const auto sourceBinding =
        problem.transfers().logicalNetSourceBindings()[logicalNet];
    const auto assignTerminal = [&](FrozenSpatialTerminalBinding binding,
                                    PnrIndex option,
                                    bool privateSelection) -> llvm::Error {
      if (binding.kind == FrozenSpatialTerminalBindingKind::PortDemand) {
        if (binding.index >= result.portAttachments.size())
          return failure<bool>(
                     SpatialMappingWarmSeedFailureKind::SelectionAbsent,
                     "route terminal names an absent PortDemand")
              .takeError();
        if (privateSelection) {
          if (result.portAttachments[binding.index] != getInvalidPnrIndex())
            return failure<bool>(
                       SpatialMappingWarmSeedFailureKind::SelectionAmbiguous,
                       "one PortDemand is both persistent and private")
                .takeError();
          const PnrIndex decision =
              relations.portDecisionOffset() + binding.index;
          if (!llvm::is_contained(released, decision)) {
            released.push_back(decision);
            ++result.privatePorts;
          }
          return llvm::Error::success();
        }
        if (result.portAttachments[binding.index] != getInvalidPnrIndex() &&
            result.portAttachments[binding.index] != option)
          return failure<bool>(
                     SpatialMappingWarmSeedFailureKind::SelectionAmbiguous,
                     "one PortDemand resolves to different parent options")
              .takeError();
        result.portAttachments[binding.index] = option;
        auto choice =
            relations.portAttachmentChoiceOrdinal(binding.index, option);
        if (!choice)
          return failure<bool>(
                     SpatialMappingWarmSeedFailureKind::SelectionAbsent,
                     "PortDemand attachment has no relation choice")
              .takeError();
        fixed[relations.portDecisionOffset() + binding.index] = *choice;
        return llvm::Error::success();
      }
      if (binding.index >= result.graphBoundaryAttachments.size())
        return failure<bool>(SpatialMappingWarmSeedFailureKind::SelectionAbsent,
                             "route terminal names an absent graph boundary")
            .takeError();
      if (privateSelection) {
        if (result.graphBoundaryAttachments[binding.index] !=
            getInvalidPnrIndex())
          return failure<bool>(
                     SpatialMappingWarmSeedFailureKind::SelectionAmbiguous,
                     "one graph boundary is both persistent and private")
              .takeError();
        const PnrIndex decision =
            relations.graphBoundaryDecisionOffset() + binding.index;
        if (!llvm::is_contained(released, decision)) {
          released.push_back(decision);
          ++result.privateBoundaries;
        }
        return llvm::Error::success();
      }
      if (result.graphBoundaryAttachments[binding.index] !=
              getInvalidPnrIndex() &&
          result.graphBoundaryAttachments[binding.index] != option)
        return failure<bool>(
                   SpatialMappingWarmSeedFailureKind::SelectionAmbiguous,
                   "one graph boundary resolves to different parent options")
            .takeError();
      result.graphBoundaryAttachments[binding.index] = option;
      auto choice =
          relations.graphBoundaryAttachmentChoiceOrdinal(binding.index, option);
      if (!choice)
        return failure<bool>(SpatialMappingWarmSeedFailureKind::SelectionAbsent,
                             "graph-boundary attachment has no relation choice")
            .takeError();
      fixed[relations.graphBoundaryDecisionOffset() + binding.index] = *choice;
      return llvm::Error::success();
    };

    if (privateDisposition) {
      if (llvm::Error error =
              assignTerminal(sourceBinding, getInvalidPnrIndex(), true))
        return std::move(error);
      for (PnrIndex localSink = 0; localSink < nets[logicalNet].sinkCount;
           ++localSink) {
        const PnrIndex sinkOrdinal = nets[logicalNet].sinkOffset + localSink;
        if (sinkOrdinal >= problem.transfers().logicalNetSinkBindings().size())
          return failure<RootSelections>(
              SpatialMappingWarmSeedFailureKind::SelectionAbsent,
              "register-FIFO sink binding is out of range");
        if (llvm::Error error = assignTerminal(
                problem.transfers().logicalNetSinkBindings()[sinkOrdinal],
                getInvalidPnrIndex(), true))
          return std::move(error);
      }
      continue;
    }

    const auto *route = findRoute(mapping, nets[logicalNet].producer);
    if (!route)
      return failure<RootSelections>(
          SpatialMappingWarmSeedFailureKind::SelectionAbsent,
          "externally routed logical net has no unique parent RouteTree");
    auto sourceEndpoint =
        problem.routing().topology().endpointOrdinal(route->rootEndpoint);
    if (!sourceEndpoint)
      return failure<RootSelections>(
          SpatialMappingWarmSeedFailureKind::SelectionAbsent,
          "parent RouteTree root endpoint is absent from the frozen Fabric");
    auto sourceOption = attachmentForTerminal(
        problem, sourceBinding, *sourceEndpoint, route->localTraversal);
    if (!sourceOption)
      return sourceOption.takeError();
    if (llvm::Error error = assignTerminal(sourceBinding, *sourceOption, false))
      return std::move(error);

    for (PnrIndex localSink = 0; localSink < nets[logicalNet].sinkCount;
         ++localSink) {
      const PnrIndex sinkOrdinal = nets[logicalNet].sinkOffset + localSink;
      if (sinkOrdinal >= sinks.size() ||
          sinkOrdinal >= problem.transfers().logicalNetSinkBindings().size())
        return failure<RootSelections>(
            SpatialMappingWarmSeedFailureKind::SelectionAbsent,
            "RouteTree sink slice is out of range");
      const auto *mappedSink = findRouteSink(*route, sinks[sinkOrdinal]);
      const auto *mappedNode =
          mappedSink ? findRouteNode(*route, mappedSink->nodeOrdinal) : nullptr;
      if (!mappedSink || !mappedNode)
        return failure<RootSelections>(
            SpatialMappingWarmSeedFailureKind::SelectionAbsent,
            "RouteTree sink does not resolve uniquely");
      auto endpoint =
          problem.routing().topology().endpointOrdinal(mappedNode->endpoint);
      if (!endpoint)
        return failure<RootSelections>(
            SpatialMappingWarmSeedFailureKind::SelectionAbsent,
            "RouteTree sink endpoint is absent from the frozen Fabric");
      auto option = attachmentForTerminal(
          problem, problem.transfers().logicalNetSinkBindings()[sinkOrdinal],
          *endpoint, mappedSink->localTraversal);
      if (!option)
        return option.takeError();
      if (llvm::Error error = assignTerminal(
              problem.transfers().logicalNetSinkBindings()[sinkOrdinal],
              *option, false))
        return std::move(error);
    }
  }

  llvm::sort(released);
  released.erase(std::unique(released.begin(), released.end()), released.end());
  if (!released.empty()) {
    detail::InitializerRelationSolver solver(relations.relations());
    const std::uint64_t limit = std::max<std::uint64_t>(
        1, problem.config()
               .policy()
               .search.initializer.assignmentAttemptLimitPerSeed);
    auto solved =
        solver.solveCanonicalWithReleasedChoices(limit, fixed, released);
    result.relationAssignmentAttempts = solver.assignmentAttempts();
    if (!solved)
      return failure<RootSelections>(
          SpatialMappingWarmSeedFailureKind::RelationInfeasible,
          "canonical private attachment completion failed: " +
              llvm::toString(solved.takeError()));
    fixed = std::move(solved->choices);
  } else if (llvm::Error error = relations.verifyChoices(fixed)) {
    return failure<RootSelections>(
        SpatialMappingWarmSeedFailureKind::RelationInfeasible,
        "parent root selections violate the frozen relation: " +
            llvm::toString(std::move(error)));
  }

  for (PnrIndex demand = 0; demand < result.portAttachments.size(); ++demand) {
    if (result.portAttachments[demand] != getInvalidPnrIndex())
      continue;
    const auto choices = relations.portAttachmentChoices(demand);
    const PnrIndex selected = fixed[relations.portDecisionOffset() + demand];
    if (selected >= choices.size())
      return failure<RootSelections>(
          SpatialMappingWarmSeedFailureKind::RelationInfeasible,
          "private PortDemand completion is outside its frozen domain");
    result.portAttachments[demand] = choices[selected];
  }
  for (PnrIndex boundary = 0; boundary < result.graphBoundaryAttachments.size();
       ++boundary) {
    if (result.graphBoundaryAttachments[boundary] != getInvalidPnrIndex())
      continue;
    const auto choices = relations.graphBoundaryAttachmentChoices(boundary);
    const PnrIndex selected =
        fixed[relations.graphBoundaryDecisionOffset() + boundary];
    if (selected >= choices.size())
      return failure<RootSelections>(
          SpatialMappingWarmSeedFailureKind::RelationInfeasible,
          "private graph-boundary completion is outside its frozen domain");
    result.graphBoundaryAttachments[boundary] = choices[selected];
  }
  return result;
}

llvm::Expected<PnrIndex>
routeArc(const FrozenSpatialPnrProblem &problem,
         const ::loom::fabric::FabricTransportEndpointRef &source,
         const ::loom::fabric::FabricTransportEndpointRef &target,
         const ::loom::fabric::FabricPhysicalTraversalRef &traversal) {
  auto sourceOrdinal = problem.routing().topology().endpointOrdinal(source);
  auto targetOrdinal = problem.routing().topology().endpointOrdinal(target);
  auto traversalOrdinal =
      problem.routing().topology().traversalOrdinal(traversal);
  if (!sourceOrdinal || !targetOrdinal || !traversalOrdinal)
    return failure<PnrIndex>(
        SpatialMappingWarmSeedFailureKind::RouteProjectionInvalid,
        "RouteTree arc names a reference outside the frozen Fabric");
  const auto arcs = problem.routing().routingArcs();
  const auto sources = problem.routing().arcSources();
  std::optional<PnrIndex> result;
  for (PnrIndex ordinal = 0; ordinal < arcs.size(); ++ordinal) {
    if (ordinal >= sources.size() || sources[ordinal] != *sourceOrdinal ||
        arcs[ordinal].target != *targetOrdinal ||
        arcs[ordinal].traversal != *traversalOrdinal)
      continue;
    if (result)
      return failure<PnrIndex>(
          SpatialMappingWarmSeedFailureKind::RouteProjectionInvalid,
          "RouteTree edge resolves to multiple frozen routing arcs");
    result = ordinal;
  }
  if (!result)
    return failure<PnrIndex>(
        SpatialMappingWarmSeedFailureKind::RouteProjectionInvalid,
        "RouteTree edge has no frozen routing arc");
  return *result;
}

llvm::Expected<llvm::APInt>
mappedTag(const ::loom::mapping::SpatialMappingView &mapping,
          const ::loom::mapping::SpatialPhysicalTagSegmentView &segment) {
  if (segment.resourceUseOrdinal >= mapping.resourceUses().size())
    return failure<llvm::APInt>(
        SpatialMappingWarmSeedFailureKind::TagProjectionInvalid,
        "Physical Tag segment ResourceUse is out of range");
  const auto &use = mapping.resourceUses()[segment.resourceUseOrdinal];
  if (!use.parameters.empty() || use.sharingAssignments.size() != 1)
    return failure<llvm::APInt>(
        SpatialMappingWarmSeedFailureKind::TagProjectionInvalid,
        "Physical Tag ResourceUse has an invalid value shape");
  const auto *tag = std::get_if<::fabric::PhysicalTagPatternValue>(
      &use.sharingAssignments.front());
  if (!tag)
    return failure<llvm::APInt>(
        SpatialMappingWarmSeedFailureKind::TagProjectionInvalid,
        "Physical Tag ResourceUse has a non-tag value");
  return tag->value;
}

llvm::Error
projectRoutesAndTags(const ::loom::mapping::SpatialMappingView &mapping,
                     SpatialFullyRoutedSnapshot &snapshot,
                     SpatialMappingWarmSeedAccounting &accounting) {
  const FrozenSpatialPnrProblem &problem = *snapshot.problem;
  const auto nets = problem.transfers().logicalNets();
  const auto sinks = problem.transfers().logicalNetSinks();
  snapshot.routeSources.assign(nets.size(), getInvalidPnrIndex());
  snapshot.routeTagValueOffsets.reserve(nets.size() + 1);
  snapshot.routeTagValueOffsets.push_back(0);
  for (PnrIndex logicalNet = 0; logicalNet < nets.size(); ++logicalNet) {
    if (snapshot.registerFifoTransfers[logicalNet] != getInvalidPnrIndex()) {
      snapshot.routeTagValueOffsets.push_back(snapshot.routeTagValues.size());
      continue;
    }
    std::size_t mappingRouteOrdinal = 0;
    const auto *route =
        findRoute(mapping, nets[logicalNet].producer, &mappingRouteOrdinal);
    if (!route)
      return failure<bool>(
                 SpatialMappingWarmSeedFailureKind::RouteProjectionInvalid,
                 "external logical net has no unique parent RouteTree")
          .takeError();
    auto root =
        problem.routing().topology().endpointOrdinal(route->rootEndpoint);
    if (!root)
      return failure<bool>(
                 SpatialMappingWarmSeedFailureKind::RouteProjectionInvalid,
                 "RouteTree root is absent from the frozen Fabric")
          .takeError();
    snapshot.routeSources[logicalNet] = *root;
    std::vector<std::uint64_t> coveredNodes;
    for (PnrIndex localSink = 0; localSink < nets[logicalNet].sinkCount;
         ++localSink) {
      const PnrIndex sinkOrdinal = nets[logicalNet].sinkOffset + localSink;
      if (sinkOrdinal >= sinks.size())
        return failure<bool>(
                   SpatialMappingWarmSeedFailureKind::RouteProjectionInvalid,
                   "RouteTree sink slice is out of range")
            .takeError();
      const auto *sink = findRouteSink(*route, sinks[sinkOrdinal]);
      const auto *sinkNode =
          sink ? findRouteNode(*route, sink->nodeOrdinal) : nullptr;
      if (!sink || !sinkNode)
        return failure<bool>(
                   SpatialMappingWarmSeedFailureKind::RouteProjectionInvalid,
                   "RouteTree sink has no unique node")
            .takeError();
      auto sinkEndpoint =
          problem.routing().topology().endpointOrdinal(sinkNode->endpoint);
      if (!sinkEndpoint)
        return failure<bool>(
                   SpatialMappingWarmSeedFailureKind::RouteProjectionInvalid,
                   "RouteTree sink endpoint is absent from the frozen Fabric")
            .takeError();

      llvm::SmallVector<PnrIndex> reverseArcs;
      const auto *node = sinkNode;
      for (std::size_t depth = 0;; ++depth) {
        if (depth > route->nodes.size())
          return failure<bool>(
                     SpatialMappingWarmSeedFailureKind::RouteProjectionInvalid,
                     "parent RouteTree contains a cycle")
              .takeError();
        if (!llvm::is_contained(coveredNodes, node->ordinal))
          coveredNodes.push_back(node->ordinal);
        if (!node->parentOrdinal) {
          if (node->incomingTraversal || node->endpoint != route->rootEndpoint)
            return failure<bool>(SpatialMappingWarmSeedFailureKind::
                                     RouteProjectionInvalid,
                                 "parent RouteTree root relation is malformed")
                .takeError();
          break;
        }
        if (!node->incomingTraversal)
          return failure<bool>(
                     SpatialMappingWarmSeedFailureKind::RouteProjectionInvalid,
                     "parent RouteTree node lacks its incoming traversal")
              .takeError();
        const auto *parent = findRouteNode(*route, *node->parentOrdinal);
        if (!parent)
          return failure<bool>(
                     SpatialMappingWarmSeedFailureKind::RouteProjectionInvalid,
                     "parent RouteTree node names an absent parent")
              .takeError();
        auto arc = routeArc(problem, parent->endpoint, node->endpoint,
                            *node->incomingTraversal);
        if (!arc)
          return arc.takeError();
        reverseArcs.push_back(*arc);
        node = parent;
      }
      const std::size_t pathOffset = snapshot.arcPaths.size();
      snapshot.arcPaths.insert(snapshot.arcPaths.end(), reverseArcs.rbegin(),
                               reverseArcs.rend());
      snapshot.sinkPaths.push_back({logicalNet, localSink, *sinkEndpoint,
                                    pathOffset, reverseArcs.size()});
      accounting.routeArcs += reverseArcs.size();
    }
    if (coveredNodes.size() != route->nodes.size())
      return failure<bool>(
                 SpatialMappingWarmSeedFailureKind::RouteProjectionInvalid,
                 "parent RouteTree contains a node outside every sink path")
          .takeError();
    ++accounting.routeTrees;

    std::size_t segmentCount = 0;
    for (const auto &segment : mapping.physicalTagSegments())
      if (segment.routeTreeOrdinal == mappingRouteOrdinal)
        ++segmentCount;
    for (std::size_t local = 0; local < segmentCount; ++local) {
      const ::loom::mapping::SpatialPhysicalTagSegmentView *selected = nullptr;
      for (const auto &segment : mapping.physicalTagSegments())
        if (segment.routeTreeOrdinal == mappingRouteOrdinal &&
            segment.segmentOrdinal == local) {
          if (selected)
            return failure<bool>(
                       SpatialMappingWarmSeedFailureKind::TagProjectionInvalid,
                       "Physical Tag segment ordinal is repeated")
                .takeError();
          selected = &segment;
        }
      if (!selected)
        return failure<bool>(
                   SpatialMappingWarmSeedFailureKind::TagProjectionInvalid,
                   "Physical Tag segment ordinal is absent")
            .takeError();
      auto tag = mappedTag(mapping, *selected);
      if (!tag)
        return tag.takeError();
      snapshot.routeTagValues.emplace_back(std::move(*tag));
      ++accounting.physicalTagSegments;
    }
    snapshot.routeTagValueOffsets.push_back(snapshot.routeTagValues.size());
  }
  return llvm::Error::success();
}

} // namespace

llvm::Expected<VerifiedSpatialMappingWarmSeed>
projectFinalizedSpatialMappingWarmSeed(
    const ::loom::mapping::FinalizedSpatialMapping &parent,
    FrozenSpatialPnrProblemHandle problem) {
  if (!problem)
    return failure<VerifiedSpatialMappingWarmSeed>(
        SpatialMappingWarmSeedFailureKind::OwnerMismatch,
        "warm-seed FrozenSpatialPnrProblem owner is null");
  const auto &mapping = parent.view();
  if (mapping.dataflowIdentity() != problem->dataflowIdentity() ||
      mapping.techMappingIdentity() != problem->techMappingIdentity() ||
      mapping.fabricIdentity() != problem->fabricIdentity())
    return failure<VerifiedSpatialMappingWarmSeed>(
        SpatialMappingWarmSeedFailureKind::OwnerMismatch,
        "parent SpatialMapping and warm problem have different D/T/F owners");

  auto computeBindings = projectComputeBindings(mapping, *problem);
  if (!computeBindings)
    return computeBindings.takeError();
  auto memoryBindings = projectMemoryPlacements(mapping, *problem);
  if (!memoryBindings)
    return memoryBindings.takeError();
  auto memory = projectMemorySelections(mapping, *problem, *memoryBindings);
  if (!memory)
    return memory.takeError();
  auto local = projectLocalDispositions(mapping, *problem);
  if (!local)
    return local.takeError();
  auto roots = projectAttachments(mapping, *problem, *computeBindings,
                                  *memoryBindings, *local);
  if (!roots)
    return roots.takeError();

  SpatialMappingWarmSeedAccounting accounting;
  accounting.computeBindings = computeBindings->size();
  accounting.memoryBindings = memoryBindings->size();
  accounting.memoryOperationPlans = memory->operationPlans.size();
  accounting.logicalMemoryBindings = memory->logicalBindings.size();
  accounting.memoryUseDispatches = memory->useDispatches.size();
  accounting.memoryExposureSelections = memory->exposureSelections.size();
  accounting.portAttachments =
      roots->portAttachments.size() - roots->privatePorts;
  accounting.graphBoundaryAttachments =
      roots->graphBoundaryAttachments.size() - roots->privateBoundaries;
  accounting.canonicalPrivatePortAttachments = roots->privatePorts;
  accounting.canonicalPrivateGraphBoundaryAttachments =
      roots->privateBoundaries;
  accounting.registerFifoTransfers =
      llvm::count_if(*local, [](PnrIndex selected) {
        return selected != getInvalidPnrIndex();
      });
  accounting.relationAssignmentAttempts = roots->relationAssignmentAttempts;

  SpatialFullyRoutedSnapshot projected;
  projected.problem = problem;
  projected.computeBindings = std::move(*computeBindings);
  projected.memoryBindings = std::move(*memoryBindings);
  projected.portAttachments = std::move(roots->portAttachments);
  projected.graphBoundaryAttachments =
      std::move(roots->graphBoundaryAttachments);
  projected.memoryOperationPlans = std::move(memory->operationPlans);
  projected.logicalMemoryBindings = std::move(memory->logicalBindings);
  projected.memoryUseDispatches = std::move(memory->useDispatches);
  projected.memoryExposureSelections = std::move(memory->exposureSelections);
  projected.registerFifoTransfers = std::move(*local);
  if (llvm::Error error = projectRoutesAndTags(mapping, projected, accounting))
    return std::move(error);

  auto candidate = SpatialCandidateState::materializeFullyRouted(projected);
  if (!candidate)
    return failure<VerifiedSpatialMappingWarmSeed>(
        SpatialMappingWarmSeedFailureKind::CandidateVerificationFailed,
        "warm parent Candidate materialization failed: " +
            llvm::toString(candidate.takeError()));
  if (llvm::Error error = (*candidate)->verify())
    return failure<VerifiedSpatialMappingWarmSeed>(
        SpatialMappingWarmSeedFailureKind::CandidateVerificationFailed,
        "warm parent Candidate cold verification failed: " +
            llvm::toString(std::move(error)));

  std::vector<const RouteTreeState *> routes;
  std::vector<llvm::ArrayRef<std::optional<llvm::APInt>>> tags;
  routes.reserve(problem->transfers().logicalNets().size());
  tags.reserve(problem->transfers().logicalNets().size());
  for (PnrIndex logicalNet = 0;
       logicalNet < problem->transfers().logicalNets().size(); ++logicalNet) {
    routes.push_back(&(*candidate)->routeTree(logicalNet));
    tags.push_back((*candidate)->tagValues(logicalNet));
  }
  auto same = spatialMappingSelectionEqualsCandidate(mapping, **candidate,
                                                     routes, tags);
  if (!same)
    return failure<VerifiedSpatialMappingWarmSeed>(
        SpatialMappingWarmSeedFailureKind::SelectionMismatch,
        "warm Candidate selection comparison could not be established: " +
            llvm::toString(same.takeError()));
  if (!*same)
    return failure<VerifiedSpatialMappingWarmSeed>(
        SpatialMappingWarmSeedFailureKind::SelectionMismatch,
        "warm Candidate does not reproduce every parent Mapping selection");

  auto canonicalSnapshot = (*candidate)->snapshotFullyRouted();
  if (!canonicalSnapshot)
    return failure<VerifiedSpatialMappingWarmSeed>(
        SpatialMappingWarmSeedFailureKind::CandidateVerificationFailed,
        "verified warm Candidate cannot be snapshotted: " +
            llvm::toString(canonicalSnapshot.takeError()));
  return VerifiedSpatialMappingWarmSeed(
      mapping.identity(), std::move(*canonicalSnapshot), accounting);
}

} // namespace loom::pnr
