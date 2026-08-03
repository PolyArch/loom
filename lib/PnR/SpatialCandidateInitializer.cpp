#include "PnR/SpatialCandidateInitializer.h"

#include "InitializerRelationSolver.h"
#include "SpatialBindingRelationModel.h"

#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

using namespace loom::pnr;

namespace {

llvm::Error initializerError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("invalid Spatial candidate initialization: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

const FrozenSpatialMemoryDispatchDomain *
dispatchDomain(const FrozenSpatialPnrProblem &problem,
               llvm::ArrayRef<SpatialMemoryBindingSelection> memoryBindings,
               const FrozenSpatialMemoryRootedUse &use) {
  const auto &realizations = problem.realizations();
  if (use.actor >= realizations.memoryActors().size())
    return nullptr;
  const PnrIndex realization =
      realizations.memoryActorRealizations()[use.actor];
  if (realization >= memoryBindings.size())
    return nullptr;
  const auto &owner = realizations.memoryRealizations()[realization];
  if (use.actor < owner.actorOffset ||
      use.actor - owner.actorOffset >= owner.actorCount)
    return nullptr;
  const PnrIndex placement = memoryBindings[realization].placement;
  const auto offsets = problem.memory().memoryPlacementDomainOffsets();
  if (placement + 1 >= offsets.size())
    return nullptr;
  const PnrIndex domain = offsets[placement] + use.actor - owner.actorOffset;
  if (domain >= offsets[placement + 1] ||
      domain >= problem.memory().dispatchDomains().size())
    return nullptr;
  return &problem.memory().dispatchDomains()[domain];
}

bool admitsRegion(const FrozenSpatialMemoryIndex &memory,
                  const FrozenSpatialMemoryDispatchOption &option,
                  std::uint64_t ordinal) {
  const auto regions = memory.dispatchServiceRegionOrdinals().slice(
      option.serviceRegionOffset, option.serviceRegionCount);
  return std::binary_search(regions.begin(), regions.end(), ordinal);
}

std::optional<PnrIndex>
matchingDispatch(const FrozenSpatialPnrProblem &problem,
                 llvm::ArrayRef<SpatialMemoryBindingSelection> memoryBindings,
                 const FrozenSpatialMemoryRootedUse &use,
                 const FrozenSpatialMemoryBindingTargetOption *bindingTarget) {
  const auto *domain = dispatchDomain(problem, memoryBindings, use);
  if (!domain)
    return std::nullopt;
  const auto &memory = problem.memory();
  for (PnrIndex optionOrdinal = domain->optionOffset;
       optionOrdinal != domain->optionOffset + domain->optionCount;
       ++optionOrdinal) {
    const auto &option = memory.dispatchOptions()[optionOrdinal];
    if (!bindingTarget) {
      if (!std::holds_alternative<::loom::fabric::LocalMemoryServiceRef>(
              option.target))
        return optionOrdinal;
      continue;
    }
    if (const auto *region =
            std::get_if<::loom::fabric::FabricMemoryServiceRegionRef>(
                &bindingTarget->target)) {
      const auto *local =
          std::get_if<::loom::fabric::LocalMemoryServiceRef>(&option.target);
      if (local && local->underlying() == region->service &&
          admitsRegion(memory, option, region->ordinal))
        return optionOrdinal;
      continue;
    }
    if (std::holds_alternative<::loom::fabric::ManagerEndpointRef>(
            option.target))
      return optionOrdinal;
  }
  return std::nullopt;
}

} // namespace

llvm::Expected<SpatialCandidateStateHandle>
loom::pnr::createCanonicalSpatialCandidate(
    FrozenSpatialPnrProblemHandle problem) {
  if (!problem)
    return initializerError("FrozenSpatialPnrProblem owner is null");

  const detail::SpatialBindingRelationModel &bindingRelations =
      problem->bindingRelations();
  if (const auto deferred = bindingRelations.deferredProjection())
    return initializerError(
        "hard equality or disjointness for projection '" +
        ::mapping::stringifySpatialConstraintProjection(*deferred) +
        "' requires its owning decision model");

  detail::InitializerRelationSolver relationSolver(
      bindingRelations.relations());
  auto relationChoices = relationSolver.solveCanonical(
      problem->config()
          .policy()
          .search.initializer.assignmentAttemptLimitPerSeed);
  if (!relationChoices)
    return relationChoices.takeError();

  const FrozenSpatialRealizationIndex &realizations = problem->realizations();
  const FrozenSpatialPortIndex &ports = problem->ports();
  const FrozenSpatialHandshakeIndex &handshake = problem->handshake();

  std::vector<SpatialComputeBindingSelection> computeBindings;
  computeBindings.reserve(realizations.computeRealizations().size());
  for (PnrIndex realization = 0;
       realization < realizations.computeRealizations().size(); ++realization) {
    const auto choices = bindingRelations.computeChoices(realization);
    const PnrIndex selected = relationChoices->choices[realization];
    if (selected >= choices.size())
      return initializerError(
          "relation solver returned a foreign compute choice");
    computeBindings.push_back(
        {choices[selected].placement, choices[selected].instructionContext});
  }

  std::vector<SpatialMemoryBindingSelection> memoryBindings;
  memoryBindings.reserve(realizations.memoryRealizations().size());
  for (PnrIndex realization = 0;
       realization < realizations.memoryRealizations().size(); ++realization) {
    const auto choices = bindingRelations.memoryChoices(realization);
    const PnrIndex selected =
        relationChoices
            ->choices[bindingRelations.computeDecisionCount() + realization];
    if (selected >= choices.size())
      return initializerError(
          "relation solver returned a foreign memory choice");
    memoryBindings.push_back({choices[selected].placement});
  }

  std::vector<PnrIndex> portAttachments;
  portAttachments.reserve(ports.portDemands().size());
  for (const FrozenSpatialPortDemand &demand : ports.portDemands()) {
    const PnrIndex placement =
        demand.kind == FrozenSpatialPortDemandKind::Compute
            ? computeBindings[demand.realization].placement
            : memoryBindings[demand.realization].placement;
    const PnrIndex ownerOffset =
        demand.kind == FrozenSpatialPortDemandKind::Compute
            ? realizations.computeRealizations()[demand.realization]
                  .placementOffset
            : realizations.memoryRealizations()[demand.realization]
                  .placementOffset;
    const PnrIndex localPlacement = placement - ownerOffset;
    if (localPlacement >= demand.placementDomainCount)
      return initializerError(
          "PortDemand has no domain for its canonical placement");
    const FrozenSpatialPortPlacementDomain &domain =
        ports.placementDomains()[demand.placementDomainOffset + localPlacement];
    if (domain.attachmentOptionCount == 0)
      return initializerError("PortDemand has an empty attachment domain");
    portAttachments.push_back(domain.attachmentOptionOffset);
  }

  std::vector<PnrIndex> graphBoundaryAttachments;
  graphBoundaryAttachments.reserve(ports.graphBoundaries().size());
  for (const FrozenSpatialGraphBoundary &boundary : ports.graphBoundaries()) {
    if (boundary.attachmentOptionCount == 0)
      return initializerError("graph boundary has an empty attachment domain");
    graphBoundaryAttachments.push_back(boundary.attachmentOptionOffset);
  }

  std::vector<PnrIndex> memoryOperationPlans(realizations.memoryActors().size(),
                                             getInvalidPnrIndex());
  for (PnrIndex realizationOrdinal = 0;
       realizationOrdinal < realizations.memoryRealizations().size();
       ++realizationOrdinal) {
    const FrozenSpatialMemoryRealization &realization =
        realizations.memoryRealizations()[realizationOrdinal];
    const PnrIndex placement = memoryBindings[realizationOrdinal].placement;
    const PnrIndex domainOffset =
        handshake.memoryPlacementDomainOffsets()[placement];
    for (PnrIndex localActor = 0; localActor < realization.actorCount;
         ++localActor) {
      const FrozenSpatialMemoryOperationHandshakeDomain &domain =
          handshake.memoryOperationDomains()[domainOffset + localActor];
      if (domain.planCount == 0)
        return initializerError("memory actor has an empty operation domain");
      memoryOperationPlans[realization.actorOffset + localActor] =
          domain.planOffset;
    }
  }

  const FrozenSpatialMemoryIndex &memory = problem->memory();
  std::vector<SpatialLogicalMemoryBindingSelection> logicalMemoryBindings(
      memory.logicalBindings().size());
  std::vector<PnrIndex> memoryUseDispatches(memory.rootedUses().size(),
                                            getInvalidPnrIndex());
  std::vector<std::uint64_t> targetNextOffset(memory.bindingTargets().size(),
                                              0);
  for (PnrIndex binding = 0; binding < memory.logicalBindings().size();
       ++binding) {
    const auto extent = memory.logicalBindings()[binding].staticExtentBytes;
    const auto uses =
        memory.bindingUses().slice(memory.bindingUseOffsets()[binding],
                                   memory.bindingUseOffsets()[binding + 1] -
                                       memory.bindingUseOffsets()[binding]);
    if (uses.empty())
      return initializerError("logical memory binding has no rooted use");
    bool selected = false;
    for (PnrIndex targetOrdinal = 0;
         targetOrdinal < memory.bindingTargets().size(); ++targetOrdinal) {
      const auto &target = memory.bindingTargets()[targetOrdinal];
      const bool local =
          std::holds_alternative<::loom::fabric::FabricMemoryServiceRegionRef>(
              target.target);
      if (local &&
          (!extent || targetNextOffset[targetOrdinal] > target.sizeBytes ||
           *extent > target.sizeBytes - targetNextOffset[targetOrdinal]))
        continue;
      std::vector<std::pair<PnrIndex, PnrIndex>> dispatches;
      dispatches.reserve(uses.size());
      bool complete = true;
      for (PnrIndex useOrdinal : uses) {
        auto dispatch = matchingDispatch(
            *problem, memoryBindings, memory.rootedUses()[useOrdinal], &target);
        if (!dispatch) {
          complete = false;
          break;
        }
        dispatches.emplace_back(useOrdinal, *dispatch);
      }
      if (!complete)
        continue;
      logicalMemoryBindings[binding] = {
          targetOrdinal, local ? targetNextOffset[targetOrdinal] : 0};
      for (const auto &[useOrdinal, dispatch] : dispatches)
        memoryUseDispatches[useOrdinal] = dispatch;
      if (local)
        targetNextOffset[targetOrdinal] += *extent;
      selected = true;
      break;
    }
    if (!selected)
      return initializerError(
          "logical memory has no jointly compatible binding and dispatch");
  }

  for (PnrIndex useOrdinal = 0; useOrdinal < memory.rootedUses().size();
       ++useOrdinal) {
    if (memoryUseDispatches[useOrdinal] != getInvalidPnrIndex())
      continue;
    const auto &use = memory.rootedUses()[useOrdinal];
    if (use.logicalBinding)
      return initializerError("addressed memory use was not assigned");
    auto dispatch = matchingDispatch(*problem, memoryBindings, use, nullptr);
    if (!dispatch)
      return initializerError("fence use has no consistency dispatch");
    memoryUseDispatches[useOrdinal] = *dispatch;
  }

  return SpatialCandidateState::create(
      std::move(problem), {computeBindings, memoryBindings, portAttachments,
                           graphBoundaryAttachments, memoryOperationPlans,
                           logicalMemoryBindings, memoryUseDispatches});
}
