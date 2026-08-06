#include "PnR/SpatialCandidateInitializer.h"

#include "InitializerChoiceOrder.h"
#include "InitializerRelationSolver.h"
#include "SpatialBindingRelationModel.h"
#include "SpatialMemoryConstraintModel.h"

#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

using namespace loom::pnr;

namespace {

using loom::pnr::detail::InitializerRelationSolveFailure;
using loom::pnr::detail::InitializerRelationSolveFailureKind;

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

void appendMatchingDispatches(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<SpatialMemoryBindingSelection> memoryBindings,
    const FrozenSpatialMemoryRootedUse &use,
    const FrozenSpatialMemoryBindingTargetOption *bindingTarget,
    std::vector<PnrIndex> &choices) {
  const auto *domain = dispatchDomain(problem, memoryBindings, use);
  if (!domain)
    return;
  const auto &memory = problem.memory();
  for (PnrIndex optionOrdinal = domain->optionOffset;
       optionOrdinal != domain->optionOffset + domain->optionCount;
       ++optionOrdinal) {
    const auto &option = memory.dispatchOptions()[optionOrdinal];
    if (!bindingTarget) {
      if (!std::holds_alternative<::loom::fabric::LocalMemoryServiceRef>(
              option.target))
        choices.push_back(optionOrdinal);
      continue;
    }
    if (const auto *region =
            std::get_if<::loom::fabric::FabricMemoryServiceRegionRef>(
                &bindingTarget->target)) {
      const auto *local =
          std::get_if<::loom::fabric::LocalMemoryServiceRef>(&option.target);
      if (local && local->underlying() == region->service &&
          admitsRegion(memory, option, region->ordinal))
        choices.push_back(optionOrdinal);
      continue;
    }
    if (std::holds_alternative<::loom::fabric::ManagerEndpointRef>(
            option.target))
      choices.push_back(optionOrdinal);
  }
}

bool exposureOptionMatches(
    const FrozenSpatialMemoryBindingTargetOption &bindingTarget,
    const FrozenSpatialMemoryExposureOption &option) {
  if (const auto *region =
          std::get_if<::loom::fabric::FabricMemoryServiceRegionRef>(
              &bindingTarget.target)) {
    const auto *local =
        std::get_if<::loom::fabric::LocalMemoryServiceRef>(&option.target);
    return local && local->underlying() == region->service;
  }
  return std::holds_alternative<::loom::fabric::ManagerEndpointRef>(
      option.target);
}

llvm::Error initializerFailure(InitializerRelationSolveFailureKind kind,
                               const llvm::Twine &message) {
  return llvm::make_error<InitializerRelationSolveFailure>(
      kind, ("Spatial initializer " + message).str());
}

class SpatialInitializerAttemptBuilder final {
public:
  SpatialInitializerAttemptBuilder(
      const FrozenSpatialPnrProblem &problem,
      DeterministicPnrRandomStream *diversificationStream,
      std::uint64_t assignmentLimit, std::uint64_t assignmentAttempts,
      std::vector<SpatialComputeBindingSelection> computeBindings,
      std::vector<SpatialMemoryBindingSelection> memoryBindings,
      std::vector<PnrIndex> portAttachments,
      std::vector<PnrIndex> graphBoundaryAttachments)
      : problem_(problem), diversificationStream_(diversificationStream),
        assignmentLimit_(assignmentLimit),
        assignmentAttempts_(assignmentAttempts),
        computeBindings_(std::move(computeBindings)),
        memoryBindings_(std::move(memoryBindings)),
        portAttachments_(std::move(portAttachments)),
        graphBoundaryAttachments_(std::move(graphBoundaryAttachments)) {}

  llvm::Error build() {
    if (llvm::Error error = prepareDecisionInventory())
      return error;
    auto completed = search();
    if (!completed)
      return completed.takeError();
    if (!*completed)
      return initializerFailure(
          InitializerRelationSolveFailureKind::FixedRootInfeasible,
          "has no complete dependent-decision assignment");
    return llvm::Error::success();
  }

  SpatialCandidateInitialization initialization() const {
    return {computeBindings_,      memoryBindings_,
            portAttachments_,      graphBoundaryAttachments_,
            memoryOperationPlans_, logicalMemoryBindings_,
            memoryUseDispatches_,  memoryExposureSelections_};
  }

  std::uint64_t assignmentAttempts() const { return assignmentAttempts_; }

private:
  enum class DecisionKind : std::uint8_t {
    MemoryOperationPlan,
    LogicalMemoryBinding,
    MemoryUseDispatch,
    MemoryExposure,
  };

  struct DecisionRecord final {
    DecisionKind kind = DecisionKind::MemoryOperationPlan;
    PnrIndex index = 0;
    std::size_t choiceOffset = 0;
    PnrIndex choiceCapacity = 0;
  };

  const FrozenSpatialMemoryOperationHandshakeDomain *
  memoryPlanDomain(PnrIndex actor) const {
    const auto &realizations = problem_.realizations();
    if (actor >= realizations.memoryActors().size())
      return nullptr;
    const PnrIndex realization = realizations.memoryActorRealizations()[actor];
    const auto &owner = realizations.memoryRealizations()[realization];
    const PnrIndex placement = memoryBindings_[realization].placement;
    if (actor < owner.actorOffset ||
        actor - owner.actorOffset >= owner.actorCount)
      return nullptr;
    const PnrIndex domainOffset =
        problem_.handshake().memoryPlacementDomainOffsets()[placement];
    return &problem_.handshake().memoryOperationDomains()[domainOffset + actor -
                                                          owner.actorOffset];
  }

  llvm::Error appendDecision(DecisionKind kind, PnrIndex index,
                             PnrIndex choiceCapacity) {
    if (choiceCapacity >
        std::numeric_limits<std::size_t>::max() - choiceStorageSize_)
      return initializerError("dependent choice storage size overflows");
    decisions_.push_back({kind, index, choiceStorageSize_, choiceCapacity});
    choiceStorageSize_ += choiceCapacity;
    return llvm::Error::success();
  }

  llvm::Error prepareDecisionInventory() {
    const auto &ports = problem_.ports();
    const auto &realizations = problem_.realizations();
    const auto &memory = problem_.memory();

    if (portAttachments_.size() != ports.portDemands().size())
      return initializerError("root decision solver omitted PortAttachments");
    if (graphBoundaryAttachments_.size() != ports.graphBoundaries().size())
      return initializerError(
          "root decision solver omitted graph-boundary attachments");
    memoryOperationPlans_.assign(realizations.memoryActors().size(),
                                 getInvalidPnrIndex());
    logicalMemoryBindings_.assign(memory.logicalBindings().size(), {});
    for (auto &binding : logicalMemoryBindings_)
      binding.target = getInvalidPnrIndex();
    memoryUseDispatches_.assign(memory.rootedUses().size(),
                                getInvalidPnrIndex());
    memoryExposureSelections_.assign(memory.exposures().size(),
                                     getInvalidPnrIndex());

    for (PnrIndex actor = 0; actor < realizations.memoryActors().size();
         ++actor) {
      const auto *domain = memoryPlanDomain(actor);
      if (!domain)
        return initializerError(
            "memory actor has no domain for its selected placement");
      if (llvm::Error error = appendDecision(DecisionKind::MemoryOperationPlan,
                                             actor, domain->planCount))
        return error;
    }
    for (PnrIndex binding = 0; binding < memory.logicalBindings().size();
         ++binding) {
      auto capacity =
          problem_.memoryConstraints().logicalBindingChoiceCapacity(binding);
      if (!capacity)
        return capacity.takeError();
      if (llvm::Error error = appendDecision(DecisionKind::LogicalMemoryBinding,
                                             binding, *capacity))
        return error;
    }
    for (PnrIndex use = 0; use < memory.rootedUses().size(); ++use) {
      const auto *domain =
          dispatchDomain(problem_, memoryBindings_, memory.rootedUses()[use]);
      if (!domain)
        return initializerError(
            "memory use has no domain for its selected placement");
      if (llvm::Error error = appendDecision(DecisionKind::MemoryUseDispatch,
                                             use, domain->optionCount))
        return error;
    }
    for (PnrIndex exposure = 0; exposure < memory.exposures().size();
         ++exposure) {
      if (memory.exposureOptions().size() > getPnrIndexMax())
        return initializerError("memory exposure domain is too large");
      if (llvm::Error error = appendDecision(
              DecisionKind::MemoryExposure, exposure,
              static_cast<PnrIndex>(memory.exposureOptions().size())))
        return error;
    }

    canonicalChoices_.resize(choiceStorageSize_);
    choiceOrder_.resize(choiceStorageSize_);
    choiceFenwick_.resize(choiceStorageSize_);
    logicalMemoryChoices_.resize(choiceStorageSize_);
    assignmentJournal_.reserve(decisions_.size());
    compatibilityChoices_.reserve(memory.dispatchOptions().size());
    return llvm::Error::success();
  }

  llvm::Error consumeAssignmentAttempt() {
    if (assignmentAttempts_ == assignmentLimit_)
      return initializerFailure(InitializerRelationSolveFailureKind::WorkLimit,
                                "exhausted its assignment work limit");
    ++assignmentAttempts_;
    return llvm::Error::success();
  }

  bool
  targetSupportsBinding(PnrIndex binding,
                        const FrozenSpatialMemoryBindingTargetOption &target) {
    const auto &memory = problem_.memory();
    const auto uses =
        memory.bindingUses().slice(memory.bindingUseOffsets()[binding],
                                   memory.bindingUseOffsets()[binding + 1] -
                                       memory.bindingUseOffsets()[binding]);
    for (PnrIndex use : uses) {
      compatibilityChoices_.clear();
      appendMatchingDispatches(problem_, memoryBindings_,
                               memory.rootedUses()[use], &target,
                               compatibilityChoices_);
      if (compatibilityChoices_.empty())
        return false;
    }
    const auto exposures = memory.bindingExposures().slice(
        memory.bindingExposureOffsets()[binding],
        memory.bindingExposureOffsets()[binding + 1] -
            memory.bindingExposureOffsets()[binding]);
    for (PnrIndex exposure : exposures) {
      (void)exposure;
      bool supported = false;
      for (const auto &option : memory.exposureOptions())
        supported |= exposureOptionMatches(target, option);
      if (!supported)
        return false;
    }
    return !uses.empty() || !exposures.empty();
  }

  bool decisionAssigned(const DecisionRecord &decision) const {
    switch (decision.kind) {
    case DecisionKind::MemoryOperationPlan:
      return memoryOperationPlans_[decision.index] != getInvalidPnrIndex();
    case DecisionKind::LogicalMemoryBinding:
      return logicalMemoryBindings_[decision.index].target !=
             getInvalidPnrIndex();
    case DecisionKind::MemoryUseDispatch:
      return memoryUseDispatches_[decision.index] != getInvalidPnrIndex();
    case DecisionKind::MemoryExposure:
      return memoryExposureSelections_[decision.index] != getInvalidPnrIndex();
    }
    llvm_unreachable("unknown Spatial initializer decision kind");
  }

  bool decisionActive(const DecisionRecord &decision) const {
    const auto &memory = problem_.memory();
    switch (decision.kind) {
    case DecisionKind::LogicalMemoryBinding:
      return decision.index == 0 ||
             logicalMemoryBindings_[decision.index - 1].target !=
                 getInvalidPnrIndex();
    case DecisionKind::MemoryUseDispatch: {
      const auto binding = memory.rootedUses()[decision.index].logicalBinding;
      return !binding ||
             logicalMemoryBindings_[*binding].target != getInvalidPnrIndex();
    }
    case DecisionKind::MemoryExposure:
      return logicalMemoryBindings_[memory.exposures()[decision.index]
                                        .logicalBinding]
                 .target != getInvalidPnrIndex();
    default:
      return true;
    }
  }

  llvm::Expected<PnrIndex> fillChoices(const DecisionRecord &decision) {
    auto choices = llvm::MutableArrayRef(canonicalChoices_)
                       .slice(decision.choiceOffset, decision.choiceCapacity);
    PnrIndex count = 0;
    switch (decision.kind) {
    case DecisionKind::MemoryOperationPlan: {
      const auto *domain = memoryPlanDomain(decision.index);
      if (!domain)
        return initializerError("memory operation domain disappeared");
      for (PnrIndex local = 0; local < domain->planCount; ++local)
        choices[count++] = domain->planOffset + local;
      break;
    }
    case DecisionKind::LogicalMemoryBinding: {
      const auto &memory = problem_.memory();
      auto values = llvm::MutableArrayRef(logicalMemoryChoices_)
                        .slice(decision.choiceOffset, decision.choiceCapacity);
      auto generated =
          problem_.memoryConstraints().collectLogicalBindingChoices(
              decision.index, logicalMemoryBindings_, values);
      if (!generated)
        return generated.takeError();
      for (PnrIndex choice = 0; choice < *generated; ++choice) {
        const auto value = values[choice];
        if (value.target >= memory.bindingTargets().size())
          return initializerError(
              "memory constraint owner produced a foreign target");
        if (!targetSupportsBinding(decision.index,
                                   memory.bindingTargets()[value.target]))
          continue;
        values[count] = value;
        choices[count] = count;
        ++count;
      }
      break;
    }
    case DecisionKind::MemoryUseDispatch: {
      const auto &memory = problem_.memory();
      const auto &use = memory.rootedUses()[decision.index];
      const auto *domain = dispatchDomain(problem_, memoryBindings_, use);
      if (!domain)
        return initializerError("memory dispatch domain disappeared");
      const FrozenSpatialMemoryBindingTargetOption *target = nullptr;
      if (use.logicalBinding)
        target =
            &memory.bindingTargets()[logicalMemoryBindings_[*use.logicalBinding]
                                         .target];
      compatibilityChoices_.clear();
      appendMatchingDispatches(problem_, memoryBindings_, use, target,
                               compatibilityChoices_);
      for (PnrIndex option : compatibilityChoices_)
        choices[count++] = option;
      break;
    }
    case DecisionKind::MemoryExposure: {
      const auto &memory = problem_.memory();
      const auto &exposure = memory.exposures()[decision.index];
      const auto &target =
          memory.bindingTargets()
              [logicalMemoryBindings_[exposure.logicalBinding].target];
      for (PnrIndex option = 0; option < memory.exposureOptions().size();
           ++option)
        if (exposureOptionMatches(target, memory.exposureOptions()[option]))
          choices[count++] = option;
      break;
    }
    }
    if (count > decision.choiceCapacity)
      return initializerError("dependent choice domain exceeds its frozen cap");
    return count;
  }

  void assignDecision(std::size_t decisionOrdinal, PnrIndex choice) {
    const auto &decision = decisions_[decisionOrdinal];
    switch (decision.kind) {
    case DecisionKind::MemoryOperationPlan:
      memoryOperationPlans_[decision.index] = choice;
      break;
    case DecisionKind::LogicalMemoryBinding: {
      logicalMemoryBindings_[decision.index] =
          logicalMemoryChoices_[decision.choiceOffset + choice];
      break;
    }
    case DecisionKind::MemoryUseDispatch:
      memoryUseDispatches_[decision.index] = choice;
      break;
    case DecisionKind::MemoryExposure:
      memoryExposureSelections_[decision.index] = choice;
      break;
    }
    assignmentJournal_.push_back(decisionOrdinal);
  }

  void rollback(std::size_t journalMark) {
    while (assignmentJournal_.size() > journalMark) {
      const std::size_t ordinal = assignmentJournal_.back();
      assignmentJournal_.pop_back();
      const auto &decision = decisions_[ordinal];
      switch (decision.kind) {
      case DecisionKind::MemoryOperationPlan:
        memoryOperationPlans_[decision.index] = getInvalidPnrIndex();
        break;
      case DecisionKind::LogicalMemoryBinding: {
        logicalMemoryBindings_[decision.index] = {};
        logicalMemoryBindings_[decision.index].target = getInvalidPnrIndex();
        break;
      }
      case DecisionKind::MemoryUseDispatch:
        memoryUseDispatches_[decision.index] = getInvalidPnrIndex();
        break;
      case DecisionKind::MemoryExposure:
        memoryExposureSelections_[decision.index] = getInvalidPnrIndex();
        break;
      }
    }
  }

  llvm::Expected<bool> search() {
    while (true) {
      bool allAssigned = true;
      bool propagated = false;
      for (std::size_t ordinal = 0; ordinal < decisions_.size(); ++ordinal) {
        const auto &decision = decisions_[ordinal];
        if (decisionAssigned(decision))
          continue;
        allAssigned = false;
        if (!decisionActive(decision))
          continue;
        auto count = fillChoices(decision);
        if (!count)
          return count.takeError();
        if (*count == 0)
          return false;
        if (*count == 1) {
          assignDecision(ordinal, canonicalChoices_[decision.choiceOffset]);
          propagated = true;
          break;
        }
      }
      if (allAssigned) {
        if (llvm::Error error =
                problem_.memoryConstraints().verify(logicalMemoryBindings_)) {
          llvm::consumeError(std::move(error));
          return false;
        }
        return true;
      }
      if (!propagated)
        break;
    }

    std::size_t selected = std::numeric_limits<std::size_t>::max();
    PnrIndex selectedCount = getInvalidPnrIndex();
    for (std::size_t ordinal = 0; ordinal < decisions_.size(); ++ordinal) {
      const auto &decision = decisions_[ordinal];
      if (decisionAssigned(decision) || !decisionActive(decision))
        continue;
      auto count = fillChoices(decision);
      if (!count)
        return count.takeError();
      if (*count == 0)
        return false;
      if (*count < selectedCount) {
        selected = ordinal;
        selectedCount = *count;
      }
    }
    if (selected == std::numeric_limits<std::size_t>::max())
      return initializerError("dependent decision prerequisites form a cycle");

    const auto &decision = decisions_[selected];
    auto count = fillChoices(decision);
    if (!count)
      return count.takeError();
    auto canonical =
        llvm::ArrayRef(canonicalChoices_).slice(decision.choiceOffset, *count);
    auto order = llvm::MutableArrayRef(choiceOrder_)
                     .slice(decision.choiceOffset, *count);
    if (llvm::Error error = detail::buildInitializerChoiceOrder(
            canonical, diversificationStream_, order,
            llvm::MutableArrayRef(choiceFenwick_)
                .slice(decision.choiceOffset, *count)))
      return std::move(error);

    for (PnrIndex choice : order) {
      if (llvm::Error error = consumeAssignmentAttempt())
        return std::move(error);
      const std::size_t journalMark = assignmentJournal_.size();
      assignDecision(selected, choice);
      auto completed = search();
      if (!completed)
        return completed.takeError();
      if (*completed)
        return true;
      rollback(journalMark);
    }
    return false;
  }

  const FrozenSpatialPnrProblem &problem_;
  DeterministicPnrRandomStream *diversificationStream_ = nullptr;
  std::uint64_t assignmentLimit_ = 0;
  std::uint64_t assignmentAttempts_ = 0;
  std::vector<SpatialComputeBindingSelection> computeBindings_;
  std::vector<SpatialMemoryBindingSelection> memoryBindings_;
  std::vector<PnrIndex> portAttachments_;
  std::vector<PnrIndex> graphBoundaryAttachments_;
  std::vector<PnrIndex> memoryOperationPlans_;
  std::vector<SpatialLogicalMemoryBindingSelection> logicalMemoryBindings_;
  std::vector<PnrIndex> memoryUseDispatches_;
  std::vector<PnrIndex> memoryExposureSelections_;
  std::vector<DecisionRecord> decisions_;
  std::vector<PnrIndex> canonicalChoices_;
  std::vector<PnrIndex> choiceOrder_;
  std::vector<PnrIndex> choiceFenwick_;
  std::vector<PnrIndex> compatibilityChoices_;
  std::vector<SpatialLogicalMemoryBindingSelection> logicalMemoryChoices_;
  std::vector<std::size_t> assignmentJournal_;
  std::size_t choiceStorageSize_ = 0;
};

} // namespace

llvm::Expected<SpatialCandidateInitializerAttempt>
loom::pnr::createSpatialCandidateInitializerAttempt(
    FrozenSpatialPnrProblemHandle problem, std::uint32_t attemptOrdinal,
    std::uint64_t &assignmentAttempts) {
  assignmentAttempts = 0;
  if (!problem)
    return initializerError("FrozenSpatialPnrProblem owner is null");
  const auto &policy = problem->config().policy();
  if (attemptOrdinal >= policy.search.initializer.seedAttemptCount)
    return initializerError("initializer attempt ordinal is out of range");

  const detail::SpatialBindingRelationModel &bindingRelations =
      problem->bindingRelations();
  if (const auto deferred = bindingRelations.deferredProjection())
    return initializerError(
        "hard equality or disjointness for projection '" +
        ::mapping::stringifySpatialConstraintProjection(*deferred) +
        "' requires its owning decision model");

  detail::InitializerRelationSolver relationSolver(
      bindingRelations.relations());
  std::optional<DeterministicPnrRandomStream> diversificationStream;
  if (attemptOrdinal != 0)
    diversificationStream.emplace(DeterministicPnrRandomStream::create(
        policy.determinism.masterSeed, attemptOrdinal,
        PnrRandomStreamPurpose::InitializerDiversification));
  auto relationChoices =
      diversificationStream
          ? relationSolver.solveDiversified(
                policy.search.initializer.assignmentAttemptLimitPerSeed,
                *diversificationStream)
          : relationSolver.solveCanonical(
                policy.search.initializer.assignmentAttemptLimitPerSeed);
  assignmentAttempts = relationSolver.assignmentAttempts();
  if (!relationChoices)
    return relationChoices.takeError();

  const FrozenSpatialRealizationIndex &realizations = problem->realizations();
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
  portAttachments.reserve(problem->ports().portDemands().size());
  for (PnrIndex demand = 0; demand < problem->ports().portDemands().size();
       ++demand) {
    const auto choices = bindingRelations.portAttachmentChoices(demand);
    const PnrIndex selected =
        relationChoices
            ->choices[bindingRelations.portDecisionOffset() + demand];
    if (selected >= choices.size())
      return initializerError(
          "relation solver returned a foreign PortAttachment choice");
    portAttachments.push_back(choices[selected]);
  }

  std::vector<PnrIndex> graphBoundaryAttachments;
  graphBoundaryAttachments.reserve(problem->ports().graphBoundaries().size());
  for (PnrIndex boundary = 0;
       boundary < problem->ports().graphBoundaries().size(); ++boundary) {
    const auto choices =
        bindingRelations.graphBoundaryAttachmentChoices(boundary);
    const PnrIndex selected =
        relationChoices
            ->choices[bindingRelations.graphBoundaryDecisionOffset() +
                      boundary];
    if (selected >= choices.size())
      return initializerError(
          "relation solver returned a foreign graph-boundary choice");
    graphBoundaryAttachments.push_back(choices[selected]);
  }

  SpatialInitializerAttemptBuilder builder(
      *problem, diversificationStream ? &*diversificationStream : nullptr,
      policy.search.initializer.assignmentAttemptLimitPerSeed,
      relationChoices->assignmentAttempts, std::move(computeBindings),
      std::move(memoryBindings), std::move(portAttachments),
      std::move(graphBoundaryAttachments));
  llvm::Error buildError = builder.build();
  assignmentAttempts = builder.assignmentAttempts();
  if (buildError)
    return std::move(buildError);
  auto candidate =
      SpatialCandidateState::create(problem, builder.initialization());
  if (!candidate)
    return candidate.takeError();
  return SpatialCandidateInitializerAttempt{std::move(*candidate)};
}

llvm::Expected<SpatialCandidateStateHandle>
loom::pnr::createCanonicalSpatialCandidate(
    FrozenSpatialPnrProblemHandle problem) {
  std::uint64_t assignmentAttempts = 0;
  auto attempt = createSpatialCandidateInitializerAttempt(std::move(problem), 0,
                                                          assignmentAttempts);
  if (!attempt)
    return attempt.takeError();
  return std::move(attempt->candidate);
}
