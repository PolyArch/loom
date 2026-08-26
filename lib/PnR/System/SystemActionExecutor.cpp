#include "PnR/System/SystemActionExecutor.h"

#include "SystemCandidateMutation.h"
#include "SystemCandidateServiceResolver.h"
#include "SystemNegotiatedRouter.h"

#include "PnR/EndpointRouter.h"
#include "PnR/InitializerRelationSolver.h"
#include "PnR/System/SystemAnnealingSearch.h"
#include "PnR/System/SystemPnrProblem.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <functional>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::pnr;

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "invalid System Action execution: " + message);
}

std::string errorMessage(const llvm::ErrorInfoBase &error) {
  std::string message;
  llvm::raw_string_ostream stream(message);
  error.log(stream);
  return message;
}

const SystemServiceRouteSelection *findRoute(const SystemCandidateState &state,
                                             PnrIndex leg) {
  const auto found = llvm::find_if(
      state.serviceRoutes(), [&](const SystemServiceRouteSelection &route) {
        return route.leg == leg;
      });
  return found == state.serviceRoutes().end() ? nullptr : &*found;
}

bool sameRoute(const SystemCandidateState &lhs, const SystemCandidateState &rhs,
               PnrIndex leg) {
  const SystemServiceRouteSelection *left = findRoute(lhs, leg);
  const SystemServiceRouteSelection *right = findRoute(rhs, leg);
  if (!left || !right || left->rootEndpoint != right->rootEndpoint ||
      left->nodeCount != right->nodeCount ||
      left->sinkCount != right->sinkCount)
    return false;
  const auto leftNodes =
      lhs.serviceRouteNodes().slice(left->nodeOffset, left->nodeCount);
  const auto rightNodes =
      rhs.serviceRouteNodes().slice(right->nodeOffset, right->nodeCount);
  for (const auto &[leftNode, rightNode] :
       llvm::zip_equal(leftNodes, rightNodes))
    if (leftNode.endpoint != rightNode.endpoint ||
        leftNode.parentNode != rightNode.parentNode ||
        leftNode.incomingTraversal != rightNode.incomingTraversal)
      return false;
  const auto leftSinks =
      lhs.serviceRouteSinks().slice(left->sinkOffset, left->sinkCount);
  const auto rightSinks =
      rhs.serviceRouteSinks().slice(right->sinkOffset, right->sinkCount);
  for (const auto &[leftSink, rightSink] :
       llvm::zip_equal(leftSinks, rightSinks))
    if (leftSink.terminal != rightSink.terminal ||
        leftSink.node != rightSink.node)
      return false;
  return true;
}

bool sameInstructionUse(const SystemInstructionResourceUseSelection &lhs,
                        const SystemInstructionResourceUseSelection &rhs) {
  return lhs.root == rhs.root && lhs.context == rhs.context &&
         lhs.pattern == rhs.pattern;
}

bool sameServiceUse(const SystemServiceResourceUseSelection &lhs,
                    const SystemServiceResourceUseSelection &rhs) {
  return lhs.context == rhs.context && lhs.subject == rhs.subject &&
         lhs.branch == rhs.branch && lhs.pattern == rhs.pattern;
}

template <typename Left, typename Right, typename Same>
void collectChangedOrdinals(Left left, Right right, Same same,
                            std::vector<PnrIndex> &changed) {
  const std::size_t count = std::max(left.size(), right.size());
  for (std::size_t ordinal = 0; ordinal < count; ++ordinal)
    if (ordinal >= left.size() || ordinal >= right.size() ||
        !same(left[ordinal], right[ordinal]))
      changed.push_back(static_cast<PnrIndex>(ordinal));
}

SystemActionMutationRecord
buildMutationRecord(const SystemCandidateState &before,
                    const SystemCandidateState &after,
                    const SystemMappingAction &action) {
  SystemActionMutationRecord result;
  result.domain = std::visit(
      [](const auto &typed) {
        using T = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<T, SystemExecutionBindingAction> ||
                      std::is_same_v<T, SystemExecutionBindingReopenAction>)
          return SystemActionMutationDomain::ExecutionBinding;
        else if constexpr (std::is_same_v<T, SystemTransportRoutingAction>)
          return SystemActionMutationDomain::TransportRouting;
        else
          return SystemActionMutationDomain::ResourceAllocation;
      },
      action);
  collectChangedOrdinals(before.threadChoices(), after.threadChoices(),
                         std::equal_to<PnrIndex>{}, result.threadDecisions);
  collectChangedOrdinals(before.graphChoices(), after.graphChoices(),
                         std::equal_to<PnrIndex>{}, result.graphDecisions);
  for (PnrIndex leg = 0; leg < before.problem().serviceLegs().size(); ++leg)
    if (!sameRoute(before, after, leg))
      result.serviceLegs.push_back(leg);
  collectChangedOrdinals(
      before.serviceTargets(), after.serviceTargets(),
      [](const auto &lhs, const auto &rhs) { return lhs == rhs; },
      result.serviceTargets);
  collectChangedOrdinals(before.instructionResourceUses(),
                         after.instructionResourceUses(), sameInstructionUse,
                         result.instructionResourceUses);
  collectChangedOrdinals(before.serviceResourceUses(),
                         after.serviceResourceUses(), sameServiceUse,
                         result.serviceResourceUses);
  result.capacityOveruseBefore = before.capacityOveruse();
  result.capacityOveruseAfter = after.capacityOveruse();
  result.recurrenceMinimumInitiationIntervalBefore =
      before.recurrenceTiming()
          .recurrenceMinimumInitiationIntervalCycles;
  result.recurrenceMinimumInitiationIntervalAfter =
      after.recurrenceTiming()
          .recurrenceMinimumInitiationIntervalCycles;
  result.resourceMinimumInitiationIntervalBefore =
      before.resourceMinimumInitiationIntervalCycles();
  result.resourceMinimumInitiationIntervalAfter =
      after.resourceMinimumInitiationIntervalCycles();
  result.transportBitCycleDemandBefore = before.transportBitCycleDemand();
  result.transportBitCycleDemandAfter = after.transportBitCycleDemand();
  result.progressBefore = before.progressClosure().kind;
  result.progressAfter = after.progressClosure().kind;
  return result;
}

llvm::Expected<std::vector<PnrIndex>>
dependencyClosureFixedChoices(const SystemCandidateState &current,
                              llvm::ArrayRef<PnrIndex> seeds) {
  const detail::InitializerRelationModel &relations =
      current.problem().initializerRelations();
  std::vector<PnrIndex> fixed;
  fixed.reserve(relations.decisionCount());
  fixed.insert(fixed.end(), current.threadChoices().begin(),
               current.threadChoices().end());
  fixed.insert(fixed.end(), current.graphChoices().begin(),
               current.graphChoices().end());

  std::vector<std::uint8_t> released(relations.decisionCount(), 0);
  std::vector<PnrIndex> pending;
  pending.reserve(seeds.size());
  for (PnrIndex seed : seeds) {
    if (seed >= relations.decisionCount())
      return invalid("binding closure seed names a foreign decision");
    if (released[seed])
      continue;
    released[seed] = 1;
    fixed[seed] = getInvalidPnrIndex();
    pending.push_back(seed);
  }
  const auto offsets = relations.decisionRelationOffsets();
  const auto incidence = relations.decisionRelations();
  for (std::size_t cursor = 0; cursor != pending.size(); ++cursor) {
    const PnrIndex decision = pending[cursor];
    for (PnrIndex offset = offsets[decision]; offset < offsets[decision + 1];
         ++offset) {
      const auto &relation = relations.relations()[incidence[offset]];
      for (const detail::InitializerRelationMember &member :
           relations.members(relation)) {
        if (released[member.decision])
          continue;
        released[member.decision] = 1;
        fixed[member.decision] = getInvalidPnrIndex();
        pending.push_back(member.decision);
      }
    }
  }
  return fixed;
}

llvm::Expected<SystemCandidateStateHandle> executeFixedBinding(
    const SystemCandidateStateHandle &current, llvm::ArrayRef<PnrIndex> fixed,
    std::uint64_t &assignmentAttempts, std::uint64_t &endpointExpansions,
    std::uint64_t &negotiationIterations) {
  const FrozenSystemPnrProblem &problem = current->problem();
  const PnrIndex threadCount = problem.threadDecisions().size();
  if (fixed.size() !=
      threadCount + static_cast<PnrIndex>(problem.graphDecisions().size()))
    return invalid("binding closure has the wrong decision width");
  SystemCandidateRouteSeed routeSeed{
      {current->serviceRoutes().begin(), current->serviceRoutes().end()},
      {current->serviceRouteNodes().begin(), current->serviceRouteNodes().end()},
      {current->serviceRouteSinks().begin(), current->serviceRouteSinks().end()},
      {}};
  for (PnrIndex leg = 0; leg < problem.serviceLegs().size(); ++leg) {
    const PnrIndex context = problem.serviceLegs()[leg].serviceContext;
    if (context >= problem.serviceContexts().size())
      return invalid("binding closure names a foreign service context");
    const auto &service = problem.serviceContexts()[context];
    const auto changed = [&](PnrIndex decision, PnrIndex currentChoice) {
      return decision != getInvalidPnrIndex() &&
             (decision >= fixed.size() || fixed[decision] == getInvalidPnrIndex() ||
              fixed[decision] != currentChoice);
    };
    bool reroute = false;
    if (service.threadDecision != getInvalidPnrIndex()) {
      if (service.threadDecision >= threadCount)
        return invalid("binding closure names a foreign thread decision");
      reroute |= changed(service.threadDecision,
                         current->threadChoice(service.threadDecision));
    }
    if (service.graphDecision != getInvalidPnrIndex()) {
      const PnrIndex graph = service.graphDecision;
      if (graph >= problem.graphDecisions().size())
        return invalid("binding closure names a foreign graph decision");
      reroute |= changed(threadCount + graph,
                         current->graphChoice(graph));
    }
    if (reroute)
      routeSeed.reroutedLegs.push_back(leg);
  }
  auto initialized = initializeSystemCandidateWithFixedChoicesAndRoutes(
      current->problemHandle(), fixed, routeSeed);
  if (!initialized) {
    llvm::Error translated = llvm::handleErrors(
        initialized.takeError(),
        [&](const SystemCandidateInitializationFailure &failure)
            -> llvm::Error {
          assignmentAttempts = failure.assignmentAttempts();
          endpointExpansions = failure.endpointExpansions();
          negotiationIterations = failure.negotiationIterations();
          switch (failure.kind()) {
          case SystemCandidateInitializationFailureKind::ProvenInfeasible:
            return llvm::make_error<SystemActionTransitionFailure>(
                SystemActionTransitionFailureKind::IntrinsicInvalid,
                errorMessage(failure));
          case SystemCandidateInitializationFailureKind::SemanticLimitReached:
            return llvm::make_error<SystemActionTransitionFailure>(
                SystemActionTransitionFailureKind::WorkLimit,
                errorMessage(failure));
          case SystemCandidateInitializationFailureKind::Internal:
            return invalid("dependency closure failed internally: " +
                           llvm::Twine(errorMessage(failure)));
          }
          llvm_unreachable("unknown System initialization failure kind");
        });
    return std::move(translated);
  }
  assignmentAttempts = initialized->assignmentAttempts;
  endpointExpansions = initialized->endpointExpansions;
  negotiationIterations = initialized->negotiationIterations;
  return std::move(initialized->state);
}

llvm::Expected<SystemCandidateStateHandle> executeBinding(
    const SystemCandidateStateHandle &current,
    SystemExecutionBindingAction action, std::uint64_t &assignmentAttempts,
    std::uint64_t &endpointExpansions, std::uint64_t &negotiationIterations) {
  const FrozenSystemPnrProblem &problem = current->problem();
  const std::size_t decisionCount =
      problem.threadDecisions().size() + problem.graphDecisions().size();
  if (action.decision >= decisionCount)
    return invalid("Action names a foreign execution decision");
  const std::size_t choiceCount =
      action.decision < problem.threadDecisions().size()
          ? problem.threadChoiceCatalogOrdinals(action.decision).size()
          : problem
                .graphChoiceCatalogOrdinals(action.decision -
                                            problem.threadDecisions().size())
                .size();
  if (action.choice >= choiceCount)
    return invalid("Action names a foreign execution choice");

  const std::array<PnrIndex, 1> seeds{action.decision};
  auto fixed = dependencyClosureFixedChoices(*current, seeds);
  if (!fixed)
    return fixed.takeError();
  (*fixed)[action.decision] = action.choice;
  return executeFixedBinding(current, *fixed, assignmentAttempts,
                             endpointExpansions, negotiationIterations);
}

llvm::Expected<SystemCandidateStateHandle>
executeBindingReopen(const SystemCandidateStateHandle &current,
                     const SystemExecutionBindingReopenAction &action,
                     std::uint64_t &assignmentAttempts,
                     std::uint64_t &endpointExpansions,
                     std::uint64_t &negotiationIterations) {
  const FrozenSystemPnrProblem &problem = current->problem();
  if (action.capacityCell >= problem.routingTopology().capacityCells().size())
    return invalid("binding reopen Action names a foreign capacity cell");
  if (action.graphDecisions.empty())
    return invalid("binding reopen Action has no graph decisions");
  std::vector<PnrIndex> seeds;
  seeds.reserve(action.graphDecisions.size());
  PnrIndex previous = getInvalidPnrIndex();
  const PnrIndex threadCount = problem.threadDecisions().size();
  for (PnrIndex graphDecision : action.graphDecisions) {
    if (graphDecision >= problem.graphDecisions().size())
      return invalid("binding reopen Action names a foreign graph decision");
    if (previous != getInvalidPnrIndex() && graphDecision <= previous)
      return invalid("binding reopen graph decisions are not canonical");
    previous = graphDecision;
    seeds.push_back(threadCount + graphDecision);
  }
  auto fixed = dependencyClosureFixedChoices(*current, seeds);
  if (!fixed)
    return fixed.takeError();
  return executeFixedBinding(current, *fixed, assignmentAttempts,
                             endpointExpansions, negotiationIterations);
}

llvm::Expected<std::optional<SystemUpstreamReopenWitness>>
projectUpstreamReopenWitness(
    const SystemCandidateState &candidate,
    const detail::SystemRoutingReopenWitness &witness) {
  SystemUpstreamReopenWitness result;
  result.capacityCell = witness.capacityCell;
  for (PnrIndex leg : witness.serviceLegs) {
    if (leg >= candidate.problem().serviceLegs().size())
      return invalid("routing reopen witness names a foreign service leg");
    const PnrIndex context =
        candidate.problem().serviceLegs()[leg].serviceContext;
    if (context >= candidate.problem().serviceContexts().size())
      return invalid("routing reopen witness has a foreign service context");
    const PnrIndex graphDecision =
        candidate.problem().serviceContexts()[context].graphDecision;
    if (graphDecision != getInvalidPnrIndex())
      result.graphDecisions.push_back(graphDecision);
  }
  llvm::sort(result.graphDecisions);
  result.graphDecisions.erase(
      std::unique(result.graphDecisions.begin(), result.graphDecisions.end()),
      result.graphDecisions.end());
  if (result.graphDecisions.empty())
    return std::optional<SystemUpstreamReopenWitness>();
  return std::optional<SystemUpstreamReopenWitness>(std::move(result));
}

llvm::Error translateMutationFailure(llvm::Error error,
                                     const SystemCandidateState &candidate) {
  return llvm::handleErrors(
      std::move(error),
      [&](const detail::SystemCandidateInfeasible &failure) -> llvm::Error {
        return llvm::make_error<SystemActionTransitionFailure>(
            SystemActionTransitionFailureKind::IntrinsicInvalid,
            errorMessage(failure));
      },
      [&](const EndpointRouteSearchFailure &failure) -> llvm::Error {
        switch (failure.kind()) {
        case EndpointRouteSearchFailureKind::Unreachable:
          return llvm::make_error<SystemActionTransitionFailure>(
              SystemActionTransitionFailureKind::IntrinsicInvalid,
              errorMessage(failure));
        case EndpointRouteSearchFailureKind::WorkLimit:
          return llvm::make_error<SystemActionTransitionFailure>(
              SystemActionTransitionFailureKind::WorkLimit,
              errorMessage(failure));
        case EndpointRouteSearchFailureKind::Invalid:
        case EndpointRouteSearchFailureKind::ArithmeticOverflow:
          return invalid("router failed internally: " +
                         llvm::Twine(errorMessage(failure)));
        }
        llvm_unreachable("unknown endpoint route failure kind");
      },
      [&](const detail::SystemRoutingClosureFailure &failure) -> llvm::Error {
        std::optional<SystemUpstreamReopenWitness> reopen;
        if (failure.reopenWitness()) {
          auto projected =
              projectUpstreamReopenWitness(candidate, *failure.reopenWitness());
          if (!projected)
            return projected.takeError();
          reopen = std::move(*projected);
        }
        return llvm::make_error<SystemActionTransitionFailure>(
            failure.kind() == detail::SystemRoutingClosureFailureKind::
                                  FixedTerminalCapacityCut
                ? SystemActionTransitionFailureKind::IntrinsicInvalid
                : SystemActionTransitionFailureKind::WorkLimit,
            errorMessage(failure), std::move(reopen));
      });
}

llvm::Expected<SystemCandidateStateHandle>
executeTransport(const SystemCandidateStateHandle &current,
                 const SystemTransportRoutingAction &action,
                 std::uint64_t &endpointExpansions,
                 std::uint64_t &negotiationIterations,
                 SystemActionExecutionContext context,
                 std::optional<SystemUpstreamReopenWitness> &reopenWitness) {
  std::optional<detail::SystemRoutingReopenWitness> routingWitness;
  auto candidate = detail::rebuildSystemCandidateRoutes(
      *current, action, endpointExpansions, negotiationIterations,
      context == SystemActionExecutionContext::FinalClosure, &routingWitness);
  if (!candidate)
    return translateMutationFailure(candidate.takeError(), *current);
  if (routingWitness) {
    auto projected = projectUpstreamReopenWitness(*current, *routingWitness);
    if (!projected)
      return projected.takeError();
    reopenWitness = std::move(*projected);
  }
  return candidate;
}

llvm::Expected<SystemCandidateStateHandle>
executeResource(const SystemCandidateStateHandle &current,
                const SystemResourceAllocationAction &action) {
  auto candidate = std::visit(
      [&](const auto &value) -> llvm::Expected<SystemCandidateStateHandle> {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, SystemServiceTargetAction>)
          return detail::rebuildSystemCandidateWithServiceTarget(
              *current, value.context, value.subject, value.choice);
        else if constexpr (std::is_same_v<T, SystemInstructionUsePatternAction>)
          return detail::rebuildSystemCandidateWithInstructionUsePattern(
              *current, value.use, value.choice);
        else
          return detail::rebuildSystemCandidateWithServiceUsePattern(
              *current, value.use, value.choice);
      },
      action);
  if (!candidate)
    return translateMutationFailure(candidate.takeError(), *current);
  return candidate;
}

} // namespace

llvm::Expected<SystemActionProbeResult>
loom::pnr::probeSystemAction(const SystemCandidateStateHandle &current,
                             const dse::ObjectiveVector &currentObjective,
                             const SystemMappingAction &action,
                             SystemActionProbeAccounting &accounting,
                             SystemActionExecutionContext context) {
  if (!current)
    return invalid("current candidate owner is null");
  if (context == SystemActionExecutionContext::FinalClosure) {
    const auto *transport = std::get_if<SystemTransportRoutingAction>(&action);
    if (!transport ||
        !std::holds_alternative<SystemGlobalRoutingAction>(*transport))
      return invalid("final closure requires one Global routing Action");
  }
  accounting = {};
  std::optional<SystemUpstreamReopenWitness> reopenWitness;
  auto candidate = std::visit(
      [&](const auto &value) -> llvm::Expected<SystemCandidateStateHandle> {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, SystemExecutionBindingAction>)
          return executeBinding(current, value, accounting.assignmentAttempts,
                                accounting.endpointExpansions,
                                accounting.negotiationIterations);
        else if constexpr (std::is_same_v<T,
                                          SystemExecutionBindingReopenAction>)
          return executeBindingReopen(
              current, value, accounting.assignmentAttempts,
              accounting.endpointExpansions, accounting.negotiationIterations);
        else if constexpr (std::is_same_v<T, SystemTransportRoutingAction>)
          return executeTransport(current, value, accounting.endpointExpansions,
                                  accounting.negotiationIterations, context,
                                  reopenWitness);
        else
          return executeResource(current, value);
      },
      action);
  if (!candidate)
    return candidate.takeError();
  auto objective = current->problem().objectiveProgram().evaluate(**candidate);
  if (!objective)
    return objective.takeError();
  auto difference =
      current->problem().objectiveProgram().selectedEnergyDifference(
          *objective, currentObjective);
  if (!difference)
    return difference.takeError();
  SystemActionMutationRecord mutation =
      buildMutationRecord(*current, **candidate, action);
  return SystemActionProbeResult{std::move(*candidate), std::move(*objective),
                                 *difference, std::move(mutation),
                                 std::move(reopenWitness)};
}
