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

std::vector<PnrIndex>
dependencyClosureFixedChoices(const SystemCandidateState &current,
                              SystemExecutionBindingAction action) {
  const detail::InitializerRelationModel &relations =
      current.problem().initializerRelations();
  std::vector<PnrIndex> fixed;
  fixed.reserve(relations.decisionCount());
  fixed.insert(fixed.end(), current.threadChoices().begin(),
               current.threadChoices().end());
  fixed.insert(fixed.end(), current.graphChoices().begin(),
               current.graphChoices().end());

  std::vector<std::uint8_t> released(relations.decisionCount(), 0);
  std::vector<PnrIndex> pending{action.decision};
  released[action.decision] = 1;
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
  fixed[action.decision] = action.choice;
  return fixed;
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

  std::vector<PnrIndex> fixed = dependencyClosureFixedChoices(*current, action);
  auto initialized = initializeSystemCandidateWithFixedChoices(
      current->problemHandle(), fixed);
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

llvm::Error translateMutationFailure(llvm::Error error) {
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
        return llvm::make_error<SystemActionTransitionFailure>(
            SystemActionTransitionFailureKind::WorkLimit,
            errorMessage(failure));
      });
}

llvm::Expected<SystemCandidateStateHandle>
executeTransport(const SystemCandidateStateHandle &current,
                 const SystemTransportRoutingAction &action,
                 std::uint64_t &endpointExpansions,
                 std::uint64_t &negotiationIterations) {
  auto candidate = detail::rebuildSystemCandidateRoutes(
      *current, action, endpointExpansions, negotiationIterations);
  if (!candidate)
    return translateMutationFailure(candidate.takeError());
  return candidate;
}

llvm::Expected<SystemCandidateStateHandle>
executeResource(const SystemCandidateStateHandle &current,
                const SystemResourceAllocationAction &action) {
  return std::visit(
      [&](const auto &value) -> llvm::Expected<SystemCandidateStateHandle> {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, SystemServiceTargetAction>)
          return detail::rebuildSystemCandidateWithServiceTarget(
              *current, value.context, value.choice);
        else if constexpr (std::is_same_v<T, SystemInstructionUsePatternAction>)
          return detail::rebuildSystemCandidateWithInstructionUsePattern(
              *current, value.use, value.choice);
        else
          return detail::rebuildSystemCandidateWithServiceUsePattern(
              *current, value.use, value.choice);
      },
      action);
}

} // namespace

llvm::Expected<SystemActionProbeResult>
loom::pnr::probeSystemAction(const SystemCandidateStateHandle &current,
                             const dse::ObjectiveVector &currentObjective,
                             const SystemMappingAction &action,
                             SystemActionProbeAccounting &accounting) {
  if (!current)
    return invalid("current candidate owner is null");
  accounting = {};
  auto candidate = std::visit(
      [&](const auto &value) -> llvm::Expected<SystemCandidateStateHandle> {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, SystemExecutionBindingAction>)
          return executeBinding(current, value, accounting.assignmentAttempts,
                                accounting.endpointExpansions,
                                accounting.negotiationIterations);
        else if constexpr (std::is_same_v<T, SystemTransportRoutingAction>)
          return executeTransport(current, value, accounting.endpointExpansions,
                                  accounting.negotiationIterations);
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
  return SystemActionProbeResult{std::move(*candidate), std::move(*objective),
                                 *difference};
}
