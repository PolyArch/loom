#include "PnR/System/SystemAnnealingSearch.h"

#include "PnR/InitializerRelationSolver.h"
#include "PnR/System/SystemPnrProblem.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <optional>
#include <utility>

using namespace loom;
using namespace loom::pnr;

char SystemActionTransitionFailure::ID;

void SystemActionTransitionFailure::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code SystemActionTransitionFailure::convertToErrorCode() const {
  return std::make_error_code(
      kind_ == SystemActionTransitionFailureKind::WorkLimit
          ? std::errc::resource_unavailable_try_again
          : std::errc::invalid_argument);
}

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "invalid System annealing search: " + message);
}

llvm::Error checkedAdd(std::uint64_t amount, std::uint64_t &target,
                       llvm::StringRef subject) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - target)
    return invalid(subject + " count overflows u64");
  target += amount;
  return llvm::Error::success();
}

llvm::Expected<std::uint64_t>
checkedMultiply(std::uint64_t lhs, std::uint64_t rhs, llvm::StringRef subject) {
  if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)
    return invalid(subject + " count overflows u64");
  return lhs * rhs;
}

std::string errorMessage(const llvm::ErrorInfoBase &error) {
  std::string message;
  llvm::raw_string_ostream stream(message);
  error.log(stream);
  return stream.str();
}

struct ProbeResult final {
  SystemCandidateStateHandle candidate;
  dse::ObjectiveVector objective;
  dse::ObjectiveSignedDifference energyDifference;
};

struct ProbeAccounting final {
  std::uint64_t assignmentAttempts = 0;
  std::uint64_t endpointExpansions = 0;
};

std::vector<PnrIndex>
dependencyClosureFixedChoices(const SystemCandidateState &current,
                              SystemExecutionBindingAction action) {
  const FrozenSystemPnrProblem &problem = current.problem();
  const detail::InitializerRelationModel &relations =
      problem.initializerRelations();
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

llvm::Expected<ProbeResult>
probeAction(const SystemCandidateStateHandle &current,
            const dse::ObjectiveVector &currentObjective,
            SystemExecutionBindingAction action, ProbeAccounting &accounting) {
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
          accounting.assignmentAttempts = failure.assignmentAttempts();
          accounting.endpointExpansions = failure.endpointExpansions();
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
            return invalid("Action dependency closure failed internally: " +
                           llvm::Twine(errorMessage(failure)));
          }
          llvm_unreachable("unknown System initialization failure kind");
        });
    return std::move(translated);
  }
  accounting.assignmentAttempts = initialized->assignmentAttempts;
  accounting.endpointExpansions = initialized->endpointExpansions;
  auto objective = problem.objectiveProgram().evaluate(*initialized->state);
  if (!objective)
    return objective.takeError();
  auto difference = problem.objectiveProgram().selectedEnergyDifference(
      *objective, currentObjective);
  if (!difference)
    return difference.takeError();
  return ProbeResult{std::move(initialized->state), std::move(*objective),
                     *difference};
}

void rebuildActions(const SystemCandidateState &candidate,
                    std::vector<SystemExecutionBindingAction> &actions,
                    std::uint64_t &movableDecisionCount) {
  actions.clear();
  movableDecisionCount = 0;
  const FrozenSystemPnrProblem &problem = candidate.problem();
  for (PnrIndex decision = 0; decision < problem.threadDecisions().size();
       ++decision) {
    const std::size_t offset = actions.size();
    for (PnrIndex choice = 0;
         choice < problem.threadChoiceCatalogOrdinals(decision).size();
         ++choice)
      if (choice != candidate.threadChoice(decision))
        actions.push_back({decision, choice});
    movableDecisionCount += actions.size() != offset;
  }
  const PnrIndex threadCount = problem.threadDecisions().size();
  for (PnrIndex decision = 0; decision < problem.graphDecisions().size();
       ++decision) {
    const std::size_t offset = actions.size();
    for (PnrIndex choice = 0;
         choice < problem.graphChoiceCatalogOrdinals(decision).size(); ++choice)
      if (choice != candidate.graphChoice(decision))
        actions.push_back({threadCount + decision, choice});
    movableDecisionCount += actions.size() != offset;
  }
}

llvm::Expected<bool> consumeTransitionFailure(llvm::Error error) {
  bool consumed = false;
  llvm::Error remaining = llvm::handleErrors(
      std::move(error),
      [&](const SystemActionTransitionFailure &) -> llvm::Error {
        consumed = true;
        return llvm::Error::success();
      });
  if (remaining)
    return std::move(remaining);
  return consumed;
}

} // namespace

llvm::Expected<SystemAnnealingStatistics>
SystemAnnealingSearchScratch::run(SystemCandidateStateHandle &candidate,
                                  std::uint64_t seedAttemptOrdinal) {
  const FrozenSystemPnrProblem &problem = candidate->problem();
  const ResolvedPnrPolicyConfig &policy = problem.config().policy();
  if (seedAttemptOrdinal >= policy.search.initializer.seedAttemptCount)
    return invalid("seed attempt ordinal is outside the fixed slot set");
  auto evaluated = problem.objectiveProgram().evaluate(*candidate);
  if (!evaluated)
    return evaluated.takeError();
  dse::ObjectiveVector currentObjective = std::move(*evaluated);

  SystemAnnealingStatistics statistics;
  const ResolvedPnrAnnealingPolicy &annealing = policy.search.annealing;
  statistics.calibrationProposalSlots = annealing.calibrationProposalCount;
  if (annealing.calibrationProposalCount >
      positiveCalibrationDeltas_.max_size())
    return invalid("calibration sample capacity exceeds host size_t");
  positiveCalibrationDeltas_.clear();
  positiveCalibrationDeltas_.reserve(
      static_cast<std::size_t>(annealing.calibrationProposalCount));
  DeterministicPnrRandomStream calibrationStream =
      DeterministicPnrRandomStream::create(policy.determinism.masterSeed,
                                           seedAttemptOrdinal,
                                           PnrRandomStreamPurpose::Calibration);
  for (std::uint64_t slot = 0; slot < annealing.calibrationProposalCount;
       ++slot) {
    std::uint64_t movable = 0;
    rebuildActions(*candidate, actions_, movable);
    auto action = proposeSystemAction(policy.search.actionProposal, {actions_},
                                      calibrationStream);
    if (!action)
      return action.takeError();
    if (!*action)
      continue;
    ProbeAccounting work;
    auto probe = probeAction(candidate, currentObjective, **action, work);
    if (llvm::Error error =
            checkedAdd(work.assignmentAttempts, statistics.assignmentAttempts,
                       "calibration assignment attempt"))
      return std::move(error);
    if (llvm::Error error =
            checkedAdd(work.endpointExpansions, statistics.endpointExpansions,
                       "calibration endpoint expansion"))
      return std::move(error);
    if (!probe) {
      auto consumed = consumeTransitionFailure(probe.takeError());
      if (!consumed)
        return consumed.takeError();
      if (!*consumed)
        return invalid("Action failure had no classification");
      continue;
    }
    if (probe->energyDifference.sign == dse::ObjectiveDifferenceSign::Positive)
      positiveCalibrationDeltas_.push_back(probe->energyDifference.magnitude);
  }

  auto initialTemperature =
      calibrateAnnealingTemperature(annealing, positiveCalibrationDeltas_);
  if (!initialTemperature)
    return initialTemperature.takeError();
  auto schedule =
      AnnealingTemperatureSchedule::create(annealing, *initialTemperature);
  if (!schedule)
    return schedule.takeError();
  DeterministicPnrRandomStream proposalStream =
      DeterministicPnrRandomStream::create(
          policy.determinism.masterSeed, seedAttemptOrdinal,
          PnrRandomStreamPurpose::ActionProposal);
  DeterministicPnrRandomStream acceptanceStream =
      DeterministicPnrRandomStream::create(policy.determinism.masterSeed,
                                           seedAttemptOrdinal,
                                           PnrRandomStreamPurpose::Acceptance);
  do {
    std::uint64_t movable = 0;
    rebuildActions(*candidate, actions_, movable);
    auto proposalCount = annealingProposalsPerLevel(annealing, movable);
    if (!proposalCount)
      return proposalCount.takeError();
    auto movableSlots =
        checkedMultiply(annealing.proposalsPerMovableDecision, movable,
                        "movable-decision proposal slot");
    if (!movableSlots)
      return movableSlots.takeError();
    if (llvm::Error error = checkedAdd(annealing.proposalsPerLevelBase,
                                       statistics.annealingBaseProposalSlots,
                                       "base proposal slot"))
      return std::move(error);
    if (llvm::Error error =
            checkedAdd(*movableSlots, statistics.annealingMovableProposalSlots,
                       "movable-decision proposal slot"))
      return std::move(error);
    for (std::uint64_t slot = 0; slot < *proposalCount; ++slot) {
      rebuildActions(*candidate, actions_, movable);
      auto action = proposeSystemAction(policy.search.actionProposal,
                                        {actions_}, proposalStream);
      if (!action)
        return action.takeError();
      if (!*action)
        continue;
      ProbeAccounting work;
      auto probe = probeAction(candidate, currentObjective, **action, work);
      if (llvm::Error error =
              checkedAdd(work.assignmentAttempts, statistics.assignmentAttempts,
                         "annealing assignment attempt"))
        return std::move(error);
      if (llvm::Error error =
              checkedAdd(work.endpointExpansions, statistics.endpointExpansions,
                         "annealing endpoint expansion"))
        return std::move(error);
      if (!probe) {
        auto consumed = consumeTransitionFailure(probe.takeError());
        if (!consumed)
          return consumed.takeError();
        if (!*consumed)
          return invalid("Action failure had no classification");
        continue;
      }
      auto accepted = acceptAnnealingDelta(
          probe->energyDifference, schedule->temperature(), acceptanceStream);
      if (!accepted)
        return accepted.takeError();
      if (*accepted) {
        candidate = std::move(probe->candidate);
        currentObjective = std::move(probe->objective);
        if (statistics.acceptedActionCount ==
            std::numeric_limits<std::uint64_t>::max())
          return invalid("accepted Action count overflows u64");
        ++statistics.acceptedActionCount;
      }
    }
  } while (schedule->advanceAfterCompletedLevel());

  if (llvm::Error error = candidate->verify())
    return std::move(error);
  return statistics;
}
