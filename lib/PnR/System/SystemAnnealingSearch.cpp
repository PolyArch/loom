#include "PnR/System/SystemAnnealingSearch.h"

#include "PnR/System/SystemPnrProblem.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <system_error>
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
                       const llvm::Twine &subject) {
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

llvm::Error accountProbe(const SystemActionProbeAccounting &work,
                         SystemAnnealingStatistics &statistics,
                         llvm::StringRef scope) {
  if (llvm::Error error =
          checkedAdd(work.assignmentAttempts, statistics.assignmentAttempts,
                     scope + " assignment attempt"))
    return error;
  return checkedAdd(work.endpointExpansions, statistics.endpointExpansions,
                    scope + " endpoint expansion");
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
    if (llvm::Error error = actionDomain_.rebuild(*candidate))
      return std::move(error);
    auto action = proposeSystemAction(policy.search.actionProposal,
                                      actionDomain_.view(), calibrationStream);
    if (!action)
      return action.takeError();
    if (!*action)
      continue;
    SystemActionProbeAccounting work;
    auto probe = probeSystemAction(candidate, currentObjective, **action, work);
    if (llvm::Error error = accountProbe(work, statistics, "calibration"))
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
    if (llvm::Error error = actionDomain_.rebuild(*candidate))
      return std::move(error);
    const std::uint64_t movable = actionDomain_.movableDecisionCount();
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
      if (llvm::Error error = actionDomain_.rebuild(*candidate))
        return std::move(error);
      auto action = proposeSystemAction(policy.search.actionProposal,
                                        actionDomain_.view(), proposalStream);
      if (!action)
        return action.takeError();
      if (!*action)
        continue;
      SystemActionProbeAccounting work;
      auto probe =
          probeSystemAction(candidate, currentObjective, **action, work);
      if (llvm::Error error = accountProbe(work, statistics, "annealing"))
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
