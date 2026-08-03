#include "PnR/SpatialAnnealingSearch.h"

#include "llvm/Support/Error.h"

#include <limits>
#include <system_error>
#include <utility>

using namespace loom;
using namespace loom::pnr;

namespace {

llvm::Error searchError(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "invalid Spatial annealing search: %s", message.str().c_str());
}

llvm::Error addCount(std::uint64_t &target, std::uint64_t amount,
                     llvm::StringRef subject) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - target)
    return searchError(subject + " count overflows u64");
  target += amount;
  return llvm::Error::success();
}

template <typename T> std::size_t retainedBytes(const std::vector<T> &values) {
  return values.capacity() * sizeof(T);
}

} // namespace

llvm::Expected<bool>
SpatialAnnealingSearchScratch::consumeTransitionFailure(llvm::Error failure) {
  bool consumed = false;
  llvm::Error unhandled = llvm::handleErrors(
      std::move(failure),
      [&](const SpatialActionTransitionFailure &) -> llvm::Error {
        consumed = true;
        return llvm::Error::success();
      });
  if (unhandled)
    return std::move(unhandled);
  return consumed;
}

llvm::Expected<SpatialAnnealingStatistics>
SpatialAnnealingSearchScratch::run(SpatialCandidateState &candidate,
                                   std::uint64_t seedAttemptOrdinal) {
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  const ResolvedPnrPolicyConfig &policy = problem.config().policy();
  if (seedAttemptOrdinal >= policy.search.initializer.seedAttemptCount)
    return searchError("seed attempt ordinal is outside the fixed slot set");
  if (llvm::Error error = actionDomain_.prepare(problem))
    return std::move(error);
  if (llvm::Error error = actionExecutor_.prepare(candidate))
    return std::move(error);

  const ResolvedPnrAnnealingPolicy &annealing = policy.search.annealing;
  if (annealing.calibrationProposalCount >
      positiveCalibrationDeltas_.max_size())
    return searchError("calibration sample capacity exceeds host size_t");
  positiveCalibrationDeltas_.clear();
  positiveCalibrationDeltas_.reserve(
      static_cast<std::size_t>(annealing.calibrationProposalCount));

  SpatialAnnealingStatistics statistics;
  statistics.calibrationProposalSlots = annealing.calibrationProposalCount;
  DeterministicPnrRandomStream calibrationStream =
      DeterministicPnrRandomStream::create(policy.determinism.masterSeed,
                                           seedAttemptOrdinal,
                                           PnrRandomStreamPurpose::Calibration);
  for (std::uint64_t slot = 0; slot < annealing.calibrationProposalCount;
       ++slot) {
    if (llvm::Error error = actionDomain_.rebuild(candidate))
      return std::move(error);
    auto action = proposeSpatialAction(policy.search.actionProposal,
                                       actionDomain_.view(), calibrationStream);
    if (!action)
      return action.takeError();
    if (!*action)
      continue;

    auto probe = actionExecutor_.probe(candidate, **action);
    if (!probe) {
      auto consumed = consumeTransitionFailure(probe.takeError());
      if (!consumed)
        return consumed.takeError();
      if (!*consumed)
        return searchError("Action failure had no failure classification");
      if (llvm::Error error =
              addCount(statistics.calibrationTransitionFailureCount, 1,
                       "calibration transition failure"))
        return std::move(error);
      continue;
    }
    if (llvm::Error error =
            addCount(statistics.calibrationProbeCount, 1, "calibration probe"))
      return std::move(error);
    if (probe->energyDifference().sign ==
        dse::ObjectiveDifferenceSign::Positive)
      positiveCalibrationDeltas_.push_back(probe->energyDifference().magnitude);
    if (llvm::Error error = probe->discard())
      return std::move(error);
  }

  auto initialTemperature =
      calibrateAnnealingTemperature(annealing, positiveCalibrationDeltas_);
  if (!initialTemperature)
    return initialTemperature.takeError();
  statistics.initialTemperature = *initialTemperature;
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
    if (llvm::Error error = actionDomain_.rebuild(candidate))
      return std::move(error);
    auto proposalCount = annealingProposalsPerLevel(
        annealing, actionDomain_.movableDecisionCount());
    if (!proposalCount)
      return proposalCount.takeError();
    if (llvm::Error error = addCount(statistics.temperatureLevelCount, 1,
                                     "annealing temperature level"))
      return std::move(error);
    if (schedule->isFinalLevel())
      if (llvm::Error error = addCount(statistics.minimumTemperatureLevelCount,
                                       1, "minimum-temperature level"))
        return std::move(error);
    if (llvm::Error error = addCount(statistics.annealingProposalSlots,
                                     *proposalCount, "annealing proposal slot"))
      return std::move(error);

    for (std::uint64_t slot = 0; slot < *proposalCount; ++slot) {
      if (llvm::Error error = actionDomain_.rebuild(candidate))
        return std::move(error);
      auto action = proposeSpatialAction(policy.search.actionProposal,
                                         actionDomain_.view(), proposalStream);
      if (!action)
        return action.takeError();
      if (!*action)
        continue;

      auto probe = actionExecutor_.probe(candidate, **action);
      if (!probe) {
        auto consumed = consumeTransitionFailure(probe.takeError());
        if (!consumed)
          return consumed.takeError();
        if (!*consumed)
          return searchError("Action failure had no failure classification");
        if (llvm::Error error =
                addCount(statistics.annealingTransitionFailureCount, 1,
                         "annealing transition failure"))
          return std::move(error);
        continue;
      }
      if (llvm::Error error =
              addCount(statistics.annealingProbeCount, 1, "annealing probe"))
        return std::move(error);
      auto resolution =
          probe->resolve(schedule->temperature(), acceptanceStream);
      if (!resolution)
        return resolution.takeError();
      std::uint64_t &count = resolution->accepted
                                 ? statistics.acceptedActionCount
                                 : statistics.rejectedActionCount;
      if (llvm::Error error = addCount(
              count, 1,
              resolution->accepted ? "accepted Action" : "rejected Action"))
        return std::move(error);
    }
  } while (schedule->advanceAfterCompletedLevel());

  if (statistics.minimumTemperatureLevelCount != 1)
    return searchError(
        "annealing schedule did not execute one minimum-temperature level");
  if (llvm::Error error = candidate.verify())
    return std::move(error);
  return statistics;
}

std::size_t SpatialAnnealingSearchScratch::retainedStorageBytes() const {
  return actionDomain_.retainedStorageBytes() +
         actionExecutor_.retainedStorageBytes() +
         retainedBytes(positiveCalibrationDeltas_);
}
