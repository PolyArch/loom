#include "PnR/SpatialCanonicalSeed.h"

#include "PnR/SpatialCandidateInitializer.h"
#include "PnR/SpatialRouteCostState.h"

#include "llvm/Support/Error.h"

#include <utility>

using namespace loom::pnr;

llvm::Expected<SpatialPathFinderSeed> loom::pnr::createPathFinderSpatialSeed(
    FrozenSpatialPnrProblemHandle problem, std::uint32_t attemptOrdinal,
    SpatialPathFinderSeedWorkSummary &workSummary,
    llvm::ArrayRef<RouteCost> evaluationPriorities) {
  workSummary = {};
  auto initialized = createSpatialCandidateInitializerAttempt(
      std::move(problem), attemptOrdinal,
      workSummary.initializerAssignmentAttempts);
  if (!initialized)
    return initialized.takeError();
  SpatialCandidateStateHandle candidate = std::move(initialized->candidate);

  SpatialCandidateScratch candidateScratch;
  if (llvm::Error error = candidateScratch.prepare(candidate->problem()))
    return std::move(error);

  auto costs = SpatialRouteCostState::create(*candidate);
  if (!costs)
    return costs.takeError();

  SpatialPathFinderRouterScratch router;
  if (llvm::Error error = router.prepare(candidate->problem()))
    return std::move(error);

  const ResolvedPnrRoutingPolicy &policy =
      candidate->problem().config().policy().search.routing;
  auto routing = router.routeToClosure(
      *candidate, candidateScratch, *costs,
      {policy.endpointExpansionLimit, policy.negotiationIterationLimit},
      evaluationPriorities);
  workSummary.endpointExpansions = router.endpointExpansionCount();
  workSummary.negotiationIterations = router.negotiationIterationCount();
  if (!routing)
    return routing.takeError();
  if (llvm::Error error = candidate->verify())
    return std::move(error);
  return SpatialPathFinderSeed{std::move(candidate), attemptOrdinal};
}

llvm::Expected<SpatialPathFinderSeed>
loom::pnr::createCanonicalPathFinderSpatialSeed(
    FrozenSpatialPnrProblemHandle problem,
    SpatialPathFinderSeedWorkSummary &workSummary,
    llvm::ArrayRef<RouteCost> evaluationPriorities) {
  return createPathFinderSpatialSeed(std::move(problem), 0, workSummary,
                                     evaluationPriorities);
}
