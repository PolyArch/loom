#include "PnR/SpatialCanonicalSeed.h"

#include "PnR/SpatialCandidateInitializer.h"
#include "PnR/SpatialRouteCostState.h"

#include "llvm/Support/Error.h"

#include <utility>

using namespace loom::pnr;

llvm::Expected<SpatialCanonicalSeed>
loom::pnr::createCanonicalPathFinderSpatialSeed(
    FrozenSpatialPnrProblemHandle problem,
    llvm::ArrayRef<RouteCost> evaluationPriorities) {
  auto candidate = createCanonicalSpatialCandidate(std::move(problem));
  if (!candidate)
    return candidate.takeError();

  SpatialCandidateScratch candidateScratch;
  if (llvm::Error error = candidateScratch.prepare((*candidate)->problem()))
    return std::move(error);

  auto costs = SpatialRouteCostState::create(**candidate);
  if (!costs)
    return costs.takeError();

  SpatialPathFinderRouterScratch router;
  if (llvm::Error error = router.prepare((*candidate)->problem()))
    return std::move(error);

  const ResolvedPnrRoutingPolicy &policy =
      (*candidate)->problem().config().policy().search.routing;
  auto routing = router.routeToClosure(
      **candidate, candidateScratch, *costs,
      {policy.endpointExpansionLimit, policy.negotiationIterationLimit},
      evaluationPriorities);
  if (!routing)
    return routing.takeError();
  if (llvm::Error error = (*candidate)->verify())
    return std::move(error);
  return SpatialCanonicalSeed{std::move(*candidate), *routing};
}
