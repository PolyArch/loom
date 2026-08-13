#include "PnR/SpatialCanonicalSeed.h"

#include "PnR/SpatialCandidateInitializer.h"
#include "PnR/SpatialRouteCostState.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <utility>

using namespace loom::pnr;

namespace {

llvm::Expected<bool> retainRolledBackInitializer(llvm::Error error) {
  bool recoverable = false;
  llvm::Error unhandled = llvm::handleErrors(
      std::move(error),
      [&](std::unique_ptr<EndpointRouteSearchFailure> failure) -> llvm::Error {
        switch (failure->kind()) {
        case EndpointRouteSearchFailureKind::Unreachable:
        case EndpointRouteSearchFailureKind::WorkLimit:
          recoverable = true;
          return llvm::Error::success();
        case EndpointRouteSearchFailureKind::Invalid:
        case EndpointRouteSearchFailureKind::ArithmeticOverflow:
          return llvm::Error(std::move(failure));
        }
        llvm_unreachable("unknown endpoint route failure kind");
      },
      [&](const SpatialPathFinderClosureFailure &) -> llvm::Error {
        recoverable = true;
        return llvm::Error::success();
      });
  if (unhandled)
    return std::move(unhandled);
  return recoverable;
}

} // namespace

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
      {policy.endpointExpansionLimit, policy.negotiationIterationLimit,
       policy.noProgressIterationLimit, policy.noProgressTrendWindow},
      evaluationPriorities,
      SpatialRoutingClosureRequirement::PolicyAdmittedTemporary);
  workSummary.endpointExpansions = router.endpointExpansionCount();
  workSummary.negotiationIterations = router.negotiationIterationCount();
  if (!routing) {
    const bool admitsUnrouted = llvm::is_contained(
        candidate->problem().config().policy().temporaryViolations.admitted,
        ResolvedPnrViolationKind::UnroutedObligation);
    if (!admitsUnrouted)
      return routing.takeError();
    auto retained = retainRolledBackInitializer(routing.takeError());
    if (!retained)
      return retained.takeError();
    if (!*retained)
      return llvm::createStringError(
          std::make_error_code(std::errc::invalid_argument),
          "initial Spatial routing failure has no classification");
  }
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
