#include "PnR/SpatialCanonicalSeed.h"

#include "InitializerRelationSolver.h"
#include "PnR/SpatialCandidateInitializer.h"
#include "PnR/SpatialRouteCostState.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>
#include <utility>

using namespace loom::pnr;

namespace {

SpatialPnrWorkLedgerView workLedger(SpatialPathFinderSeedWorkSummary &summary) {
  std::array<SpatialPnrWorkCounterRef, spatialPnrWorkKindCount> counters{};
  counters[static_cast<std::size_t>(SpatialPnrWorkKind::SeedAttempt)] = {
      &summary.plannedSeedAttempts, &summary.seedAttempts};
  counters[static_cast<std::size_t>(
      SpatialPnrWorkKind::InitializerAssignment)] = {
      &summary.plannedInitializerAssignmentAttempts,
      &summary.initializerAssignmentAttempts};
  counters[static_cast<std::size_t>(SpatialPnrWorkKind::EndpointExpansion)] = {
      &summary.plannedEndpointExpansions, &summary.endpointExpansions};
  counters[static_cast<std::size_t>(SpatialPnrWorkKind::NegotiationIteration)] =
      {&summary.plannedNegotiationIterations, &summary.negotiationIterations};
  return SpatialPnrWorkLedgerView(counters);
}

std::string errorMessage(const llvm::ErrorInfoBase &error) {
  std::string message;
  llvm::raw_string_ostream stream(message);
  error.log(stream);
  return message;
}

llvm::Error retainSeedFailure(llvm::Error error,
                              SpatialPnrWorkLedgerView ledger) {
  bool sawCompletedFailure = false;
  bool sawInternalFailure = false;
  llvm::Error retained = llvm::handleErrors(
      std::move(error),
      [&](std::unique_ptr<detail::InitializerRelationSolveFailure> failure)
          -> llvm::Error {
        sawCompletedFailure = true;
        return llvm::Error(std::move(failure));
      },
      [&](std::unique_ptr<EndpointRouteSearchFailure> failure) -> llvm::Error {
        if (failure->kind() == EndpointRouteSearchFailureKind::Invalid)
          sawInternalFailure = true;
        else
          sawCompletedFailure = true;
        return llvm::Error(std::move(failure));
      },
      [&](std::unique_ptr<RoutingNegotiationError> failure) -> llvm::Error {
        if (failure->kind() ==
            RoutingNegotiationError::Kind::ArithmeticOverflow)
          sawCompletedFailure = true;
        else
          sawInternalFailure = true;
        return llvm::Error(std::move(failure));
      },
      [&](std::unique_ptr<SpatialPathFinderClosureFailure> failure)
          -> llvm::Error {
        // An interrupted closure completed no owner boundary, so the seed
        // attempt stays planned and unconsumed.
        if (failure->kind() != SpatialPathFinderClosureFailure::Kind::Interrupted)
          sawCompletedFailure = true;
        return llvm::Error(std::move(failure));
      },
      [&](const llvm::ErrorInfoBase &failure) -> llvm::Error {
        sawInternalFailure = true;
        return llvm::make_error<llvm::StringError>(
            errorMessage(failure), failure.convertToErrorCode());
      });
  if (sawCompletedFailure && !sawInternalFailure)
    if (llvm::Error consume = ledger.consume(SpatialPnrWorkKind::SeedAttempt))
      retained = llvm::joinErrors(std::move(retained), std::move(consume));
  return retained;
}

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
      [&](std::unique_ptr<SpatialPathFinderClosureFailure> failure)
          -> llvm::Error {
        if (failure->kind() == SpatialPathFinderClosureFailure::Kind::Interrupted)
          return llvm::Error(std::move(failure));
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
    llvm::ArrayRef<RouteCost> evaluationPriorities,
    ExecutionControlView executionControl) {
  workSummary = {};
  const SpatialPnrWorkLedgerView ledger = workLedger(workSummary);
  if (llvm::Error error = ledger.plan(SpatialPnrWorkKind::SeedAttempt))
    return std::move(error);
  std::uint64_t initializerAssignmentAttempts = 0;
  auto initialized = createSpatialCandidateInitializerAttempt(
      std::move(problem), attemptOrdinal, initializerAssignmentAttempts,
      ledger);
  if (!initialized)
    return retainSeedFailure(initialized.takeError(), ledger);
  if (initializerAssignmentAttempts !=
      workSummary.initializerAssignmentAttempts)
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "Spatial initializer counters disagree with the work ledger");
  const SpatialCandidateInitializerPreference initializerPreference =
      initialized->preference;
  SpatialCandidateStateHandle candidate = std::move(initialized->candidate);

  SpatialCandidateScratch candidateScratch;
  if (llvm::Error error = candidateScratch.prepare(candidate->problem()))
    return std::move(error);

  auto costs = SpatialRouteCostState::create(*candidate);
  if (!costs)
    return costs.takeError();

  SpatialPathFinderRouterScratch router;
  if (llvm::Error error =
          router.prepare(candidate->problem(), ledger, executionControl))
    return std::move(error);

  const ResolvedPnrRoutingPolicy &policy =
      candidate->problem().config().policy().search.routing;
  auto routing = router.routeToClosure(
      *candidate, candidateScratch, *costs,
      {policy.endpointExpansionLimit, policy.negotiationIterationLimit,
       policy.noProgressIterationLimit, policy.noProgressTrendWindow},
      evaluationPriorities,
      SpatialRoutingClosureRequirement::PolicyAdmittedTemporary);
  if (routing &&
      (workSummary.endpointExpansions != router.endpointExpansionCount() ||
       workSummary.negotiationIterations != router.negotiationIterationCount()))
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "Spatial seed routing counters disagree with the work ledger");
  if (!routing) {
    const bool admitsUnrouted = llvm::is_contained(
        candidate->problem().config().policy().temporaryViolations.admitted,
        ResolvedPnrViolationKind::UnroutedObligation);
    if (!admitsUnrouted)
      return retainSeedFailure(routing.takeError(), ledger);
    auto retained = retainRolledBackInitializer(routing.takeError());
    if (!retained)
      return retainSeedFailure(retained.takeError(), ledger);
    if (!*retained)
      return llvm::createStringError(
          std::make_error_code(std::errc::invalid_argument),
          "initial Spatial routing failure has no classification");
  }
  if (llvm::Error error = candidate->verify())
    return std::move(error);
  if (llvm::Error error = ledger.consume(SpatialPnrWorkKind::SeedAttempt))
    return std::move(error);
  return SpatialPathFinderSeed{std::move(candidate), attemptOrdinal,
                               initializerPreference};
}

llvm::Expected<SpatialPathFinderSeed>
loom::pnr::createCanonicalPathFinderSpatialSeed(
    FrozenSpatialPnrProblemHandle problem,
    SpatialPathFinderSeedWorkSummary &workSummary,
    llvm::ArrayRef<RouteCost> evaluationPriorities) {
  return createPathFinderSpatialSeed(std::move(problem), 0, workSummary,
                                     evaluationPriorities);
}
