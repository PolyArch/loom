#include "PnR/SpatialPathFinderRouter.h"

#include "Common/MappingDebugLog.h"
#include "PnR/MappingObjective.h"
#include "SpatialPathFinderRouterInternal.h"
#include "SpatialTagPressureDiagnostic.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace loom::pnr;
using loom::pnr::detail::classifyIterationFailure;
using loom::pnr::detail::encodeLogicalNetDetail;
using loom::pnr::detail::errorMessage;
using loom::pnr::detail::pathFinderError;

namespace {

enum class RankTrendTransition : std::uint8_t {
  Equal,
  Improved,
  Ineligible,
  Regressed,
};

template <typename T> std::size_t retainedBytes(const std::vector<T> &values) {
  return values.capacity() * sizeof(T);
}

llvm::Error rollbackIteration(SpatialMoveTransaction &move,
                              SpatialRouteCostState &costs,
                              llvm::Error failure) {
  move.rollback();
  if (llvm::Error reset = costs.resetFromCandidate())
    return llvm::joinErrors(std::move(failure), std::move(reset));
  return failure;
}

struct ProjectionDisposition final {
  bool violationsZero = true;
  bool temporaryAdmitted = false;
};

llvm::Expected<ProjectionDisposition>
classifyProjection(const SpatialCandidateState &candidate,
                   const SpatialCandidateRouteProjection &projection,
                   SpatialRoutingClosureRequirement closureRequirement) {
  ProjectionDisposition result;
  result.temporaryAdmitted =
      closureRequirement ==
      SpatialRoutingClosureRequirement::PolicyAdmittedTemporary;
  const auto admitted =
      candidate.problem().config().policy().temporaryViolations.admitted;
  for (std::uint32_t ordinal = 0;
       ordinal != loom::resolvedPnrViolationKindCount; ++ordinal) {
    const auto kind = static_cast<loom::ResolvedPnrViolationKind>(ordinal);
    auto value = spatialMappingViolationValue(candidate, projection, kind);
    if (!value)
      return value.takeError();
    if (*value == 0)
      continue;
    result.violationsZero = false;
    result.temporaryAdmitted &= llvm::is_contained(admitted, kind);
  }
  return result;
}

} // namespace

llvm::Error
SpatialPathFinderRouterScratch::prepare(const FrozenSpatialPnrProblem &problem,
                                        SpatialPnrWorkLedgerView workLedger,
                                        ExecutionControlView executionControl) {
  if (llvm::Error error = netRouter_.prepare(problem, workLedger))
    return error;
  executionControl_ = executionControl;
  const std::size_t logicalNetCount = problem.transfers().logicalNets().size();
  const std::size_t routeClaimCount = problem.routing().routeClaims().size();
  const std::size_t capacityCount =
      problem.resources().capacityDimensions().size();

  netOrder_.clear();
  netOrder_.reserve(logicalNetCount);
  activeClaimBits_.assign((routeClaimCount + 63) / 64, 0);
  claimEpochs_.assign(routeClaimCount, 0);
  capacityEpochs_.assign(capacityCount, 0);
  capacityNetQCosts_.assign(capacityCount, 0);
  touchedCapacities_.clear();
  touchedCapacities_.reserve(capacityCount);
  conflictCapacities_.clear();
  conflictCapacities_.reserve(capacityCount);
  regionalCapacityMarks_.assign(capacityCount, 0);
  regionalTagDomainMarks_.assign(
      problem.routing().tagContinuity().matchDomains().size(), 0);
  routingRegionNetMarks_.assign(logicalNetCount, 0);
  routingRegionNets_.clear();
  routingRegionNets_.reserve(logicalNetCount);
  constraintSweepNets_.clear();
  constraintSweepNets_.reserve(logicalNetCount);
  capturedSinkPathOffsets_.clear();
  capturedSinkPathOffsets_.reserve(
      problem.transfers().logicalNetSinks().size() + 1);
  capturedForwardArcs_.clear();
  reversePath_.clear();
  reversePath_.reserve(problem.routing().routingEndpoints().size());
  cutBlockedTraversals_.assign(problem.routing().traversals().size(), 0);
  cutReachableEndpoints_.assign(problem.routing().routingEndpoints().size(), 0);
  cutSeenTraversals_.assign(problem.routing().traversals().size(), 0);
  cutSeenEndpoints_.assign(problem.routing().routingEndpoints().size(), 0);
  cutWorklist_.clear();
  cutWorklist_.reserve(problem.routing().routingEndpoints().size());
  cutContributingNets_.clear();
  cutContributingNets_.reserve(logicalNetCount);
  cutForcedNetCuts_.clear();
  cutForcedNetCuts_.reserve(logicalNetCount);
  cutCertificateForcedNetCuts_.clear();
  cutCertificateForcedNetCuts_.reserve(logicalNetCount);
  cutPayloadWidths_.clear();
  cutPayloadWidths_.reserve(logicalNetCount);
  cutMinimumClaims_.clear();
  cutMinimumClaims_.reserve(logicalNetCount);
  cutTouchedClaims_.clear();
  cutTouchedClaims_.reserve(routeClaimCount);
  cutNetClaimRefcounts_.assign(routeClaimCount, 0);
  cutClaimSelectionCounts_.assign(routeClaimCount, 0);
  cutClaimTraversalRefcounts_.assign(routeClaimCount, 0);
  rankTrendTransitions_.clear();
  projectionEpoch_ = 0;
  negotiationIterationCount_ = 0;
  workLedger_ = workLedger;
  preparedProblem_ = &problem;
  return llvm::Error::success();
}

llvm::Error SpatialPathFinderRouterScratch::captureCurrentRoutes(
    const SpatialCandidateState &candidate,
    llvm::ArrayRef<PnrIndex> logicalNets) {
  capturedSinkPathOffsets_.clear();
  capturedForwardArcs_.clear();
  capturedSinkPathOffsets_.push_back(0);
  const FrozenSpatialRoutingGraph &routing = candidate.problem().routing();
  const auto arcs = routing.routingArcs();
  const auto captureLogicalNet = [&](PnrIndex logicalNet) -> llvm::Error {
    if (logicalNet >= candidate.problem().transfers().logicalNets().size())
      return pathFinderError("temporary routing region contains a foreign net");
    if (candidate.usesRegisterFifo(logicalNet))
      return llvm::Error::success();
    const RouteTreeState &tree = candidate.routeTree(logicalNet);
    if (!tree.isRouted())
      return pathFinderError("temporary iterate contains an unrouted net");
    const PnrIndex sinkCount =
        candidate.problem().transfers().logicalNets()[logicalNet].sinkCount;
    for (PnrIndex sink = 0; sink < sinkCount; ++sink) {
      const auto endpoint = tree.sinkEndpoint(sink);
      if (!endpoint)
        return pathFinderError(
            "temporary iterate has an unattached sink obligation");
      auto slot = tree.findNode(*endpoint);
      if (!slot)
        return pathFinderError(
            "temporary iterate sink is absent from its RouteTree");
      reversePath_.clear();
      for (std::size_t depth = 0;; ++depth) {
        if (depth > tree.nodeStorage().size())
          return pathFinderError(
              "temporary iterate RouteTree contains a cycle");
        const RouteTreeNode &node = tree.node(*slot);
        if (node.parentArc == getInvalidPnrIndex())
          break;
        if (node.parentArc >= arcs.size())
          return pathFinderError(
              "temporary iterate RouteTree arc is out of range");
        reversePath_.push_back(node.parentArc);
        slot = tree.parentNodeSlot(*slot);
        if (!slot)
          return pathFinderError(
              "temporary iterate RouteTree parent is absent");
      }
      capturedForwardArcs_.insert(capturedForwardArcs_.end(),
                                  reversePath_.rbegin(), reversePath_.rend());
      capturedSinkPathOffsets_.push_back(capturedForwardArcs_.size());
    }
    return llvm::Error::success();
  };
  if (logicalNets.empty()) {
    for (PnrIndex logicalNet = 0;
         logicalNet < candidate.problem().transfers().logicalNets().size();
         ++logicalNet)
      if (llvm::Error error = captureLogicalNet(logicalNet))
        return error;
  } else {
    for (PnrIndex logicalNet : logicalNets)
      if (llvm::Error error = captureLogicalNet(logicalNet))
        return error;
  }
  return llvm::Error::success();
}

llvm::Error SpatialPathFinderRouterScratch::restoreCapturedRoutes(
    SpatialMoveTransaction &move, SpatialCandidateState &candidate,
    SpatialRouteCostState &costs, llvm::ArrayRef<PnrIndex> logicalNets,
    const dse::ObjectiveVector &expectedObjective,
    const SpatialCandidateRouteProjection &expectedProjection) {
  const FrozenSpatialRoutingGraph &routing = candidate.problem().routing();
  const auto arcs = routing.routingArcs();
  std::size_t sinkPath = 0;
  const auto restoreLogicalNet = [&](PnrIndex logicalNet) -> llvm::Error {
    if (logicalNet >= candidate.problem().transfers().logicalNets().size())
      return pathFinderError("captured routing region contains a foreign net");
    if (candidate.usesRegisterFifo(logicalNet))
      return llvm::Error::success();
    auto projection = projectLogicalNet(candidate, costs, logicalNet);
    if (!projection)
      return projection.takeError();
    if (llvm::Error error =
            costs.selectLogicalNet(logicalNet, activeClaimBits_))
      return error;
    if (llvm::Error error = move.ripUpWholeRoute(logicalNet))
      return error;
    const PnrIndex source = candidate.logicalNetSourceEndpoint(logicalNet);
    if (llvm::Error error = move.bindRouteSource(logicalNet, source))
      return error;
    const PnrIndex sinkCount =
        candidate.problem().transfers().logicalNets()[logicalNet].sinkCount;
    for (PnrIndex sink = 0; sink < sinkCount; ++sink, ++sinkPath) {
      if (llvm::Error error = move.bindRouteSink(
              logicalNet, sink,
              candidate.logicalNetSinkEndpoint(logicalNet, sink)))
        return error;
      if (sinkPath + 1 >= capturedSinkPathOffsets_.size())
        return pathFinderError("captured route path domain is truncated");
      const std::size_t begin = capturedSinkPathOffsets_[sinkPath];
      const std::size_t end = capturedSinkPathOffsets_[sinkPath + 1];
      if (begin > end || end > capturedForwardArcs_.size())
        return pathFinderError("captured route path range is invalid");
      const llvm::ArrayRef<PnrIndex> path(capturedForwardArcs_.data() + begin,
                                          end - begin);
      PnrIndex attachment = source;
      std::size_t pathBegin = 0;
      const RouteTreeState &tree = candidate.routeTree(logicalNet);
      for (auto [index, arc] : llvm::enumerate(path)) {
        if (arc >= arcs.size())
          return pathFinderError("captured route arc is out of range");
        if (tree.findNode(arcs[arc].target)) {
          attachment = arcs[arc].target;
          pathBegin = index + 1;
        }
      }
      if (llvm::Error error = move.attachRoutePath(
              logicalNet, attachment, path.drop_front(pathBegin), sink))
        return error;
    }
    auto restored = projectLogicalNet(candidate, costs, logicalNet);
    if (!restored)
      return restored.takeError();
    if (llvm::Error error =
            costs.updateSelectedLogicalNetClaims(activeClaimBits_))
      return error;
    if (llvm::Error error = costs.acceptSelectedLogicalNet())
      return error;
    return llvm::Error::success();
  };
  if (logicalNets.empty()) {
    for (PnrIndex logicalNet = 0;
         logicalNet < candidate.problem().transfers().logicalNets().size();
         ++logicalNet)
      if (llvm::Error error = restoreLogicalNet(logicalNet))
        return error;
  } else {
    for (PnrIndex logicalNet : logicalNets)
      if (llvm::Error error = restoreLogicalNet(logicalNet))
        return error;
  }
  if (sinkPath + 1 != capturedSinkPathOffsets_.size())
    return pathFinderError("captured route path domain has trailing entries");
  SpatialTagAssignmentSummary restoredTagSummary;
  auto restoredProjection = move.projectCurrentRoutes(restoredTagSummary);
  if (!restoredProjection)
    return restoredProjection.takeError();
  if (llvm::Error error =
          costs.synchronizeTagProjection(restoredTagSummary, logicalNets))
    return error;
  auto restored =
      candidate.problem().objectiveProgram().evaluateSpatialProjection(
          candidate, *restoredProjection);
  if (!restored)
    return restored.takeError();
  if (restored->codes() != expectedObjective.codes()) {
    loom::mapping_debug::emit(
        loom::mapping_debug::Level::Decision,
        loom::mapping_debug::Stage::SpatialPnr,
        loom::mapping_debug::Event::MappingFailure,
        [&](llvm::json::Object &fields) {
          llvm::json::Array expectedCodes;
          for (std::uint64_t code : expectedObjective.codes())
            expectedCodes.push_back(code);
          llvm::json::Array restoredCodes;
          for (std::uint64_t code : restored->codes())
            restoredCodes.push_back(code);
          fields["operation"] = "captured_route_objective_restore";
          fields["expected_codes"] = std::move(expectedCodes);
          fields["restored_codes"] = std::move(restoredCodes);
          fields["expected_selected_traversal_claim"] =
              expectedProjection.totalSelectedTraversalClaim;
          fields["restored_selected_traversal_claim"] =
              restoredProjection->totalSelectedTraversalClaim;
          fields["logical_net_count"] =
              logicalNets.empty()
                  ? candidate.problem().transfers().logicalNets().size()
                  : logicalNets.size();
        });
    return pathFinderError(
        "captured temporary routes did not restore their objective vector");
  }
  if (restoredProjection->unroutedObligationCount !=
          expectedProjection.unroutedObligationCount ||
      restoredProjection->routeCapacityOveruse !=
          expectedProjection.routeCapacityOveruse ||
      restoredProjection->tagResidentCapacityOveruse !=
          expectedProjection.tagResidentCapacityOveruse ||
      restoredProjection->tagUnassignedCount !=
          expectedProjection.tagUnassignedCount ||
      restoredProjection->tagConflictCount !=
          expectedProjection.tagConflictCount ||
      restoredProjection->hardProgressViolation !=
          expectedProjection.hardProgressViolation ||
      restoredProjection->progressProofDebtWitnessCount !=
          expectedProjection.progressProofDebtWitnessCount ||
      restoredProjection->progressCapacityShortfall !=
          expectedProjection.progressCapacityShortfall ||
      restoredProjection->progressRouteAnchorCount !=
          expectedProjection.progressRouteAnchorCount ||
      restoredProjection->runtimeCounterexampleViolation !=
          expectedProjection.runtimeCounterexampleViolation ||
      restoredProjection->totalSelectedTraversalClaim !=
          expectedProjection.totalSelectedTraversalClaim ||
      restoredProjection->routeReleaseLatencyCycles !=
          expectedProjection.routeReleaseLatencyCycles ||
      restoredProjection->routeMinimumInitiationIntervalCycles !=
          expectedProjection.routeMinimumInitiationIntervalCycles ||
      restoredProjection->transportBitCycleDemand !=
          expectedProjection.transportBitCycleDemand ||
      restoredProjection->routeTerminalsCompatible !=
          expectedProjection.routeTerminalsCompatible ||
      restoredProjection->selectedHandshakeAcyclic !=
          expectedProjection.selectedHandshakeAcyclic)
    return pathFinderError(
        "captured temporary routes did not restore their Mapping projection");
  return llvm::Error::success();
}

llvm::Expected<SpatialPathFinderClosureResult>
SpatialPathFinderRouterScratch::routeToClosure(
    SpatialCandidateState &candidate, SpatialCandidateScratch &candidateScratch,
    SpatialRouteCostState &costs, SpatialPathFinderRoutingLimits limits,
    llvm::ArrayRef<RouteCost> evaluationPriorities,
    SpatialRoutingClosureRequirement closureRequirement) {
  if (llvm::Error error = costs.resetFromCandidate())
    return std::move(error);
  auto moveOrError = candidate.beginMove(candidateScratch);
  if (!moveOrError)
    return moveOrError.takeError();
  SpatialMoveTransaction move = std::move(*moveOrError);

  auto result = routeToClosureInMove(move, candidate, costs, limits, {},
                                     evaluationPriorities, closureRequirement);
  if (!result)
    return rollbackIteration(move, costs, result.takeError());
  auto closed = move.close();
  if (!closed)
    return rollbackIteration(move, costs, closed.takeError());
  if (!*closed)
    return rollbackIteration(
        move, costs,
        llvm::make_error<SpatialPathFinderClosureFailure>(
            SpatialPathFinderClosureFailure::Kind::
                SelectedCombinationalHandshakeCycle,
            "Spatial PathFinder selected a combinational handshake cycle"));
  if (llvm::Error error = move.commit())
    return error;
  return result;
}

llvm::Expected<SpatialPathFinderClosureResult>
SpatialPathFinderRouterScratch::routeToClosureInMove(
    SpatialMoveTransaction &move, SpatialCandidateState &candidate,
    SpatialRouteCostState &costs, SpatialPathFinderRoutingLimits limits,
    llvm::ArrayRef<PnrIndex> logicalNets,
    llvm::ArrayRef<RouteCost> evaluationPriorities,
    SpatialRoutingClosureRequirement closureRequirement,
    std::uint64_t exactRegionalLogicalNetLimit,
    std::optional<SpatialTraversalRouteCut> routeCut) {
  if (!preparedProblem_ || preparedProblem_ != &candidate.problem())
    return pathFinderError("scratch is not prepared for the candidate freeze");
  if (!costs.isBoundTo(candidate))
    return pathFinderError("route costs are bound to another candidate");
  if (limits.endpointExpansionLimit == 0 || limits.iterationLimit == 0 ||
      limits.noProgressIterationLimit == 0 || limits.noProgressTrendWindow == 0)
    return pathFinderError("routing work limits must be positive");
  if (limits.noProgressTrendWindow > limits.noProgressIterationLimit ||
      limits.noProgressIterationLimit > limits.iterationLimit)
    return pathFinderError("routing no-progress limits are not canonical");
  if (limits.noProgressTrendWindow > std::numeric_limits<std::size_t>::max())
    return pathFinderError("routing trend window exceeds host size_t");
  const std::size_t trendWindow =
      static_cast<std::size_t>(limits.noProgressTrendWindow);
  if (rankTrendTransitions_.size() < trendWindow)
    rankTrendTransitions_.resize(trendWindow);
  if (!evaluationPriorities.empty() &&
      evaluationPriorities.size() !=
          candidate.problem().transfers().logicalNets().size())
    return pathFinderError("evaluation-priority vector has the wrong width");
  if (costs.selectedLogicalNet())
    return pathFinderError("route costs already have a selected logical net");
  if (routeCut) {
    const auto nets = candidate.problem().transfers().logicalNets();
    if (routeCut->logicalNet >= nets.size())
      return pathFinderError("route cut logical net is out of range");
    if (routeCut->traversal >=
        candidate.problem().routing().traversals().size())
      return pathFinderError("route cut traversal is out of range");
    if (routeCut->sinkObligation &&
        *routeCut->sinkObligation >= nets[routeCut->logicalNet].sinkCount)
      return pathFinderError("route cut sink is out of range");
  }

  const bool regionalRouting = !logicalNets.empty();
  routingRegionNets_.clear();
  std::fill(routingRegionNetMarks_.begin(), routingRegionNetMarks_.end(), 0);
  if (regionalRouting) {
    if (closureRequirement == SpatialRoutingClosureRequirement::ExactRegional &&
        exactRegionalLogicalNetLimit == 0)
      return pathFinderError(
          "exact regional routing requires a logical-net limit");
    if (closureRequirement != SpatialRoutingClosureRequirement::ExactRegional &&
        exactRegionalLogicalNetLimit != 0)
      return pathFinderError(
          "only exact regional routing may carry a logical-net limit");
    for (PnrIndex logicalNet : logicalNets) {
      if (logicalNet >= routingRegionNetMarks_.size())
        return pathFinderError("routing region contains a foreign logical net");
      if (routingRegionNetMarks_[logicalNet])
        return pathFinderError("routing region repeats a logical net");
      routingRegionNetMarks_[logicalNet] = 1;
      routingRegionNets_.push_back(logicalNet);
    }
    llvm::sort(routingRegionNets_);
    if (routingRegionNets_.size() > exactRegionalLogicalNetLimit &&
        closureRequirement == SpatialRoutingClosureRequirement::ExactRegional)
      return llvm::make_error<SpatialPathFinderClosureFailure>(
          SpatialPathFinderClosureFailure::Kind::RegionalLimit,
          "Spatial PathFinder routing region exceeds its logical-net limit",
          SpatialFixedTerminalCutCertificate{}, 0, 0, routingRegionNets_.size(),
          exactRegionalLogicalNetLimit);
  } else if (exactRegionalLogicalNetLimit != 0) {
    return pathFinderError(
        "full-design routing cannot carry a regional logical-net limit");
  }
  if (regionalRouting) {
    auto expanded = expandRoutingRelationClosure(
        closureRequirement == SpatialRoutingClosureRequirement::ExactRegional
            ? exactRegionalLogicalNetLimit
            : 0);
    if (!expanded)
      return expanded.takeError();
  }
  if (routeCut && regionalRouting &&
      !routingRegionNetMarks_[routeCut->logicalNet])
    return pathFinderError("route cut is outside the routing region");
  const auto activeLogicalNets = [&]() -> llvm::ArrayRef<PnrIndex> {
    return regionalRouting ? llvm::ArrayRef<PnrIndex>(routingRegionNets_)
                           : llvm::ArrayRef<PnrIndex>{};
  };

  SpatialTagAssignmentSummary initialTagSummary;
  auto initialProjection = move.projectCurrentRoutes(initialTagSummary);
  if (!initialProjection)
    return initialProjection.takeError();
  if (llvm::Error error = costs.synchronizeTagProjection(initialTagSummary))
    return std::move(error);
  auto initialRegion =
      projectRoutingRegion(candidate, costs, activeLogicalNets());
  if (!initialRegion)
    return initialRegion.takeError();
  auto initialObjective =
      candidate.problem().objectiveProgram().evaluateSpatialProjection(
          candidate, *initialProjection);
  if (!initialObjective)
    return initialObjective.takeError();
  bool initialRouteCutHolds = false;
  if (routeCut) {
    auto holds = netRouter_.internalRouteCutHolds(candidate, *routeCut);
    if (!holds)
      return holds.takeError();
    initialRouteCutHolds = *holds;
  }
  if (initialProjection->routeTerminalsCompatible &&
      initialProjection->selectedHandshakeAcyclic &&
      initialRegion->unroutedObligationCount == 0 &&
      initialRegion->routeCapacityOveruse == 0 &&
      initialRegion->tagResidentCapacityOveruse == 0 &&
      initialRegion->tagUnassignedCount == 0 &&
      initialRegion->tagConflictCount == 0 && !initialRouteCutHolds)
    return SpatialPathFinderClosureResult{0, true};

  std::optional<dse::ObjectiveVector> bestRankObjective;
  std::optional<dse::ObjectiveVector> previousRankObjective;
  std::optional<dse::ObjectiveVector> bestTemporaryObjective;
  std::optional<SpatialCandidateRouteProjection> bestTemporaryProjection;
  const auto compareSelectedRank =
      [&](const dse::ObjectiveVector &leftObjective,
          const dse::ObjectiveVector &rightObjective) -> llvm::Expected<int> {
    return candidate.problem().objectiveProgram().compareSelectedRank(
        leftObjective, {}, rightObjective, {});
  };
  std::uint64_t consecutiveNoProgressIterations = 0;
  std::size_t trendHead = 0;
  std::size_t trendCount = 0;
  std::uint64_t trendImprovedCount = 0;
  std::uint64_t trendIneligibleCount = 0;
  std::uint64_t trendRegressedCount = 0;
  const std::uint64_t initialEndpointExpansions =
      netRouter_.endpointExpansionCount();
  loom::mapping_debug::MappingRunStatistics debugStatistics;
  std::uint64_t preservedNetRoutes = 0;
  std::uint64_t selectedSinkRoutes = 0;
  std::uint64_t wholeNetRoutes = 0;
  const auto emitStatistics = [&](loom::mapping_debug::ClosureStatus status) {
    debugStatistics.aStarExpansions =
        netRouter_.endpointExpansionCount() - initialEndpointExpansions;
    debugStatistics.emit(loom::mapping_debug::Stage::SpatialPnr, status,
                         [&](llvm::json::Object &fields) {
                           fields["preserved_net_routes"] = preservedNetRoutes;
                           fields["selected_sink_routes"] = selectedSinkRoutes;
                           fields["whole_net_routes"] = wholeNetRoutes;
                         });
  };

  const auto interrupted = [&](llvm::StringRef boundary) {
    emitStatistics(loom::mapping_debug::ClosureStatus::CancelledOrTimeout);
    return llvm::make_error<SpatialPathFinderClosureFailure>(
        SpatialPathFinderClosureFailure::Kind::Interrupted,
        ("Spatial PathFinder observed an execution stop " + boundary).str());
  };
  for (std::uint64_t iteration = 0; iteration < limits.iterationLimit;
       ++iteration) {
    if (executionControl_.stopRequested())
      return interrupted("before a negotiation iteration");
    if (negotiationIterationCount_ ==
        std::numeric_limits<std::uint64_t>::max()) {
      ++debugStatistics.arithmeticFailures;
      loom::mapping_debug::emit(loom::mapping_debug::Level::Decision,
                                loom::mapping_debug::Stage::SpatialPnr,
                                loom::mapping_debug::Event::ArithmeticFailure,
                                [&](llvm::json::Object &fields) {
                                  fields["iteration"] = iteration;
                                  fields["operation"] =
                                      "negotiation_iteration_counter";
                                });
      emitStatistics(loom::mapping_debug::ClosureStatus::ArithmeticFailure);
      return pathFinderError("negotiation iteration count overflows u64");
    }
    if (llvm::Error error =
            workLedger_.plan(SpatialPnrWorkKind::NegotiationIteration))
      return std::move(error);
    const std::uint64_t sessionIteration = negotiationIterationCount_;
    bool iterationConsumed = false;
    const auto consumeIteration = [&]() -> llvm::Error {
      if (iterationConsumed)
        return pathFinderError("negotiation iteration was consumed twice");
      if (llvm::Error error =
              workLedger_.consume(SpatialPnrWorkKind::NegotiationIteration))
        return error;
      ++negotiationIterationCount_;
      ++debugStatistics.negotiatedIterations;
      iterationConsumed = true;
      return llvm::Error::success();
    };
    const auto completeIterationFailure = [&](llvm::Error failure) {
      bool completed = false;
      llvm::Error classified =
          classifyIterationFailure(std::move(failure), completed);
      if (!completed)
        return classified;
      if (llvm::Error error = consumeIteration())
        return llvm::joinErrors(std::move(classified), std::move(error));
      return classified;
    };
    if (llvm::Error error = buildCanonicalNetOrder(
            candidate, costs, activeLogicalNets(), evaluationPriorities))
      return completeIterationFailure(std::move(error));

    constraintSweepNets_.clear();
    for (const NetOrderEntry &entry : netOrder_)
      constraintSweepNets_.push_back(entry.logicalNet);
    if (llvm::Error error =
            netRouter_.beginConstraintSweep(constraintSweepNets_))
      return completeIterationFailure(std::move(error));

    for (const NetOrderEntry &entry : netOrder_) {
      if (executionControl_.stopRequested())
        return completeIterationFailure(
            interrupted("between the net routes of an iteration"));
      auto routePlan =
          netRouter_.planNegotiatedRoute(candidate, costs, entry.logicalNet);
      if (!routePlan)
        return completeIterationFailure(routePlan.takeError());
      const std::optional<SpatialTraversalRouteCut> entryCut =
          routeCut && routeCut->logicalNet == entry.logicalNet ? routeCut
                                                               : std::nullopt;
      if (entryCut) {
        auto holds = netRouter_.internalRouteCutHolds(candidate, *entryCut);
        if (!holds)
          return completeIterationFailure(holds.takeError());
        if (*holds)
          *routePlan = detail::SpatialNegotiatedRoutePlan{
              detail::SpatialNegotiatedRouteScope::WholeNet, {}};
      }
      const FrozenSpatialLogicalNet &logicalNet =
          candidate.problem().transfers().logicalNets()[entry.logicalNet];
      if (routePlan->scope == detail::SpatialNegotiatedRouteScope::Preserve) {
        ++preservedNetRoutes;
        loom::mapping_debug::emit(
            loom::mapping_debug::Level::Detail,
            loom::mapping_debug::Stage::SpatialPnr,
            loom::mapping_debug::Event::NetRoute,
            [&](llvm::json::Object &fields) {
              fields["iteration"] = iteration;
              fields["session_iteration"] = sessionIteration;
              fields["logical_net"] = entry.logicalNet;
              fields["route_cost"] = 0;
              fields["route_scope"] = "preserved";
              fields["source_endpoint"] =
                  candidate.logicalNetSourceEndpoint(entry.logicalNet);
              fields["sink_count"] = logicalNet.sinkCount;
              fields["rerouted_sink_count"] = 0;
              fields["preserved_sink_count"] = logicalNet.sinkCount;
            });
        if (llvm::Error error =
                netRouter_.finishConstraintNet(entry.logicalNet))
          return completeIterationFailure(std::move(error));
        continue;
      }
      auto projection = projectLogicalNet(candidate, costs, entry.logicalNet);
      if (!projection)
        return completeIterationFailure(projection.takeError());
      if (llvm::Error error =
              costs.selectLogicalNet(entry.logicalNet, activeClaimBits_))
        return completeIterationFailure(std::move(error));
      const bool selectedSinks =
          routePlan->scope ==
          detail::SpatialNegotiatedRouteScope::SelectedSinks;
      if (selectedSinks)
        ++selectedSinkRoutes;
      else
        ++wholeNetRoutes;
      auto route = selectedSinks ? netRouter_.routeSinkSet(
                                       move, candidate, costs, entry.logicalNet,
                                       routePlan->sinkObligations,
                                       limits.endpointExpansionLimit, entryCut)
                                 : netRouter_.routeWholeNet(
                                       move, candidate, costs, entry.logicalNet,
                                       limits.endpointExpansionLimit, entryCut);
      if (!route) {
        llvm::Error routeFailure = route.takeError();
        bool emittedTypedFailure = false;
        if (loom::mapping_debug::enabled(
                loom::mapping_debug::Level::Decision)) {
          routeFailure = llvm::handleErrors(
              std::move(routeFailure),
              [&](const EndpointRouteSearchFailure &failure) -> llvm::Error {
                emittedTypedFailure = true;
                std::string diagnostic = errorMessage(failure);
                loom::mapping_debug::emit(
                    loom::mapping_debug::Level::Decision,
                    loom::mapping_debug::Stage::SpatialPnr,
                    loom::mapping_debug::Event::MappingFailure,
                    [&](llvm::json::Object &fields) {
                      fields["iteration"] = iteration;
                      fields["session_iteration"] = sessionIteration;
                      fields["logical_net"] = entry.logicalNet;
                      fields["source_endpoint"] =
                          candidate.logicalNetSourceEndpoint(entry.logicalNet);
                      fields["sink_count"] = logicalNet.sinkCount;
                      fields["logical_net_detail"] =
                          encodeLogicalNetDetail(candidate, entry.logicalNet);
                      fields["operation"] =
                          selectedSinks ? "route_sink_set" : "route_whole_net";
                      fields["failure_kind"] =
                          stringifyEndpointRouteSearchFailureKind(
                              failure.kind());
                      fields["diagnostic"] = diagnostic;
                    });
                return llvm::make_error<EndpointRouteSearchFailure>(
                    failure.kind(), std::move(diagnostic));
              });
        }
        if (!emittedTypedFailure)
          loom::mapping_debug::emit(
              loom::mapping_debug::Level::Decision,
              loom::mapping_debug::Stage::SpatialPnr,
              loom::mapping_debug::Event::MappingFailure,
              [&](llvm::json::Object &fields) {
                fields["iteration"] = iteration;
                fields["session_iteration"] = sessionIteration;
                fields["logical_net"] = entry.logicalNet;
                fields["source_endpoint"] =
                    candidate.logicalNetSourceEndpoint(entry.logicalNet);
                fields["sink_count"] = logicalNet.sinkCount;
                fields["logical_net_detail"] =
                    encodeLogicalNetDetail(candidate, entry.logicalNet);
                fields["operation"] =
                    selectedSinks ? "route_sink_set" : "route_whole_net";
              });
        emitStatistics(loom::mapping_debug::ClosureStatus::RouteFailure);
        return completeIterationFailure(std::move(routeFailure));
      }
      loom::mapping_debug::emit(
          loom::mapping_debug::Level::Detail,
          loom::mapping_debug::Stage::SpatialPnr,
          loom::mapping_debug::Event::NetRoute,
          [&](llvm::json::Object &fields) {
            fields["iteration"] = iteration;
            fields["session_iteration"] = sessionIteration;
            fields["logical_net"] = entry.logicalNet;
            fields["route_cost"] = *route;
            fields["route_scope"] =
                selectedSinks ? "selected_sinks" : "whole_net";
            fields["source_endpoint"] =
                candidate.logicalNetSourceEndpoint(entry.logicalNet);
            fields["sink_count"] = logicalNet.sinkCount;
            fields["rerouted_sink_count"] =
                selectedSinks ? routePlan->sinkObligations.size()
                              : logicalNet.sinkCount;
            fields["preserved_sink_count"] =
                selectedSinks
                    ? logicalNet.sinkCount - routePlan->sinkObligations.size()
                    : 0;
          });
      if (llvm::Error error = costs.acceptSelectedLogicalNet())
        return completeIterationFailure(std::move(error));
      if (llvm::Error error = netRouter_.finishConstraintNet(entry.logicalNet))
        return completeIterationFailure(std::move(error));
    }

    const std::uint64_t completedIterations = iteration + 1;
    SpatialTagAssignmentSummary tagSummary;
    auto projection = move.projectCurrentRoutes(tagSummary);
    if (!projection)
      return completeIterationFailure(projection.takeError());
    if (llvm::Error error =
            costs.synchronizeTagProjection(tagSummary, activeLogicalNets()))
      return completeIterationFailure(std::move(error));
    auto region = projectRoutingRegion(candidate, costs, activeLogicalNets());
    if (!region)
      return completeIterationFailure(region.takeError());
    auto capacityOveruse = spatialMappingViolationValue(
        candidate, *projection, ResolvedPnrViolationKind::CapacityOveruse);
    if (!capacityOveruse)
      return completeIterationFailure(capacityOveruse.takeError());
    const bool hasRouteCapacityOveruse = region->routeCapacityOveruse != 0;
    const bool hasTagCapacityOveruse = region->tagResidentCapacityOveruse != 0;
    const bool hasTagEncodingViolation =
        region->tagUnassignedCount != 0 || region->tagConflictCount != 0;
    const bool hasCapacityOveruse = hasRouteCapacityOveruse ||
                                    hasTagCapacityOveruse ||
                                    hasTagEncodingViolation;
    if (logicalNets.empty() &&
        hasRouteCapacityOveruse != (projection->routeCapacityOveruse != 0))
      return pathFinderError(
          "working route capacity disagrees with the provisional RouteTrees");
    if (logicalNets.empty() &&
        region->tagUnassignedCount != projection->tagUnassignedCount)
      return pathFinderError(
          "working unassigned tags disagree with the provisional RouteTrees");
    CapacityConflictAnalysis conflictAnalysis;
    if (hasRouteCapacityOveruse) {
      auto analyzed = analyzeCapacityConflicts(candidate, costs, iteration,
                                               sessionIteration);
      if (!analyzed)
        return completeIterationFailure(analyzed.takeError());
      conflictAnalysis = *analyzed;
    }
    if (hasRouteCapacityOveruse || hasTagCapacityOveruse ||
        hasTagEncodingViolation) {
      if (regionalRouting &&
          closureRequirement ==
              SpatialRoutingClosureRequirement::ExactRegional) {
        const std::uint64_t previousNetCount = routingRegionNets_.size();
        auto expanded = expandExactRegionalConflictClosure(
            candidate, costs, exactRegionalLogicalNetLimit);
        if (!expanded)
          return completeIterationFailure(expanded.takeError());
        if (*expanded) {
          loom::mapping_debug::emit(
              loom::mapping_debug::Level::Decision,
              loom::mapping_debug::Stage::SpatialPnr,
              loom::mapping_debug::Event::ActionProposal,
              [&](llvm::json::Object &fields) {
                fields["search_scope"] = "route_conflict_closure";
                fields["iteration"] = iteration;
                fields["session_iteration"] = sessionIteration;
                fields["previous_logical_net_count"] = previousNetCount;
                fields["logical_net_count"] = routingRegionNets_.size();
                fields["logical_net_limit"] = exactRegionalLogicalNetLimit;
              });
          bestRankObjective.reset();
          previousRankObjective.reset();
          bestTemporaryObjective.reset();
          bestTemporaryProjection.reset();
          consecutiveNoProgressIterations = 0;
          trendHead = 0;
          trendCount = 0;
          trendImprovedCount = 0;
          trendIneligibleCount = 0;
          trendRegressedCount = 0;
          if (llvm::Error error = costs.advancePathFinderIteration())
            return completeIterationFailure(std::move(error));
          if (llvm::Error error = consumeIteration())
            return std::move(error);
          continue;
        }
      }
    }
    const std::uint64_t tagPressureEvents = reportSpatialTagDomainPressure(
        candidate, costs, tagSummary, iteration, sessionIteration);
    const std::uint64_t capacityConflicts =
        conflictAnalysis.conflictCount + tagPressureEvents;
    debugStatistics.capacityConflicts += capacityConflicts;
    bool selectedRankImproved = false;
    bool selectedRankImprovedFromInitial = false;
    bool temporaryAdmitted = false;
    bool mappingViolationsZero = false;
    const bool selectedRankEligible = projection->routeTerminalsCompatible &&
                                      projection->selectedHandshakeAcyclic;
    std::optional<RankTrendTransition> rankTrendTransition;
    if (selectedRankEligible) {
      auto disposition =
          classifyProjection(candidate, *projection, closureRequirement);
      if (!disposition)
        return completeIterationFailure(disposition.takeError());
      temporaryAdmitted = disposition->temporaryAdmitted;
      mappingViolationsZero = disposition->violationsZero;
      auto objective =
          candidate.problem().objectiveProgram().evaluateSpatialProjection(
              candidate, *projection);
      if (!objective)
        return completeIterationFailure(objective.takeError());
      selectedRankImproved = !bestRankObjective;
      if (bestRankObjective) {
        auto comparison = compareSelectedRank(*objective, *bestRankObjective);
        if (!comparison)
          return completeIterationFailure(comparison.takeError());
        selectedRankImproved = *comparison < 0;
      }
      if (selectedRankImproved) {
        bestRankObjective = *objective;
        consecutiveNoProgressIterations = 0;
      } else {
        ++consecutiveNoProgressIterations;
      }

      if (previousRankObjective) {
        auto comparison =
            compareSelectedRank(*objective, *previousRankObjective);
        if (!comparison)
          return completeIterationFailure(comparison.takeError());
        rankTrendTransition = *comparison < 0   ? RankTrendTransition::Improved
                              : *comparison > 0 ? RankTrendTransition::Regressed
                                                : RankTrendTransition::Equal;
      }
      previousRankObjective = *objective;

      auto initialComparison =
          compareSelectedRank(*objective, *initialObjective);
      if (!initialComparison)
        return completeIterationFailure(initialComparison.takeError());
      selectedRankImprovedFromInitial = *initialComparison < 0;

      if (temporaryAdmitted) {
        bool replace = !bestTemporaryObjective;
        if (bestTemporaryObjective) {
          auto comparison =
              compareSelectedRank(*objective, *bestTemporaryObjective);
          if (!comparison)
            return completeIterationFailure(comparison.takeError());
          replace = *comparison < 0;
        }
        if (replace) {
          if (llvm::Error error =
                  captureCurrentRoutes(candidate, activeLogicalNets()))
            return completeIterationFailure(std::move(error));
          bestTemporaryProjection = *projection;
          bestTemporaryObjective = std::move(*objective);
        }
      }
    } else {
      ++consecutiveNoProgressIterations;
      rankTrendTransition = RankTrendTransition::Ineligible;
    }
    if (rankTrendTransition) {
      if (trendCount == trendWindow) {
        const auto evicted =
            static_cast<RankTrendTransition>(rankTrendTransitions_[trendHead]);
        trendImprovedCount -= evicted == RankTrendTransition::Improved ? 1 : 0;
        trendIneligibleCount -=
            evicted == RankTrendTransition::Ineligible ? 1 : 0;
        trendRegressedCount -=
            evicted == RankTrendTransition::Regressed ? 1 : 0;
      } else {
        ++trendCount;
      }
      rankTrendTransitions_[trendHead] =
          static_cast<std::uint8_t>(*rankTrendTransition);
      trendHead = (trendHead + 1) % trendWindow;
      trendImprovedCount +=
          *rankTrendTransition == RankTrendTransition::Improved ? 1 : 0;
      trendIneligibleCount +=
          *rankTrendTransition == RankTrendTransition::Ineligible ? 1 : 0;
      trendRegressedCount +=
          *rankTrendTransition == RankTrendTransition::Regressed ? 1 : 0;
    }
    loom::mapping_debug::emit(
        loom::mapping_debug::Level::Decision,
        loom::mapping_debug::Stage::SpatialPnr,
        loom::mapping_debug::Event::NegotiationIteration,
        [&](llvm::json::Object &fields) {
          fields["iteration"] = iteration;
          fields["session_iteration"] = sessionIteration;
          fields["logical_net_count"] = netOrder_.size();
          fields["capacity_conflicts"] = capacityConflicts;
          fields["capacity_conflict_events"] =
              conflictAnalysis.diagnosticConflictCount;
          fields["capacity_conflict_events_omitted"] =
              conflictAnalysis.conflictCount -
              conflictAnalysis.diagnosticConflictCount;
          fields["tag_domain_pressure_events"] = tagPressureEvents;
          fields["route_capacity_closed"] = !hasRouteCapacityOveruse;
          fields["tag_capacity_closed"] = !hasTagCapacityOveruse;
          fields["tag_encoding_closed"] = !hasTagEncodingViolation;
          fields["capacity_closed"] = !hasCapacityOveruse;
          fields["mapping_closed"] = mappingViolationsZero;
          fields["route_terminals_compatible"] =
              projection->routeTerminalsCompatible;
          fields["selected_handshake_acyclic"] =
              projection->selectedHandshakeAcyclic;
          fields["unrouted_obligations"] = projection->unroutedObligationCount;
          fields["capacity_overuse"] = *capacityOveruse;
          fields["route_capacity_overuse"] = projection->routeCapacityOveruse;
          fields["regional_unrouted_obligations"] =
              region->unroutedObligationCount;
          fields["regional_route_capacity_overuse"] =
              region->routeCapacityOveruse;
          fields["regional_tag_resident_capacity_overuse"] =
              region->tagResidentCapacityOveruse;
          fields["regional_tag_unassigned"] = region->tagUnassignedCount;
          fields["regional_tag_conflicts"] = region->tagConflictCount;
          fields["tag_resident_capacity_overuse"] =
              projection->tagResidentCapacityOveruse;
          fields["tag_unassigned"] = projection->tagUnassignedCount;
          fields["tag_conflicts"] = projection->tagConflictCount;
          fields["hard_progress_violations"] =
              projection->hardProgressViolation;
          fields["progress_proof_debt_witnesses"] =
              projection->progressProofDebtWitnessCount;
          fields["progress_capacity_shortfall"] =
              projection->progressCapacityShortfall;
          fields["progress_route_anchors"] =
              projection->progressRouteAnchorCount;
          fields["runtime_counterexample_violations"] =
              projection->runtimeCounterexampleViolation;
          fields["selected_traversal_claim"] =
              projection->totalSelectedTraversalClaim;
          fields["temporary_admitted"] = temporaryAdmitted;
          fields["selected_rank_improved"] = selectedRankImproved;
          fields["consecutive_no_progress_iterations"] =
              consecutiveNoProgressIterations;
          fields["no_progress_iteration_limit"] =
              limits.noProgressIterationLimit;
          fields["no_progress_trend_window"] = limits.noProgressTrendWindow;
          fields["rank_trend_transition_count"] = trendCount;
          fields["rank_trend_improved_count"] = trendImprovedCount;
          fields["rank_trend_ineligible_count"] = trendIneligibleCount;
          fields["rank_trend_regressed_count"] = trendRegressedCount;
          fields["a_star_expansions"] =
              netRouter_.endpointExpansionCount() - initialEndpointExpansions;
        });
    bool routeCutClosed = true;
    if (routeCut) {
      auto holds = netRouter_.internalRouteCutHolds(candidate, *routeCut);
      if (!holds)
        return completeIterationFailure(holds.takeError());
      routeCutClosed = !*holds;
    }
    const bool routingClosed =
        projection->routeTerminalsCompatible &&
        projection->selectedHandshakeAcyclic &&
        region->unroutedObligationCount == 0 && !hasCapacityOveruse &&
        routeCutClosed &&
        (!regionalRouting || selectedRankImprovedFromInitial || routeCut ||
         (closureRequirement == SpatialRoutingClosureRequirement::ExactRegional &&
          mappingViolationsZero));
    if (routingClosed) {
      emitStatistics(loom::mapping_debug::ClosureStatus::Closed);
      if (llvm::Error error = consumeIteration())
        return std::move(error);
      return SpatialPathFinderClosureResult{completedIterations, true};
    }
    if (!hasCapacityOveruse) {
      if (bestTemporaryObjective) {
        if (llvm::Error error = restoreCapturedRoutes(
                move, candidate, costs, activeLogicalNets(),
                *bestTemporaryObjective, *bestTemporaryProjection))
          return completeIterationFailure(std::move(error));
        emitStatistics(loom::mapping_debug::ClosureStatus::TemporaryMapping);
        if (llvm::Error error = consumeIteration())
          return std::move(error);
        return SpatialPathFinderClosureResult{completedIterations, false};
      }
      emitStatistics(
          !projection->routeTerminalsCompatible
              ? loom::mapping_debug::ClosureStatus::RouteTerminalMismatch
          : projection->selectedHandshakeAcyclic
              ? loom::mapping_debug::ClosureStatus::MappingNonclosure
              : loom::mapping_debug::ClosureStatus::SelectedHandshakeCycle);
      if (!projection->routeTerminalsCompatible)
        return pathFinderError(
            "provisional RouteTree terminals disagree with the candidate");
      if (!projection->selectedHandshakeAcyclic) {
        if (llvm::Error error = consumeIteration())
          return std::move(error);
        return llvm::make_error<SpatialPathFinderClosureFailure>(
            SpatialPathFinderClosureFailure::Kind::
                SelectedCombinationalHandshakeCycle,
            "Spatial PathFinder selected a combinational handshake cycle");
      }
      if (llvm::Error error = consumeIteration())
        return std::move(error);
      return llvm::make_error<SpatialPathFinderClosureFailure>(
          SpatialPathFinderClosureFailure::Kind::NonClosure,
          "Spatial PathFinder closed route capacity without Mapping closure");
    }
    if (conflictAnalysis.hasCertificate()) {
      loom::mapping_debug::emit(
          loom::mapping_debug::Level::Decision,
          loom::mapping_debug::Stage::SpatialPnr,
          loom::mapping_debug::Event::MappingFailure,
          [&](llvm::json::Object &fields) {
            fields["iteration"] = iteration;
            fields["session_iteration"] = sessionIteration;
            fields["operation"] = "fixed_terminal_capacity_cut";
            fields["capacity_ref"] = conflictAnalysis.certificateCapacity;
            fields["mandatory_usage"] = conflictAnalysis.mandatoryUsage;
            fields["capacity"] = conflictAnalysis.physicalCapacity;
            fields["temporary_return"] = false;
          });
      emitStatistics(loom::mapping_debug::ClosureStatus::FixedTerminalCut);
      if (llvm::Error error = consumeIteration())
        return std::move(error);
      return llvm::make_error<SpatialPathFinderClosureFailure>(
          SpatialPathFinderClosureFailure::Kind::FixedTerminalCapacityCut,
          "Spatial PathFinder proved fixed-terminal capacity cut at capacity " +
              std::to_string(conflictAnalysis.certificateCapacity) +
              " with mandatory usage " +
              std::to_string(conflictAnalysis.mandatoryUsage) +
              " greater than capacity " +
              std::to_string(conflictAnalysis.physicalCapacity),
          SpatialFixedTerminalCutCertificate{
              conflictAnalysis.certificateCapacity,
              cutCertificateForcedNetCuts_},
          conflictAnalysis.mandatoryUsage, conflictAnalysis.physicalCapacity);
    }
    if (consecutiveNoProgressIterations >= limits.noProgressIterationLimit &&
        trendCount == trendWindow &&
        trendImprovedCount <= trendRegressedCount + trendIneligibleCount) {
      loom::mapping_debug::emit(
          loom::mapping_debug::Level::Decision,
          loom::mapping_debug::Stage::SpatialPnr,
          loom::mapping_debug::Event::MappingFailure,
          [&](llvm::json::Object &fields) {
            fields["iteration"] = iteration;
            fields["session_iteration"] = sessionIteration;
            fields["operation"] = "selected_rank_no_progress";
            fields["consecutive_no_progress_iterations"] =
                consecutiveNoProgressIterations;
            fields["no_progress_iteration_limit"] =
                limits.noProgressIterationLimit;
            fields["no_progress_trend_window"] = limits.noProgressTrendWindow;
            fields["rank_trend_improved_count"] = trendImprovedCount;
            fields["rank_trend_ineligible_count"] = trendIneligibleCount;
            fields["rank_trend_regressed_count"] = trendRegressedCount;
            fields["temporary_return"] = bestTemporaryObjective.has_value();
          });
      if (bestTemporaryObjective) {
        if (llvm::Error error = restoreCapturedRoutes(
                move, candidate, costs, activeLogicalNets(),
                *bestTemporaryObjective, *bestTemporaryProjection))
          return completeIterationFailure(std::move(error));
        emitStatistics(loom::mapping_debug::ClosureStatus::NoProgressTemporary);
        if (llvm::Error error = consumeIteration())
          return std::move(error);
        return SpatialPathFinderClosureResult{completedIterations, false};
      }
      emitStatistics(loom::mapping_debug::ClosureStatus::NoProgress);
      if (llvm::Error error = consumeIteration())
        return std::move(error);
      return llvm::make_error<SpatialPathFinderClosureFailure>(
          SpatialPathFinderClosureFailure::Kind::NoProgress,
          "Spatial PathFinder exhausted its closure-rank no-progress limit "
          "before capacity closure");
    }
    if (completedIterations == limits.iterationLimit) {
      if (bestTemporaryObjective) {
        if (llvm::Error error = restoreCapturedRoutes(
                move, candidate, costs, activeLogicalNets(),
                *bestTemporaryObjective, *bestTemporaryProjection))
          return completeIterationFailure(std::move(error));
        emitStatistics(loom::mapping_debug::ClosureStatus::TemporaryCapacity);
        if (llvm::Error error = consumeIteration())
          return std::move(error);
        return SpatialPathFinderClosureResult{completedIterations, false};
      }
      emitStatistics(loom::mapping_debug::ClosureStatus::IterationLimit);
      if (llvm::Error error = consumeIteration())
        return std::move(error);
      return llvm::make_error<SpatialPathFinderClosureFailure>(
          SpatialPathFinderClosureFailure::Kind::NonClosure,
          "Spatial PathFinder exhausted its iteration limit before capacity "
          "closure");
    }
    if (llvm::Error error = costs.advancePathFinderIteration()) {
      ++debugStatistics.arithmeticFailures;
      loom::mapping_debug::emit(
          loom::mapping_debug::Level::Decision,
          loom::mapping_debug::Stage::SpatialPnr,
          loom::mapping_debug::Event::ArithmeticFailure,
          [&](llvm::json::Object &fields) {
            fields["iteration"] = iteration;
            fields["session_iteration"] = sessionIteration;
            fields["operation"] = "advance_pathfinder_iteration";
            fields["capacity_conflicts"] = capacityConflicts;
          });
      emitStatistics(loom::mapping_debug::ClosureStatus::ArithmeticFailure);
      return completeIterationFailure(std::move(error));
    }
    if (llvm::Error error = consumeIteration())
      return std::move(error);
  }
  llvm_unreachable("positive iteration limit executes or returns");
}

llvm::Expected<RouteCost> SpatialPathFinderRouterScratch::routeWholeNetInMove(
    SpatialMoveTransaction &move, const SpatialCandidateState &candidate,
    SpatialRouteCostState &costs, PnrIndex logicalNet,
    std::uint64_t endpointExpansionLimit,
    std::optional<SpatialTraversalRouteCut> cut) {
  if (!preparedProblem_ || preparedProblem_ != &candidate.problem())
    return pathFinderError("scratch is not prepared for the candidate freeze");
  if (!costs.isBoundTo(candidate))
    return pathFinderError("route costs are bound to another candidate");
  return netRouter_.routeWholeNet(move, candidate, costs, logicalNet,
                                  endpointExpansionLimit, cut);
}

llvm::Expected<RouteCost> SpatialPathFinderRouterScratch::routeSingleSinkInMove(
    SpatialMoveTransaction &move, const SpatialCandidateState &candidate,
    SpatialRouteCostState &costs, PnrIndex logicalNet, PnrIndex sinkObligation,
    std::uint64_t endpointExpansionLimit,
    std::optional<SpatialTraversalRouteCut> cut) {
  if (!preparedProblem_ || preparedProblem_ != &candidate.problem())
    return pathFinderError("scratch is not prepared for the candidate freeze");
  if (!costs.isBoundTo(candidate))
    return pathFinderError("route costs are bound to another candidate");
  return netRouter_.routeSingleSink(move, candidate, costs, logicalNet,
                                    sinkObligation, endpointExpansionLimit,
                                    cut);
}

llvm::Expected<RouteCost>
SpatialPathFinderRouterScratch::routeRootedSubtreeInMove(
    SpatialMoveTransaction &move, const SpatialCandidateState &candidate,
    SpatialRouteCostState &costs, PnrIndex logicalNet, PnrIndex rootEndpoint,
    std::uint64_t endpointExpansionLimit,
    std::optional<SpatialTraversalRouteCut> cut) {
  if (!preparedProblem_ || preparedProblem_ != &candidate.problem())
    return pathFinderError("scratch is not prepared for the candidate freeze");
  if (!costs.isBoundTo(candidate))
    return pathFinderError("route costs are bound to another candidate");
  return netRouter_.routeRootedSubtree(move, candidate, costs, logicalNet,
                                       rootEndpoint, endpointExpansionLimit,
                                       cut);
}

llvm::Error SpatialPathFinderRouterScratch::beginConstraintSweep(
    llvm::ArrayRef<PnrIndex> logicalNets) {
  return netRouter_.beginConstraintSweep(logicalNets);
}

llvm::Error
SpatialPathFinderRouterScratch::finishConstraintNet(PnrIndex logicalNet) {
  return netRouter_.finishConstraintNet(logicalNet);
}

std::size_t SpatialPathFinderRouterScratch::retainedStorageBytes() const {
  return netRouter_.retainedStorageBytes() + retainedBytes(netOrder_) +
         retainedBytes(activeClaimBits_) + retainedBytes(claimEpochs_) +
         retainedBytes(capacityEpochs_) + retainedBytes(capacityNetQCosts_) +
         retainedBytes(touchedCapacities_) +
         retainedBytes(conflictCapacities_) +
         retainedBytes(regionalCapacityMarks_) +
         retainedBytes(regionalTagDomainMarks_) +
         retainedBytes(routingRegionNetMarks_) +
         retainedBytes(routingRegionNets_) +
         retainedBytes(constraintSweepNets_) +
         retainedBytes(capturedSinkPathOffsets_) +
         retainedBytes(capturedForwardArcs_) + retainedBytes(reversePath_) +
         retainedBytes(cutBlockedTraversals_) +
         retainedBytes(cutReachableEndpoints_) +
         retainedBytes(cutSeenTraversals_) + retainedBytes(cutSeenEndpoints_) +
         retainedBytes(cutWorklist_) + retainedBytes(cutContributingNets_) +
         retainedBytes(cutForcedNetCuts_) +
         retainedBytes(cutCertificateForcedNetCuts_) +
         retainedBytes(cutPayloadWidths_) + retainedBytes(cutMinimumClaims_) +
         retainedBytes(rankTrendTransitions_) +
         retainedBytes(cutTouchedClaims_) +
         retainedBytes(cutNetClaimRefcounts_) +
         retainedBytes(cutClaimSelectionCounts_) +
         retainedBytes(cutClaimTraversalRefcounts_) +
         retainedBytes(timingRouteNodeArrivals_) +
         retainedBytes(timingRouteNodeWorklist_);
}
