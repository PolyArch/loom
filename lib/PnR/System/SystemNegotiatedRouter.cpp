#include "SystemNegotiatedRouter.h"

#include "SystemCapacityProjection.h"

#include "Common/MappingDebugLog.h"
#include "Fabric/Identity/FabricRefText.h"
#include "PnR/EndpointRouter.h"
#include "PnR/RoutingNegotiation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Error.h"

#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::pnr;

char detail::SystemRoutingClosureFailure::ID;

void detail::SystemRoutingClosureFailure::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code
detail::SystemRoutingClosureFailure::convertToErrorCode() const {
  return std::make_error_code(std::errc::resource_unavailable_try_again);
}

namespace {

enum class RankTrendTransition : std::uint8_t {
  Equal,
  Improved,
  Regressed,
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "system_routing_negotiation_invalid: " + message);
}

llvm::Error checkedAdd(std::uint64_t amount, std::uint64_t &target,
                       llvm::StringRef subject) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - target)
    return llvm::make_error<RoutingNegotiationError>(
        RoutingNegotiationError::Kind::ArithmeticOverflow,
        ("routing negotiation arithmetic overflow: " + subject +
         " accounting exceeds uint64_t")
            .str());
  target += amount;
  return llvm::Error::success();
}

std::string errorMessage(const llvm::ErrorInfoBase &error) {
  std::string message;
  llvm::raw_string_ostream stream(message);
  error.log(stream);
  return message;
}

llvm::Error
observeArithmeticFailure(llvm::Error error, std::uint64_t iteration,
                         llvm::StringRef operation,
                         mapping_debug::MappingRunStatistics &statistics,
                         mapping_debug::ClosureStatus &closureStatus) {
  return llvm::handleErrors(
      std::move(error),
      [&](const RoutingNegotiationError &failure) -> llvm::Error {
        if (failure.kind() ==
            RoutingNegotiationError::Kind::ArithmeticOverflow) {
          ++statistics.arithmeticFailures;
          closureStatus = mapping_debug::ClosureStatus::ArithmeticFailure;
          mapping_debug::emit(mapping_debug::Level::Decision,
                              mapping_debug::Stage::SystemPnr,
                              mapping_debug::Event::ArithmeticFailure,
                              [&](llvm::json::Object &fields) {
                                fields["iteration"] = iteration;
                                fields["operation"] = operation;
                                fields["failure_kind"] = "routing_negotiation";
                              });
        }
        return llvm::make_error<RoutingNegotiationError>(failure.kind(),
                                                         errorMessage(failure));
      },
      [&](const EndpointRouteSearchFailure &failure) -> llvm::Error {
        if (failure.kind() ==
            EndpointRouteSearchFailureKind::ArithmeticOverflow) {
          ++statistics.arithmeticFailures;
          closureStatus = mapping_debug::ClosureStatus::ArithmeticFailure;
          mapping_debug::emit(
              mapping_debug::Level::Decision, mapping_debug::Stage::SystemPnr,
              mapping_debug::Event::ArithmeticFailure,
              [&](llvm::json::Object &fields) {
                fields["iteration"] = iteration;
                fields["operation"] = operation;
                fields["failure_kind"] = "endpoint_route_search";
              });
        }
        return llvm::make_error<EndpointRouteSearchFailure>(
            failure.kind(), errorMessage(failure));
      });
}

llvm::Expected<std::vector<PnrIndex>>
selectReroutedLegOrder(llvm::ArrayRef<PnrIndex> completeOrder,
                       llvm::ArrayRef<PnrIndex> reroutedLegs,
                       std::size_t legCount) {
  if (reroutedLegs.empty())
    return completeOrder.vec();
  std::vector<std::uint8_t> selected(legCount, 0);
  for (PnrIndex leg : reroutedLegs) {
    if (leg >= legCount || selected[leg]++)
      return invalid("rerouted leg set is not canonical and unique");
  }
  std::vector<PnrIndex> result;
  result.reserve(reroutedLegs.size());
  for (PnrIndex leg : completeOrder)
    if (leg < selected.size() && selected[leg])
      result.push_back(leg);
  if (result.size() != reroutedLegs.size())
    return invalid("rerouted leg set is outside the complete route order");
  return result;
}

bool isCapacityClosed(const FrozenEndpointRoutingTopology &topology,
                      llvm::ArrayRef<std::uint64_t> usage) {
  if (usage.size() != topology.capacityCells().size())
    return false;
  for (PnrIndex cell = 0; cell < usage.size(); ++cell)
    if (usage[cell] > topology.capacityCells()[cell].capacity)
      return false;
  return true;
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 64; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> (shift - 8)));
}

std::vector<std::uint8_t> canonicalRoutingCandidateKey(
    llvm::ArrayRef<PnrIndex> threadChoices,
    llvm::ArrayRef<PnrIndex> graphChoices,
    const detail::CanonicalSystemServiceRoutes &routes) {
  std::vector<std::uint8_t> result;
  const auto appendIndices = [&](llvm::ArrayRef<PnrIndex> values) {
    appendU64(result, values.size());
    for (PnrIndex value : values)
      appendU64(result, value);
  };
  appendIndices(threadChoices);
  appendIndices(graphChoices);
  appendU64(result, routes.routes.size());
  for (const SystemServiceRouteSelection &route : routes.routes) {
    appendU64(result, route.leg);
    appendU64(result, route.rootEndpoint);
    appendU64(result, route.nodeOffset);
    appendU64(result, route.nodeCount);
    appendU64(result, route.sinkOffset);
    appendU64(result, route.sinkCount);
  }
  appendU64(result, routes.nodes.size());
  for (const SystemServiceRouteNodeSelection &node : routes.nodes) {
    appendU64(result, node.endpoint);
    appendU64(result, node.parentNode);
    appendU64(result, node.incomingTraversal);
  }
  appendU64(result, routes.sinks.size());
  for (const SystemServiceRouteSinkSelection &sink : routes.sinks) {
    appendU64(result, sink.terminal);
    appendU64(result, sink.node);
  }
  return result;
}

llvm::json::Array encodeCapacityConflictLogicalNets(
    const FrozenSystemPnrProblem &problem,
    const detail::SystemFixedTerminalCapacityConflict &conflict,
    bool includeReachability) {
  const FrozenEndpointRoutingTopology &topology = problem.routingTopology();
  llvm::json::Array logicalNets;
  for (const detail::SystemFixedTerminalCapacityLegEvidence &evidence :
       conflict.logicalNets) {
    llvm::json::Object logicalNet;
    logicalNet["logical_leg"] = evidence.leg;
    logicalNet["source_endpoint"] = evidence.sourceEndpoint;
    logicalNet["required_payload_width_bits"] =
        problem.serviceLegs()[evidence.leg].requiredPayloadWidthBits;
    logicalNet["minimum_claim"] = evidence.minimumClaim;
    logicalNet["forced"] = evidence.isForced();

    llvm::json::Array sinkEndpoints;
    for (PnrIndex endpoint : evidence.sinkEndpoints)
      sinkEndpoints.push_back(endpoint);
    logicalNet["sink_endpoints"] = std::move(sinkEndpoints);

    llvm::json::Array traversals;
    for (PnrIndex traversalOrdinal : evidence.claimingTraversals) {
      const EndpointRoutingTraversal &traversal =
          topology.traversals()[traversalOrdinal];
      llvm::json::Array claims;
      for (const EndpointRoutingCapacityClaim &claim :
           topology.capacityClaims().slice(traversal.capacityClaimOffset,
                                           traversal.capacityClaimCount)) {
        if (claim.cell != conflict.capacityCell || claim.amount == 0)
          continue;
        llvm::json::Object encodedClaim;
        encodedClaim["activation"] = claim.activation;
        encodedClaim["amount"] = claim.amount;
        encodedClaim["q_cost"] = claim.qCost;
        claims.push_back(std::move(encodedClaim));
      }
      llvm::json::Object encodedTraversal;
      encodedTraversal["traversal"] = traversalOrdinal;
      encodedTraversal["claims"] = std::move(claims);
      traversals.push_back(std::move(encodedTraversal));
    }
    logicalNet["claiming_traversals"] = std::move(traversals);

    if (includeReachability) {
      logicalNet["reachable_endpoint_count"] = evidence.reachableEndpointCount;
      llvm::json::Array unreachableSinkEndpoints;
      for (PnrIndex endpoint : evidence.unreachableSinkEndpoints)
        unreachableSinkEndpoints.push_back(endpoint);
      logicalNet["unreachable_sink_endpoints"] =
          std::move(unreachableSinkEndpoints);
    }
    logicalNets.push_back(std::move(logicalNet));
  }
  return logicalNets;
}

detail::SystemRoutingReopenWitness projectSystemRoutingReopenWitness(
    const detail::SystemFixedTerminalCapacityConflict &certificate) {
  detail::SystemRoutingReopenWitness result;
  result.capacityCell = certificate.capacityCell;
  result.serviceLegs.reserve(certificate.logicalNets.size());
  for (const detail::SystemFixedTerminalCapacityLegEvidence &evidence :
       certificate.logicalNets)
    result.serviceLegs.push_back(evidence.leg);
  llvm::sort(result.serviceLegs);
  result.serviceLegs.erase(
      std::unique(result.serviceLegs.begin(), result.serviceLegs.end()),
      result.serviceLegs.end());
  return result;
}

llvm::Expected<std::vector<RouteCost>>
pathFinderArcCosts(const FrozenEndpointRoutingTopology &topology,
                   llvm::ArrayRef<std::uint64_t> usage,
                   const ResolvedPathFinderPolicy &policy,
                   std::uint64_t presentPressure,
                   llvm::ArrayRef<std::uint64_t> historyPressure) {
  if (usage.size() != topology.capacityCells().size() ||
      historyPressure.size() != topology.capacityCells().size())
    return invalid("PathFinder capacity projection has the wrong width");
  std::vector<RouteCost> traversalCosts(topology.traversals().size(), 0);
  for (const auto &[traversalOrdinal, traversal] :
       llvm::enumerate(topology.traversals())) {
    if (traversal.capacityClaimOffset > topology.capacityClaims().size() ||
        traversal.capacityClaimCount >
            topology.capacityClaims().size() - traversal.capacityClaimOffset)
      return invalid("PathFinder traversal capacity range is out of bounds");
    RouteCost cost = 0;
    for (const auto &claim : topology.capacityClaims().slice(
             traversal.capacityClaimOffset, traversal.capacityClaimCount)) {
      if (claim.cell >= usage.size())
        return invalid("PathFinder claim names an invalid capacity cell");
      if (claim.qCost == 0)
        continue;
      auto overuse = normalizedRouteOveruseCost(
          usage[claim.cell], claim.amount,
          topology.capacityCells()[claim.cell].capacity);
      if (!overuse)
        return overuse.takeError();
      auto term =
          pathFinderResourceCost(policy.priceKernel, claim.qCost, *overuse,
                                 presentPressure, historyPressure[claim.cell]);
      if (!term)
        return term.takeError();
      auto accumulated = accumulateRouteCost(cost, *term);
      if (!accumulated)
        return accumulated.takeError();
      cost = *accumulated;
    }
    traversalCosts[traversalOrdinal] = cost;
  }
  std::vector<RouteCost> arcCosts;
  arcCosts.reserve(topology.arcs().size());
  for (const auto &arc : topology.arcs()) {
    if (arc.traversal >= traversalCosts.size())
      return invalid("PathFinder arc names an invalid traversal");
    arcCosts.push_back(traversalCosts[arc.traversal]);
  }
  return arcCosts;
}

llvm::Expected<std::vector<RouteCost>>
dualArcCosts(const FrozenEndpointRoutingTopology &topology,
             llvm::ArrayRef<DualPrice> prices) {
  if (prices.size() != topology.capacityCells().size())
    return invalid("Dual price projection has the wrong width");
  std::vector<RouteCost> traversalCosts(topology.traversals().size(), 0);
  for (const auto &[traversalOrdinal, traversal] :
       llvm::enumerate(topology.traversals())) {
    if (traversal.capacityClaimOffset > topology.capacityClaims().size() ||
        traversal.capacityClaimCount >
            topology.capacityClaims().size() - traversal.capacityClaimOffset)
      return invalid("Dual traversal capacity range is out of bounds");
    RouteCost cost = 0;
    for (const auto &claim : topology.capacityClaims().slice(
             traversal.capacityClaimOffset, traversal.capacityClaimCount)) {
      if (claim.cell >= prices.size())
        return invalid("Dual claim names an invalid capacity cell");
      if (claim.qCost == 0)
        continue;
      auto term = dualArcResourceCost(claim.qCost, prices[claim.cell]);
      if (!term)
        return term.takeError();
      auto accumulated = accumulateRouteCost(cost, *term);
      if (!accumulated)
        return accumulated.takeError();
      cost = *accumulated;
    }
    traversalCosts[traversalOrdinal] = cost;
  }
  std::vector<RouteCost> arcCosts;
  arcCosts.reserve(topology.arcs().size());
  for (const auto &arc : topology.arcs()) {
    if (arc.traversal >= traversalCosts.size())
      return invalid("Dual arc names an invalid traversal");
    arcCosts.push_back(traversalCosts[arc.traversal]);
  }
  return arcCosts;
}

} // namespace

llvm::Expected<detail::CanonicalSystemServiceRoutes>
detail::negotiateSystemServiceRoutes(
    const FrozenSystemPnrProblem &problem,
    llvm::ArrayRef<PnrIndex> threadChoices,
    llvm::ArrayRef<PnrIndex> graphChoices,
    llvm::ArrayRef<SystemInstructionResourceUseSelection>
        instructionResourceUses,
    llvm::ArrayRef<SystemServiceResourceUseSelection> serviceResourceUses,
    std::uint64_t &endpointExpansions, std::uint64_t &negotiationIterations,
    llvm::ArrayRef<PnrIndex> reroutedLegs,
    std::optional<SystemServiceRoutesView> priorRoutes,
    std::optional<SystemServiceRouteTraversalExclusion> exclusion,
    std::optional<SystemServiceRouteRepairRegion> repairRegion,
    SystemRoutingClosureRequirement closureRequirement,
    std::optional<SystemRoutingReopenWitness> *reopenWitness) {
  endpointExpansions = 0;
  negotiationIterations = 0;
  if (reopenWitness)
    reopenWitness->reset();
  const auto &routing = problem.config().policy().search.routing;
  if (routing.endpointExpansionLimit == 0 ||
      routing.negotiationIterationLimit == 0 ||
      routing.noProgressIterationLimit == 0 ||
      routing.noProgressTrendWindow == 0)
    return invalid("routing work limits must be positive");
  if (routing.noProgressTrendWindow > routing.noProgressIterationLimit ||
      routing.noProgressIterationLimit > routing.negotiationIterationLimit)
    return invalid("routing no-progress limits are not canonical");
  if (routing.noProgressTrendWindow > std::numeric_limits<std::size_t>::max())
    return invalid("routing trend window exceeds host size_t");
  const FrozenEndpointRoutingTopology &topology = problem.routingTopology();

  mapping_debug::MappingRunStatistics debugStatistics;
  mapping_debug::ClosureStatus closureStatus =
      mapping_debug::ClosureStatus::Failed;
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
      mapping_debug::Event::InvocationBegin, [&](llvm::json::Object &fields) {
        fields["logical_leg_count"] = problem.serviceLegs().size();
        fields["negotiation_iteration_limit"] =
            routing.negotiationIterationLimit;
        fields["no_progress_iteration_limit"] =
            routing.noProgressIterationLimit;
        fields["no_progress_trend_window"] = routing.noProgressTrendWindow;
        fields["endpoint_expansion_limit"] = routing.endpointExpansionLimit;
        fields["strict_closure"] =
            closureRequirement == SystemRoutingClosureRequirement::Strict;
      });
  llvm::scope_exit emitFinalDiagnostics([&] {
    debugStatistics.aStarExpansions = endpointExpansions;
    debugStatistics.negotiatedIterations = negotiationIterations;
    mapping_debug::emit(
        mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
        mapping_debug::Event::InvocationEnd, [&](llvm::json::Object &fields) {
          fields["closure_status"] =
              mapping_debug::closureStatusSpelling(closureStatus);
          fields["a_star_expansions"] = endpointExpansions;
          fields["negotiated_iterations"] = negotiationIterations;
        });
    debugStatistics.emit(mapping_debug::Stage::SystemPnr, closureStatus);
  });

  auto lower = buildSystemServiceRouteLowerBoundArcCosts(topology);
  if (!lower)
    return observeArithmeticFailure(lower.takeError(), 0,
                                    "lower_bound_cost_projection",
                                    debugStatistics, closureStatus);

  std::vector<PnrIndex> order(problem.serviceLegs().size());
  std::iota(order.begin(), order.end(), PnrIndex{0});
  auto selectedOrder =
      selectReroutedLegOrder(order, reroutedLegs, problem.serviceLegs().size());
  if (!selectedOrder)
    return selectedOrder.takeError();
  order = std::move(*selectedOrder);
  std::optional<CanonicalSystemServiceRoutes> previous;
  std::optional<CanonicalSystemServiceRoutes> best;
  std::optional<dse::ObjectiveVector> bestObjective;
  std::optional<dse::ObjectiveVector> bestRankObjective;
  std::optional<dse::ObjectiveVector> previousRankObjective;
  std::vector<std::uint8_t> bestKey;
  std::uint64_t consecutiveNoProgressIterations = 0;
  const std::size_t trendWindow =
      static_cast<std::size_t>(routing.noProgressTrendWindow);
  std::vector<std::uint8_t> rankTrendTransitions(trendWindow);
  std::size_t trendHead = 0;
  std::size_t trendCount = 0;
  std::uint64_t trendImprovedCount = 0;
  std::uint64_t trendRegressedCount = 0;
  std::vector<std::uint64_t> previousUsage;
  if (priorRoutes) {
    auto usage =
        measureSystemServiceRouteCapacityUsage(topology, *priorRoutes, false);
    if (!usage)
      return usage.takeError();
    previousUsage = std::move(*usage);
    auto routedOrder =
        buildSystemServiceRouteLegOrder(topology, *priorRoutes, previousUsage);
    if (!routedOrder)
      return routedOrder.takeError();
    selectedOrder = selectReroutedLegOrder(*routedOrder, reroutedLegs,
                                           problem.serviceLegs().size());
    if (!selectedOrder)
      return selectedOrder.takeError();
    order = std::move(*selectedOrder);
  }

  std::vector<std::uint64_t> history(topology.capacityCells().size(), 0);
  std::vector<DualPrice> prices(topology.capacityCells().size(), 0);
  std::vector<DualDirection> previousDirections(topology.capacityCells().size(),
                                                0);
  std::uint64_t presentPressure = 1;
  if (const auto *pathFinder =
          std::get_if<ResolvedPathFinderPolicy>(&routing.negotiation))
    presentPressure = pathFinder->presentPressureInitial;

  for (std::uint64_t iteration = 0;
       iteration < routing.negotiationIterationLimit; ++iteration) {
    ++negotiationIterations;
    std::vector<RouteCost> dualCosts;
    if (std::holds_alternative<ResolvedDualSubgradientPolicy>(
            routing.negotiation)) {
      auto costs = dualArcCosts(topology, prices);
      if (!costs)
        return observeArithmeticFailure(costs.takeError(), iteration,
                                        "dual_arc_cost_projection",
                                        debugStatistics, closureStatus);
      dualCosts = std::move(*costs);
    }
    std::vector<RouteCost> projectedCosts;
    const auto currentCosts = [&](llvm::ArrayRef<std::uint64_t> workingUsage)
        -> llvm::Expected<llvm::ArrayRef<RouteCost>> {
      if (const auto *pathFinder =
              std::get_if<ResolvedPathFinderPolicy>(&routing.negotiation)) {
        auto costs = pathFinderArcCosts(topology, workingUsage, *pathFinder,
                                        presentPressure, history);
        if (!costs)
          return observeArithmeticFailure(costs.takeError(), iteration,
                                          "pathfinder_arc_cost_projection",
                                          debugStatistics, closureStatus);
        projectedCosts = std::move(*costs);
        return projectedCosts;
      }
      return dualCosts;
    };
    std::optional<SystemServiceRoutesView> prior = priorRoutes;
    if (previous)
      prior = SystemServiceRoutesView{previous->routes, previous->nodes,
                                      previous->sinks};
    std::uint64_t iterationExpansions = 0;
    auto built = buildSystemServiceRoutes(
        problem, threadChoices, graphChoices,
        {order, *lower, currentCosts, prior, exclusion, repairRegion, false},
        iterationExpansions);
    if (llvm::Error error = checkedAdd(iterationExpansions, endpointExpansions,
                                       "endpoint expansion"))
      return observeArithmeticFailure(std::move(error), iteration,
                                      "endpoint_expansion_accounting",
                                      debugStatistics, closureStatus);
    if (!built)
      return observeArithmeticFailure(built.takeError(), iteration,
                                      "service_route_search", debugStatistics,
                                      closureStatus);
    auto capacity = problem.capacityModel().project(
        problem, {threadChoices, graphChoices, built->selections.routes,
                  built->selections.nodes, built->selections.sinks,
                  instructionResourceUses, serviceResourceUses});
    if (!capacity)
      return capacity.takeError();
    if (isCapacityClosed(topology, built->capacityUsage)) {
      if (llvm::Error error = verifySystemServiceRoutes(
              problem, threadChoices, graphChoices, built->selections.routes,
              built->selections.nodes, built->selections.sinks))
        return std::move(error);
      mapping_debug::emit(
          mapping_debug::Level::Decision, mapping_debug::Stage::SystemPnr,
          mapping_debug::Event::NegotiationIteration,
          [&](llvm::json::Object &fields) {
            fields["iteration"] = iteration;
            fields["logical_leg_count"] = problem.serviceLegs().size();
            fields["capacity_conflicts"] = 0;
            fields["capacity_closed"] = true;
            fields["a_star_expansions"] = endpointExpansions;
          });
      closureStatus = mapping_debug::ClosureStatus::Closed;
      return std::move(built->selections);
    }
    const bool admitsTemporary =
        closureRequirement ==
            SystemRoutingClosureRequirement::PolicyAdmittedTemporary &&
        llvm::is_contained(
            problem.config().policy().temporaryViolations.admitted,
            ResolvedPnrViolationKind::CapacityOveruse);
    auto traversalClaim = measureSystemServiceRouteTraversalClaim(
        topology, {built->selections.routes, built->selections.nodes,
                   built->selections.sinks});
    if (!traversalClaim)
      return traversalClaim.takeError();
    auto objective = problem.objectiveProgram().evaluateSystemProjection(
        problem, graphChoices, capacity->capacity.total, *traversalClaim,
        capacity->timing.minimumInitiationIntervalCycles,
        capacity->timing.transportBitCycleDemand, capacity->progress);
    if (!objective)
      return objective.takeError();
    bool selectedRankImproved = !bestRankObjective;
    if (bestRankObjective) {
      auto comparison = problem.objectiveProgram().compareSelectedRank(
          *objective, {}, *bestRankObjective, {});
      if (!comparison)
        return comparison.takeError();
      selectedRankImproved = *comparison < 0;
    }
    if (selectedRankImproved) {
      bestRankObjective = *objective;
      consecutiveNoProgressIterations = 0;
    } else {
      ++consecutiveNoProgressIterations;
    }
    if (previousRankObjective) {
      auto comparison = problem.objectiveProgram().compareSelectedRank(
          *objective, {}, *previousRankObjective, {});
      if (!comparison)
        return comparison.takeError();
      const RankTrendTransition transition =
          *comparison < 0   ? RankTrendTransition::Improved
          : *comparison > 0 ? RankTrendTransition::Regressed
                            : RankTrendTransition::Equal;
      if (trendCount == trendWindow) {
        const auto evicted =
            static_cast<RankTrendTransition>(rankTrendTransitions[trendHead]);
        trendImprovedCount -= evicted == RankTrendTransition::Improved ? 1 : 0;
        trendRegressedCount -=
            evicted == RankTrendTransition::Regressed ? 1 : 0;
      } else {
        ++trendCount;
      }
      rankTrendTransitions[trendHead] = static_cast<std::uint8_t>(transition);
      trendHead = (trendHead + 1) % trendWindow;
      trendImprovedCount += transition == RankTrendTransition::Improved ? 1 : 0;
      trendRegressedCount +=
          transition == RankTrendTransition::Regressed ? 1 : 0;
    }
    previousRankObjective = *objective;
    if (admitsTemporary) {
      std::vector<std::uint8_t> key = canonicalRoutingCandidateKey(
          threadChoices, graphChoices, built->selections);
      bool replace = !best;
      if (best) {
        auto comparison = problem.objectiveProgram().compareSelectedRank(
            *objective, key, *bestObjective, bestKey);
        if (!comparison)
          return comparison.takeError();
        replace = *comparison < 0;
      }
      if (replace) {
        best = built->selections;
        bestObjective = std::move(*objective);
        bestKey = std::move(key);
      }
    }

    auto conflicts = analyzeSystemFixedTerminalCapacityConflicts(
        problem,
        {built->selections.routes, built->selections.nodes,
         built->selections.sinks},
        built->capacityUsage);
    if (!conflicts)
      return observeArithmeticFailure(conflicts.takeError(), iteration,
                                      "fixed_terminal_capacity_cut",
                                      debugStatistics, closureStatus);
    debugStatistics.capacityConflicts += conflicts->size();
    const SystemFixedTerminalCapacityConflict *certificate = nullptr;
    for (const SystemFixedTerminalCapacityConflict &conflict : *conflicts) {
      const EndpointRoutingCapacityCell &cell =
          topology.capacityCells()[conflict.capacityCell];
      mapping_debug::emit(
          mapping_debug::Level::Decision, mapping_debug::Stage::SystemPnr,
          mapping_debug::Event::CapacityConflict,
          [&](llvm::json::Object &fields) {
            llvm::json::Array contributingLegs;
            llvm::json::Array forcedLegs;
            for (const SystemFixedTerminalCapacityLegEvidence &evidence :
                 conflict.logicalNets) {
              contributingLegs.push_back(evidence.leg);
              if (evidence.isForced())
                forcedLegs.push_back(evidence.leg);
            }
            fields["iteration"] = iteration;
            fields["capacity_ref"] = conflict.capacityCell;
            fields["usage"] = conflict.usage;
            fields["capacity"] = conflict.capacity;
            fields["overuse"] = conflict.usage - conflict.capacity;
            fields["initial_occupancy"] = cell.initialOccupancy;
            fields["mandatory_usage_lower_bound"] = conflict.mandatoryUsage;
            fields["fixed_terminal_cut_certificate"] =
                conflict.hasCertificate();
            fields["contributing_logical_legs"] = std::move(contributingLegs);
            fields["forced_logical_legs"] = std::move(forcedLegs);
            fields["logical_nets"] = encodeCapacityConflictLogicalNets(
                problem, conflict, /*includeReachability=*/false);
            fields["resource_owner_ref"] = fabric::printFabricRef(cell.owner);
            fields["resource_state"] = cell.state.ordinal();
            fields["capacity_dimension"] = cell.dimension.ordinal();
          });
      mapping_debug::emit(
          mapping_debug::Level::Detail, mapping_debug::Stage::SystemPnr,
          mapping_debug::Event::CutAnalysis, [&](llvm::json::Object &fields) {
            llvm::json::Array contributingLegs;
            llvm::json::Array forcedLegs;
            for (const SystemFixedTerminalCapacityLegEvidence &evidence :
                 conflict.logicalNets) {
              contributingLegs.push_back(evidence.leg);
              if (evidence.isForced())
                forcedLegs.push_back(evidence.leg);
            }
            fields["analysis_scope"] = "fixed_terminal_capacity_certificate";
            fields["iteration"] = iteration;
            fields["capacity_ref"] = conflict.capacityCell;
            fields["mandatory_usage_lower_bound"] = conflict.mandatoryUsage;
            fields["capacity"] = conflict.capacity;
            fields["fixed_terminal_cut_certificate"] =
                conflict.hasCertificate();
            fields["contributing_logical_legs"] = std::move(contributingLegs);
            fields["forced_logical_legs"] = std::move(forcedLegs);
            fields["logical_nets"] = encodeCapacityConflictLogicalNets(
                problem, conflict, /*includeReachability=*/true);
            fields["resource_owner_ref"] = fabric::printFabricRef(cell.owner);
            fields["resource_state"] = cell.state.ordinal();
            fields["capacity_dimension"] = cell.dimension.ordinal();
          });
      if (!certificate && conflict.hasCertificate())
        certificate = &conflict;
    }
    mapping_debug::emit(
        mapping_debug::Level::Decision, mapping_debug::Stage::SystemPnr,
        mapping_debug::Event::NegotiationIteration,
        [&](llvm::json::Object &fields) {
          fields["iteration"] = iteration;
          fields["logical_leg_count"] = problem.serviceLegs().size();
          fields["capacity_conflicts"] = conflicts->size();
          fields["capacity_closed"] = false;
          fields["fixed_terminal_cut_certificate"] = certificate != nullptr;
          fields["selected_rank_improved"] = selectedRankImproved;
          fields["consecutive_no_progress_iterations"] =
              consecutiveNoProgressIterations;
          fields["no_progress_iteration_limit"] =
              routing.noProgressIterationLimit;
          fields["no_progress_trend_window"] = routing.noProgressTrendWindow;
          fields["rank_trend_transition_count"] = trendCount;
          fields["rank_trend_improved_count"] = trendImprovedCount;
          fields["rank_trend_regressed_count"] = trendRegressedCount;
          fields["a_star_expansions"] = endpointExpansions;
        });
    if (certificate) {
      mapping_debug::emit(
          mapping_debug::Level::Decision, mapping_debug::Stage::SystemPnr,
          mapping_debug::Event::MappingFailure,
          [&](llvm::json::Object &fields) {
            fields["iteration"] = iteration;
            fields["operation"] = "fixed_terminal_capacity_cut";
            fields["capacity_ref"] = certificate->capacityCell;
            fields["mandatory_usage"] = certificate->mandatoryUsage;
            fields["capacity"] = certificate->capacity;
            fields["temporary_return"] = best.has_value();
          });
      SystemRoutingReopenWitness projectedWitness =
          projectSystemRoutingReopenWitness(*certificate);
      if (reopenWitness)
        *reopenWitness = projectedWitness;
      if (best) {
        closureStatus = mapping_debug::ClosureStatus::FixedTerminalCutTemporary;
        return std::move(*best);
      }
      closureStatus = mapping_debug::ClosureStatus::FixedTerminalCut;
      return llvm::make_error<SystemRoutingClosureFailure>(
          SystemRoutingClosureFailureKind::FixedTerminalCapacityCut,
          "System routing negotiation proved fixed-terminal capacity cut at "
          "capacity " +
              std::to_string(certificate->capacityCell) +
              " with mandatory usage " +
              std::to_string(certificate->mandatoryUsage) +
              " greater than capacity " + std::to_string(certificate->capacity),
          std::move(projectedWitness));
    }
    if (consecutiveNoProgressIterations >= routing.noProgressIterationLimit &&
        trendCount == trendWindow &&
        trendImprovedCount <= trendRegressedCount) {
      mapping_debug::emit(
          mapping_debug::Level::Decision, mapping_debug::Stage::SystemPnr,
          mapping_debug::Event::MappingFailure,
          [&](llvm::json::Object &fields) {
            fields["iteration"] = iteration;
            fields["operation"] = "selected_rank_no_progress";
            fields["consecutive_no_progress_iterations"] =
                consecutiveNoProgressIterations;
            fields["no_progress_iteration_limit"] =
                routing.noProgressIterationLimit;
            fields["no_progress_trend_window"] = routing.noProgressTrendWindow;
            fields["rank_trend_improved_count"] = trendImprovedCount;
            fields["rank_trend_regressed_count"] = trendRegressedCount;
            fields["temporary_return"] = best.has_value();
          });
      if (best) {
        closureStatus = mapping_debug::ClosureStatus::NoProgressTemporary;
        return std::move(*best);
      }
      closureStatus = mapping_debug::ClosureStatus::NoProgress;
      return llvm::make_error<SystemRoutingClosureFailure>(
          SystemRoutingClosureFailureKind::NoProgress,
          "System routing negotiation exhausted its selected-rank "
          "no-progress limit before capacity closure");
    }
    if (iteration + 1 == routing.negotiationIterationLimit) {
      if (best) {
        closureStatus = mapping_debug::ClosureStatus::TemporaryCapacity;
        return std::move(*best);
      }
      closureStatus = mapping_debug::ClosureStatus::IterationLimit;
      return llvm::make_error<SystemRoutingClosureFailure>(
          SystemRoutingClosureFailureKind::NonClosure,
          "System routing negotiation exhausted its iteration limit before "
          "capacity closure");
    }

    if (const auto *pathFinder =
            std::get_if<ResolvedPathFinderPolicy>(&routing.negotiation)) {
      for (PnrIndex cell = 0; cell < history.size(); ++cell) {
        auto overuse =
            normalizedRouteOveruseCost(built->capacityUsage[cell], 0,
                                       topology.capacityCells()[cell].capacity);
        if (!overuse)
          return observeArithmeticFailure(overuse.takeError(), iteration,
                                          "pathfinder_overuse_projection",
                                          debugStatistics, closureStatus);
        auto next = pathFinderHistoryUpdate(
            history[cell], pathFinder->historyPressureIncrement, *overuse);
        if (!next)
          return observeArithmeticFailure(next.takeError(), iteration,
                                          "pathfinder_history_update",
                                          debugStatistics, closureStatus);
        history[cell] = *next;
      }
      auto nextPressure = ceilMulDiv(
          presentPressure, pathFinder->presentPressureGrowth.numerator,
          pathFinder->presentPressureGrowth.denominator);
      if (!nextPressure)
        return observeArithmeticFailure(nextPressure.takeError(), iteration,
                                        "pathfinder_present_pressure_update",
                                        debugStatistics, closureStatus);
      presentPressure = *nextPressure;
    } else {
      const auto &dual =
          std::get<ResolvedDualSubgradientPolicy>(routing.negotiation);
      auto step = dualStepAt(dual.stepSchedule, iteration);
      if (!step)
        return observeArithmeticFailure(step.takeError(), iteration,
                                        "dual_step_projection", debugStatistics,
                                        closureStatus);
      for (PnrIndex cell = 0; cell < prices.size(); ++cell) {
        auto residual = dualResidual(built->capacityUsage[cell],
                                     topology.capacityCells()[cell].capacity);
        if (!residual)
          return observeArithmeticFailure(residual.takeError(), iteration,
                                          "dual_residual_projection",
                                          debugStatistics, closureStatus);
        auto direction = dualDirectionFromResidual(dual, *residual,
                                                   previousDirections[cell]);
        if (!direction)
          return observeArithmeticFailure(direction.takeError(), iteration,
                                          "dual_direction_update",
                                          debugStatistics, closureStatus);
        auto price = dualPriceUpdate(prices[cell], *step, *direction);
        if (!price)
          return observeArithmeticFailure(price.takeError(), iteration,
                                          "dual_price_update", debugStatistics,
                                          closureStatus);
        prices[cell] = *price;
        previousDirections[cell] =
            dual.directionKernel ==
                    ResolvedDualDirectionKernel::MomentumDeflected
                ? *direction
                : 0;
      }
    }

    previousUsage = built->capacityUsage;
    previous = std::move(built->selections);
    auto nextOrder = buildSystemServiceRouteLegOrder(
        topology, {previous->routes, previous->nodes, previous->sinks},
        previousUsage);
    if (!nextOrder)
      return nextOrder.takeError();
    selectedOrder = selectReroutedLegOrder(*nextOrder, reroutedLegs,
                                           problem.serviceLegs().size());
    if (!selectedOrder)
      return selectedOrder.takeError();
    order = std::move(*selectedOrder);
  }
  llvm_unreachable("positive System negotiation limit executes or returns");
}
