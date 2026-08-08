#include "SystemNegotiatedRouter.h"

#include "SystemCapacityProjection.h"

#include "PnR/RoutingNegotiation.h"

#include "llvm/ADT/STLExtras.h"
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

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "system_routing_negotiation_invalid: " + message);
}

llvm::Error checkedAdd(std::uint64_t amount, std::uint64_t &target,
                       llvm::StringRef subject) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - target)
    return invalid(subject + " accounting overflows u64");
  target += amount;
  return llvm::Error::success();
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
    SystemRoutingClosureRequirement closureRequirement) {
  endpointExpansions = 0;
  negotiationIterations = 0;
  const auto &routing = problem.config().policy().search.routing;
  if (routing.endpointExpansionLimit == 0 ||
      routing.negotiationIterationLimit == 0)
    return invalid("routing work limits must be positive");
  const FrozenEndpointRoutingTopology &topology = problem.routingTopology();
  auto lower = buildSystemServiceRouteLowerBoundArcCosts(topology);
  if (!lower)
    return lower.takeError();

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
  std::vector<std::uint8_t> bestKey;
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
        return costs.takeError();
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
          return costs.takeError();
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
      return std::move(error);
    if (!built)
      return built.takeError();
    auto capacity = problem.capacityModel().project(
        problem, {threadChoices, graphChoices, built->selections.routes,
                  built->selections.nodes, instructionResourceUses,
                  serviceResourceUses});
    if (!capacity)
      return capacity.takeError();
    if (isCapacityClosed(topology, built->capacityUsage)) {
      if (llvm::Error error = verifySystemServiceRoutes(
              problem, threadChoices, graphChoices, built->selections.routes,
              built->selections.nodes, built->selections.sinks))
        return std::move(error);
      return std::move(built->selections);
    }
    const bool admitsTemporary =
        closureRequirement ==
            SystemRoutingClosureRequirement::PolicyAdmittedTemporary &&
        llvm::is_contained(
            problem.config().policy().temporaryViolations.admitted,
            ResolvedPnrViolationKind::CapacityOveruse);
    if (admitsTemporary) {
      auto traversalClaim = measureSystemServiceRouteTraversalClaim(
          topology, {built->selections.routes, built->selections.nodes,
                     built->selections.sinks});
      if (!traversalClaim)
        return traversalClaim.takeError();
      auto objective = problem.objectiveProgram().evaluateSystemProjection(
          problem, capacity->total, *traversalClaim);
      if (!objective)
        return objective.takeError();
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
    if (iteration + 1 == routing.negotiationIterationLimit) {
      if (best)
        return std::move(*best);
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
          return overuse.takeError();
        auto next = pathFinderHistoryUpdate(
            history[cell], pathFinder->historyPressureIncrement, *overuse);
        if (!next)
          return next.takeError();
        history[cell] = *next;
      }
      auto nextPressure = ceilMulDiv(
          presentPressure, pathFinder->presentPressureGrowth.numerator,
          pathFinder->presentPressureGrowth.denominator);
      if (!nextPressure)
        return nextPressure.takeError();
      presentPressure = *nextPressure;
    } else {
      const auto &dual =
          std::get<ResolvedDualSubgradientPolicy>(routing.negotiation);
      auto step = dualStepAt(dual.stepSchedule, iteration);
      if (!step)
        return step.takeError();
      for (PnrIndex cell = 0; cell < prices.size(); ++cell) {
        auto residual = dualResidual(built->capacityUsage[cell],
                                     topology.capacityCells()[cell].capacity);
        if (!residual)
          return residual.takeError();
        auto direction = dualDirectionFromResidual(dual, *residual,
                                                   previousDirections[cell]);
        if (!direction)
          return direction.takeError();
        auto price = dualPriceUpdate(prices[cell], *step, *direction);
        if (!price)
          return price.takeError();
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
