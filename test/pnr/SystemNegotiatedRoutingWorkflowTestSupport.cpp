#include "SystemCandidateStateTestSupport.h"

#include "ADG/Builder.h"
#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "PnR/MappingObjective.h"
#include "PnR/PnrConfig.h"
#include "PnR/System/SystemActionDomain.h"
#include "PnR/System/SystemActionExecutor.h"
#include "PnR/System/SystemAnnealingSearch.h"
#include "PnR/System/SystemCandidateServiceResolver.h"
#include "PnR/System/SystemCapacityProjection.h"
#include "PnR/System/SystemMappingMaterializer.h"
#include "PnR/System/SystemPnrProblem.h"
#include "PnR/System/SystemPnrSearchDomain.h"
#include "PnR/System/SystemServiceRouter.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using namespace loom;
using namespace loom::pnr;

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "System negotiated routing workflow failed: " << message
               << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

std::uint64_t maximumPatternMultiplicity(
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::fabric::FabricUsePatternRef &reference) {
  const auto *contract = fabric.resourceContract(reference.owner.catalog());
  require(contract && reference.ordinal < contract->usePatternCount(),
          "capacity fixture selected a foreign ResourceUse pattern");
  const auto pattern =
      contract->usePattern(::fabric::UsePatternKey(reference.ordinal));
  std::uint64_t result = std::numeric_limits<std::uint64_t>::max();
  for (const auto &claim : pattern.claims) {
    require(claim.state.ordinal() < contract->stateCount(),
            "capacity fixture selected a foreign resource state");
    const auto dimensions = contract->capacityDimensions(claim.state);
    require(claim.dimension.ordinal() < dimensions.size(),
            "capacity fixture selected a foreign capacity dimension");
    const auto &dimension = dimensions[claim.dimension.ordinal()];
    const std::uint64_t amount = claim.amount.value();
    if (amount == 0)
      continue;
    require(dimension.initialOccupancy.value() <= dimension.capacity.value(),
            "capacity fixture has invalid initial occupancy");
    result = std::min(result, (dimension.capacity.value() -
                               dimension.initialOccupancy.value()) /
                                  amount);
  }
  return result;
}

template <typename Selection>
std::vector<Selection>
repeatedOveruseSelections(const ::loom::fabric::FabricArtifactView &fabric,
                          llvm::ArrayRef<Selection> selections,
                          llvm::StringRef label) {
  for (const Selection &selection : selections) {
    const std::uint64_t multiplicity =
        maximumPatternMultiplicity(fabric, selection.pattern);
    if (multiplicity != std::numeric_limits<std::uint64_t>::max() &&
        multiplicity < 4096)
      return std::vector<Selection>(multiplicity + 1, selection);
  }
  fail(label + " fixture has no bounded ResourceUse pattern");
}

::loom::mapping::detail::ResourceCapacityOveruseProjection projectCapacity(
    const SystemCandidateState &candidate,
    llvm::ArrayRef<SystemServiceRouteSelection> routes,
    llvm::ArrayRef<SystemServiceRouteNodeSelection> routeNodes,
    llvm::ArrayRef<SystemInstructionResourceUseSelection> instructionUses,
    llvm::ArrayRef<SystemServiceResourceUseSelection> serviceUses) {
  return take(candidate.problem().capacityModel().project(
                  candidate.problem(),
                  {candidate.threadChoices(), candidate.graphChoices(), routes,
                   routeNodes, candidate.serviceRouteSinks(), instructionUses,
                   serviceUses}))
      .capacity;
}

::loom::mapping::detail::ResourceCapacityOveruseProjection
projectImportedCapacity(const SystemCandidateState &candidate,
                        llvm::ArrayRef<PnrIndex> threadChoices,
                        llvm::ArrayRef<PnrIndex> graphChoices) {
  return take(candidate.problem().capacityModel().project(
                  candidate.problem(),
                  {threadChoices, graphChoices, {}, {}, {}, {}, {}}))
      .capacity;
}

bool traversalClaimsCell(const FrozenEndpointRoutingTopology &topology,
                         PnrIndex traversal, PnrIndex capacityCell) {
  require(traversal < topology.traversals().size(),
          "selected route has a foreign traversal");
  const auto &record = topology.traversals()[traversal];
  require(record.capacityClaimOffset <= topology.capacityClaims().size() &&
              record.capacityClaimCount <=
                  topology.capacityClaims().size() - record.capacityClaimOffset,
          "selected route has an invalid capacity claim range");
  return llvm::any_of(
      topology.capacityClaims().slice(record.capacityClaimOffset,
                                      record.capacityClaimCount),
      [&](const EndpointRoutingCapacityClaim &claim) {
        return claim.cell == capacityCell;
      });
}

std::vector<PnrIndex> witnessLegs(const SystemCandidateState &candidate,
                                  PnrIndex capacityCell) {
  std::vector<PnrIndex> result;
  const auto &topology = candidate.problem().routingTopology();
  for (const SystemServiceRouteSelection &route : candidate.serviceRoutes()) {
    const auto nodes =
        candidate.serviceRouteNodes().slice(route.nodeOffset, route.nodeCount);
    if (llvm::any_of(nodes, [&](const SystemServiceRouteNodeSelection &node) {
          return node.incomingTraversal != getInvalidPnrIndex() &&
                 traversalClaimsCell(topology, node.incomingTraversal,
                                     capacityCell);
        }))
      result.push_back(route.leg);
  }
  return result;
}

using ClaimKey = std::pair<PnrIndex, PnrIndex>;

std::map<ClaimKey, std::uint64_t>
routeClaims(const FrozenEndpointRoutingTopology &topology,
            llvm::ArrayRef<SystemServiceRouteNodeSelection> nodes) {
  std::map<ClaimKey, std::uint64_t> claims;
  for (const auto &node : nodes) {
    if (node.incomingTraversal == getInvalidPnrIndex())
      continue;
    const auto &traversal = topology.traversals()[node.incomingTraversal];
    for (const auto &claim : topology.capacityClaims().slice(
             traversal.capacityClaimOffset, traversal.capacityClaimCount)) {
      const ClaimKey key{claim.activation, claim.cell};
      auto [position, inserted] = claims.try_emplace(key, claim.amount);
      require(inserted || position->second == claim.amount,
              "one route activation has inconsistent capacity claims");
    }
  }
  return claims;
}

std::vector<std::uint64_t>
selectedUsage(const SystemCandidateState &candidate) {
  const auto &topology = candidate.problem().routingTopology();
  std::vector<std::uint64_t> usage;
  usage.reserve(topology.capacityCells().size());
  for (const auto &cell : topology.capacityCells())
    usage.push_back(cell.initialOccupancy);
  for (const auto &route : candidate.serviceRoutes()) {
    const auto claims = routeClaims(
        topology,
        candidate.serviceRouteNodes().slice(route.nodeOffset, route.nodeCount));
    for (const auto &[key, amount] : claims) {
      require(key.second < usage.size() &&
                  amount <= std::numeric_limits<std::uint64_t>::max() -
                                usage[key.second],
              "selected route capacity usage overflows u64");
      usage[key.second] += amount;
    }
  }
  return usage;
}

bool findCapacityFeasibleAlternatePath(const SystemCandidateState &candidate,
                                       PnrIndex leg,
                                       PnrIndex forbiddenCapacityCell) {
  const auto &topology = candidate.problem().routingTopology();
  const auto route =
      llvm::find_if(candidate.serviceRoutes(),
                    [&](const auto &value) { return value.leg == leg; });
  require(route != candidate.serviceRoutes().end() && route->sinkCount == 1,
          "bottleneck fixture requires one exact sink on the repaired leg");
  const auto routeNodes =
      candidate.serviceRouteNodes().slice(route->nodeOffset, route->nodeCount);
  const auto routeSinks =
      candidate.serviceRouteSinks().slice(route->sinkOffset, route->sinkCount);
  require(routeSinks.front().node < routeNodes.size(),
          "bottleneck fixture has a foreign sink node");
  const PnrIndex source = route->rootEndpoint;
  const PnrIndex target = routeNodes[routeSinks.front().node].endpoint;
  std::vector<std::uint64_t> baseUsage = selectedUsage(candidate);
  for (const auto &[key, amount] : routeClaims(topology, routeNodes)) {
    require(key.second < baseUsage.size() && amount <= baseUsage[key.second],
            "route removal underflows selected capacity usage");
    baseUsage[key.second] -= amount;
  }

  std::vector<std::uint8_t> visited(topology.endpoints().size(), 0);
  std::vector<PnrIndex> path;
  std::function<bool(PnrIndex)> search = [&](PnrIndex endpoint) {
    if (endpoint == target && path.size() >= 2) {
      std::map<ClaimKey, std::uint64_t> claims;
      for (PnrIndex traversalOrdinal : path) {
        const auto &traversal = topology.traversals()[traversalOrdinal];
        for (const auto &claim : topology.capacityClaims().slice(
                 traversal.capacityClaimOffset, traversal.capacityClaimCount)) {
          if (claim.cell == forbiddenCapacityCell)
            return false;
          const ClaimKey key{claim.activation, claim.cell};
          auto [position, inserted] = claims.try_emplace(key, claim.amount);
          if (!inserted && position->second != claim.amount)
            return false;
        }
      }
      for (const auto &[key, amount] : claims)
        if (key.second >= baseUsage.size() ||
            baseUsage[key.second] >
                topology.capacityCells()[key.second].capacity ||
            amount > topology.capacityCells()[key.second].capacity -
                         baseUsage[key.second])
          return false;
      return true;
    }
    if (endpoint >= topology.endpoints().size() || visited[endpoint])
      return false;
    visited[endpoint] = 1;
    const PnrIndex begin = topology.adjacencyOffsets()[endpoint];
    const PnrIndex end = topology.adjacencyOffsets()[endpoint + 1];
    for (PnrIndex arc = begin; arc < end; ++arc) {
      const auto &record = topology.arcs()[arc];
      if (record.payloadCapacityBits <
              candidate.problem().serviceLegs()[leg].requiredPayloadWidthBits ||
          traversalClaimsCell(topology, record.traversal,
                              forbiddenCapacityCell))
        continue;
      path.push_back(record.traversal);
      if (search(record.target))
        return true;
      path.pop_back();
    }
    visited[endpoint] = 0;
    return false;
  };
  return search(source);
}

bool sameRoute(const SystemCandidateState &lhs, const SystemCandidateState &rhs,
               PnrIndex leg) {
  const auto left = llvm::find_if(
      lhs.serviceRoutes(), [&](const auto &route) { return route.leg == leg; });
  const auto right = llvm::find_if(
      rhs.serviceRoutes(), [&](const auto &route) { return route.leg == leg; });
  if (left == lhs.serviceRoutes().end() || right == rhs.serviceRoutes().end() ||
      left->rootEndpoint != right->rootEndpoint ||
      left->nodeCount != right->nodeCount ||
      left->sinkCount != right->sinkCount)
    return false;
  const auto leftNodes =
      lhs.serviceRouteNodes().slice(left->nodeOffset, left->nodeCount);
  const auto rightNodes =
      rhs.serviceRouteNodes().slice(right->nodeOffset, right->nodeCount);
  for (const auto &[leftNode, rightNode] :
       llvm::zip_equal(leftNodes, rightNodes))
    if (leftNode.endpoint != rightNode.endpoint ||
        leftNode.parentNode != rightNode.parentNode ||
        leftNode.incomingTraversal != rightNode.incomingTraversal)
      return false;
  const auto leftSinks =
      lhs.serviceRouteSinks().slice(left->sinkOffset, left->sinkCount);
  const auto rightSinks =
      rhs.serviceRouteSinks().slice(right->sinkOffset, right->sinkCount);
  for (const auto &[leftSink, rightSink] :
       llvm::zip_equal(leftSinks, rightSinks))
    if (leftSink.terminal != rightSink.terminal ||
        leftSink.node != rightSink.node)
      return false;
  return true;
}

bool sameRoutes(const detail::CanonicalSystemServiceRoutes &lhs,
                const detail::CanonicalSystemServiceRoutes &rhs) {
  if (lhs.routes.size() != rhs.routes.size() ||
      lhs.nodes.size() != rhs.nodes.size() ||
      lhs.sinks.size() != rhs.sinks.size())
    return false;
  for (const auto &[left, right] : llvm::zip_equal(lhs.routes, rhs.routes))
    if (std::tie(left.leg, left.rootEndpoint, left.nodeOffset, left.nodeCount,
                 left.sinkOffset, left.sinkCount) !=
        std::tie(right.leg, right.rootEndpoint, right.nodeOffset,
                 right.nodeCount, right.sinkOffset, right.sinkCount))
      return false;
  for (const auto &[left, right] : llvm::zip_equal(lhs.nodes, rhs.nodes))
    if (std::tie(left.endpoint, left.parentNode, left.incomingTraversal) !=
        std::tie(right.endpoint, right.parentNode, right.incomingTraversal))
      return false;
  for (const auto &[left, right] : llvm::zip_equal(lhs.sinks, rhs.sinks))
    if (std::tie(left.terminal, left.node) !=
        std::tie(right.terminal, right.node))
      return false;
  return true;
}

void verifyServiceRouterScratchReuse(const SystemCandidateState &baseline) {
  const FrozenSystemPnrProblem &problem = baseline.problem();
  detail::SystemServiceRouterScratch scratch;
  if (llvm::Error error = scratch.prepare(problem))
    fail(llvm::toString(std::move(error)));

  std::vector<PnrIndex> order(problem.serviceLegs().size());
  std::iota(order.begin(), order.end(), PnrIndex{0});
  const auto currentCosts = [&](llvm::ArrayRef<std::uint64_t>)
      -> llvm::Expected<llvm::ArrayRef<RouteCost>> {
    return scratch.lowerBoundArcCosts();
  };
  const detail::SystemServiceRouteBuildRequest request{
      order, currentCosts, std::nullopt, std::nullopt, std::nullopt, {}, false};
  std::uint64_t coldExpansions = 0;
  auto cold = take(detail::buildSystemServiceRoutes(
      problem, baseline.threadChoices(), baseline.graphChoices(), scratch,
      request, coldExpansions));
  const std::uint64_t coldBuilds = scratch.heuristicBuildCount();
  const std::uint64_t coldHits = scratch.heuristicCacheHitCount();

  std::uint64_t warmExpansions = 0;
  auto warm = take(detail::buildSystemServiceRoutes(
      problem, baseline.threadChoices(), baseline.graphChoices(), scratch,
      request, warmExpansions));
  require(scratch.heuristicBuildCount() == coldBuilds &&
              scratch.heuristicCacheHitCount() > coldHits,
          "prepared service router did not reuse exact endpoint heuristics");
  require(coldExpansions == warmExpansions &&
              sameRoutes(cold.selections, warm.selections),
          "warm service routing changed canonical work or route selections");

  const auto verifyCold =
      [&](const detail::CanonicalSystemServiceRoutes &routes) {
        if (llvm::Error error = detail::verifySystemServiceRoutes(
                problem, baseline.threadChoices(), baseline.graphChoices(),
                routes.routes, routes.nodes, routes.sinks))
          fail(llvm::toString(std::move(error)));
      };
  verifyCold(cold.selections);
  verifyCold(warm.selections);

  const auto candidateFor =
      [&](const detail::CanonicalSystemServiceRoutes &routes) {
        return take(SystemCandidateState::create(
            baseline.problemHandle(),
            {baseline.threadChoices(), baseline.graphChoices(), routes.routes,
             routes.nodes, routes.sinks, baseline.serviceTargets(),
             baseline.instructionResourceUses(),
             baseline.serviceResourceUses()}));
      };
  auto coldCandidate = candidateFor(cold.selections);
  auto warmCandidate = candidateFor(warm.selections);
  const auto coldObjective =
      take(problem.objectiveProgram().evaluate(*coldCandidate));
  const auto warmObjective =
      take(problem.objectiveProgram().evaluate(*warmCandidate));
  require(coldObjective.codes() == warmObjective.codes(),
          "warm service routing changed the exact objective vector");
}

struct WorkflowProblem final {
  ResolvedPnrConfigView config;
  SystemPnrSearchDomainView searchDomain;
  FrozenSystemPnrProblemHandle problem;
};

WorkflowProblem buildProblem(
    const ResolvedConfig &base, std::uint64_t iterationLimit,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &system,
    const ::loom::mapping::FinalizedSystemMappingConstraintSet &constraints,
    const SystemBindingPartitionPlan &partition,
    const ArtifactRootReference &spatialMapping, const ArtifactStore &store,
    bool admitTemporary = true) {
  ResolvedConfig resolved = base;
  auto &routing = resolved.dse.systemPnr.search.routing;
  routing.negotiationIterationLimit = iterationLimit;
  routing.noProgressIterationLimit =
      std::min(routing.noProgressIterationLimit, iterationLimit);
  routing.noProgressTrendWindow =
      std::min(routing.noProgressTrendWindow,
               routing.noProgressIterationLimit);
  resolved.dse.systemPnr.temporaryViolations.admitted.clear();
  if (admitTemporary)
    resolved.dse.systemPnr.temporaryViolations.admitted = {
        ResolvedPnrViolationKind::CapacityOveruse};
  auto config = take(projectResolvedSystemPnrConfigView(resolved));
  auto searchDomain = take(projectSystemPnrSearchDomain(
      dataflow, system, config, constraints, partition,
      SystemHierarchicalGraphSearchInput{{spatialMapping}}, store));
  auto problem = take(freezeSystemPnrProblemWithNormalizedTiming(
      dataflow, system, searchDomain, config, constraints, store));
  return {std::move(config), std::move(searchDomain), std::move(problem)};
}

struct BottleneckCandidate final {
  SystemCandidateStateHandle state;
  std::uint64_t endpointExpansions = 0;
  std::uint64_t negotiationIterations = 0;
};

BottleneckCandidate
initializeBottleneckCandidate(const FrozenSystemPnrProblemHandle &problem) {
  const auto binding =
      llvm::find_if(problem->memoryServiceBindings(), [](const auto &record) {
        return llvm::any_of(record.usePatternDomains, [](const auto &domain) {
          return domain.patterns.size() > 1;
        });
      });
  require(binding != problem->memoryServiceBindings().end(),
          "custom bottleneck Fabric has no selectable memory binding");
  const auto corePosition = llvm::find(problem->accCores(), binding->accCore);
  require(corePosition != problem->accCores().end(),
          "custom bottleneck AccCore is absent from the frozen catalog");
  const PnrIndex targetCore =
      static_cast<PnrIndex>(corePosition - problem->accCores().begin());
  const PnrIndex targetClass = problem->accCoreTargetClass(targetCore);

  std::vector<PnrIndex> threadChoices(problem->threadDecisions().size(), 0);
  for (PnrIndex decision = 0; decision < problem->threadDecisions().size();
       ++decision) {
    const auto choices = problem->threadChoiceCatalogOrdinals(decision);
    const auto selected = llvm::find_if(choices, [&](PnrIndex core) {
      return problem->accCoreTargetClass(core) == targetClass;
    });
    require(selected != choices.end(),
            "custom bottleneck class is absent from a thread domain");
    threadChoices[decision] = static_cast<PnrIndex>(selected - choices.begin());
  }
  std::vector<PnrIndex> graphChoices(problem->graphDecisions().size(), 0);
  for (PnrIndex decision = 0; decision < problem->graphDecisions().size();
       ++decision) {
    const auto choices = problem->graphChoiceCatalogOrdinals(decision);
    const auto selected = llvm::find_if(choices, [&](PnrIndex mapping) {
      return problem->spatialMappingTargetClass(mapping) == targetClass;
    });
    require(selected != choices.end(),
            "custom bottleneck class is absent from a graph domain");
    graphChoices[decision] = static_cast<PnrIndex>(selected - choices.begin());
  }
  std::string lastDiagnostic;
  for (PnrIndex decision = 0; decision < problem->threadDecisions().size();
       ++decision) {
    const auto choices = problem->threadChoiceCatalogOrdinals(decision);
    const auto selected = llvm::find(choices, targetCore);
    if (selected == choices.end())
      continue;
    auto trial = threadChoices;
    trial[decision] = static_cast<PnrIndex>(selected - choices.begin());
    std::uint64_t endpointExpansions = 0;
    std::uint64_t negotiationIterations = 0;
    auto candidate =
        initializeSystemCandidate(problem, trial, graphChoices,
                                  &endpointExpansions, &negotiationIterations);
    if (!candidate) {
      lastDiagnostic = llvm::toString(candidate.takeError());
      continue;
    }
    if ((*candidate)->routeCapacityOveruse() != 0)
      return {std::move(*candidate), endpointExpansions, negotiationIterations};
    const auto usage = selectedUsage(**candidate);
    std::string observation;
    llvm::raw_string_ostream stream(observation);
    stream << "routes=" << (*candidate)->serviceRoutes().size()
           << ", cells=" << usage.size() << ", used=";
    for (PnrIndex cell = 0; cell < usage.size(); ++cell)
      if (usage[cell] != 0)
        stream << cell << ':' << usage[cell] << '/'
               << problem->routingTopology().capacityCells()[cell].capacity
               << ',';
    lastDiagnostic = stream.str();
  }
  fail("no custom AccCore assignment exposed the first-iteration bottleneck: " +
       lastDiagnostic);
}

} // namespace

void loom::pnr::test::verifySystemImportedCapacityWorkflow(
    const SystemCandidateState &candidate) {
  const FrozenSystemPnrProblem &problem = candidate.problem();
  require(problem.threadDecisions().size() == 2 &&
              problem.graphDecisions().size() == 4,
          "imported capacity fixture lost its repeated execution contexts");

  struct CorePair final {
    PnrIndex firstChoice = getInvalidPnrIndex();
    PnrIndex secondChoice = getInvalidPnrIndex();
  };
  std::optional<CorePair> sharedOccurrence;
  std::optional<CorePair> distinctOccurrences;
  const auto firstCores = problem.threadChoiceCatalogOrdinals(0);
  const auto secondCores = problem.threadChoiceCatalogOrdinals(1);
  for (const auto &[firstChoice, firstCore] : llvm::enumerate(firstCores)) {
    for (const auto &[secondChoice, secondCore] :
         llvm::enumerate(secondCores)) {
      if (problem.accCoreTargetClass(firstCore) !=
          problem.accCoreTargetClass(secondCore))
        continue;
      CorePair pair{static_cast<PnrIndex>(firstChoice),
                    static_cast<PnrIndex>(secondChoice)};
      if (firstCore == secondCore)
        sharedOccurrence = pair;
      else
        distinctOccurrences = pair;
    }
  }
  require(sharedOccurrence && distinctOccurrences,
          "imported capacity fixture has no shared and distinct core choices");

  const auto choicesFor = [&](CorePair pair) {
    std::vector<PnrIndex> threads(candidate.threadChoices().begin(),
                                  candidate.threadChoices().end());
    threads[0] = pair.firstChoice;
    threads[1] = pair.secondChoice;
    std::vector<PnrIndex> graphs(problem.graphDecisions().size(), 0);
    for (PnrIndex graph = 0; graph < problem.graphDecisions().size(); ++graph) {
      const auto overlaps = problem.graphThreadOverlaps(graph);
      require(!overlaps.empty(),
              "imported capacity graph has no execution context");
      const auto selectedCores =
          problem.threadChoiceCatalogOrdinals(overlaps.front());
      require(overlaps.front() < threads.size() &&
                  threads[overlaps.front()] < selectedCores.size(),
              "imported capacity thread choice is out of range");
      const PnrIndex targetClass =
          problem.accCoreTargetClass(selectedCores[threads[overlaps.front()]]);
      for (PnrIndex thread : overlaps) {
        const auto cores = problem.threadChoiceCatalogOrdinals(thread);
        require(thread < threads.size() && threads[thread] < cores.size() &&
                    problem.accCoreTargetClass(cores[threads[thread]]) ==
                        targetClass,
                "one graph context spans incompatible core classes");
      }
      const auto mappings = problem.graphChoiceCatalogOrdinals(graph);
      const auto selected = llvm::find_if(mappings, [&](PnrIndex mapping) {
        return problem.spatialMappingTargetClass(mapping) == targetClass;
      });
      require(selected != mappings.end(),
              "imported capacity graph has no compatible SpatialMapping");
      graphs[graph] = static_cast<PnrIndex>(selected - mappings.begin());
    }
    return std::pair{std::move(threads), std::move(graphs)};
  };

  const auto shared = choicesFor(*sharedOccurrence);
  const auto sharedProjection =
      projectImportedCapacity(candidate, shared.first, shared.second);
  require(sharedProjection.total == 0,
          "repeated graph contexts counted imported routes more than once");

  const auto distinct = choicesFor(*distinctOccurrences);
  const auto distinctProjection =
      projectImportedCapacity(candidate, distinct.first, distinct.second);
  require(distinctProjection.total == 0,
          "distinct SpatialCore occurrences shared one capacity namespace");
}

void loom::pnr::test::verifySystemNegotiatedRoutingWorkflow(
    ArtifactStore &store, const fabric::FinalizedFabricRoot &baselineSystem,
    const fabric::FinalizedFabricRoot &primaryModule,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ArtifactRootReference &spatialMapping, const ResolvedConfig &resolved,
    mlir::MLIRContext &context) {
  auto design = buildNegotiatedRoutingSystem(store, baselineSystem,
                                             primaryModule, context);
  auto system = take(fabric::requireSystemRoot(design.roots().front().view()));
  std::vector<dataflow::RootThreadLaunchRef> roots{
      dataflow.rootThreadLaunches().front().ref};
  auto constraints = take(mapping::finalizeEmptySystemMappingConstraintSet(
      dataflow, system, roots, store));
  auto partition = take(projectWholeDomainPresburgerPartitionPlan(
      dataflow, constraints.view().rootThreadLaunches()));

  auto limited = buildProblem(resolved, 1, dataflow, system, constraints,
                              partition, spatialMapping, store);
  auto first = initializeBottleneckCandidate(limited.problem);
  verifyServiceRouterScratchReuse(*first.state);
  require(first.negotiationIterations == 1,
          "limited initializer did not exhaust one negotiated iteration");
  require(first.state->capacityOveruse() != 0 &&
              first.state->capacityOveruse() ==
                  first.state->routeCapacityOveruse(),
          "first negotiated iterate did not expose one exact route bottleneck");
  const auto firstUsage = selectedUsage(*first.state);
  const auto firstConflicts =
      take(detail::analyzeSystemFixedTerminalCapacityConflicts(
          first.state->problem(),
          {first.state->serviceRoutes(), first.state->serviceRouteNodes(),
           first.state->serviceRouteSinks()},
          firstUsage));
  require(firstConflicts.size() == 1 &&
              !firstConflicts.front().hasCertificate(),
          "reroutable bottleneck was misclassified as a fixed-terminal cut");

  const auto instructionOveruse = repeatedOveruseSelections(
      system.artifact(), first.state->instructionResourceUses(),
      "InstructionCore ResourceUse");
  const auto serviceOveruse = repeatedOveruseSelections(
      system.artifact(), first.state->serviceResourceUses(),
      "service ResourceUse");
  const auto importedOnly = projectCapacity(*first.state, {}, {}, {}, {});
  const auto instructionOnly =
      projectCapacity(*first.state, {}, {}, instructionOveruse, {});
  const auto serviceOnly =
      projectCapacity(*first.state, {}, {}, {}, serviceOveruse);
  const auto routeOnly =
      projectCapacity(*first.state, first.state->serviceRoutes(),
                      first.state->serviceRouteNodes(), {}, {});
  const auto mixed = projectCapacity(*first.state, first.state->serviceRoutes(),
                                     first.state->serviceRouteNodes(),
                                     instructionOveruse, serviceOveruse);
  require(importedOnly.total == 0 && instructionOnly.total != 0 &&
              serviceOnly.total != 0,
          "System projection lost root ResourceUse capacity overuse");
  require(routeOnly.total == first.state->routeCapacityOveruse() &&
              mixed.total > routeOnly.total &&
              mixed.total >= instructionOnly.total &&
              mixed.total >= serviceOnly.total,
          "System projection did not compose route and ResourceUse totals");

  auto rejecting = buildProblem(resolved, 1, dataflow, system, constraints,
                                partition, spatialMapping, store,
                                /*admitTemporary=*/false);
  auto rejected = SystemCandidateState::create(
      rejecting.problem,
      {first.state->threadChoices(), first.state->graphChoices(),
       first.state->serviceRoutes(), first.state->serviceRouteNodes(),
       first.state->serviceRouteSinks(), first.state->serviceTargets(),
       first.state->instructionResourceUses(),
       first.state->serviceResourceUses()});
  require(!rejected, "non-admitting CandidateState accepted capacity overuse");
  bool typedInfeasible = false;
  llvm::Error rejectionRemaining = llvm::handleErrors(
      rejected.takeError(), [&](const detail::SystemCandidateInfeasible &) {
        typedInfeasible = true;
      });
  if (rejectionRemaining)
    fail(llvm::toString(std::move(rejectionRemaining)));
  require(typedInfeasible,
          "capacity rejection lost its typed infeasibility cause");

  const auto witnesses = first.state->routeCapacityOveruseWitnesses();
  require(witnesses.size() == 1 &&
              witnesses.front().overuse == first.state->capacityOveruse() &&
              witnesses.front().usage ==
                  witnesses.front().capacity + witnesses.front().overuse,
          "first negotiated iterate has no canonical exact capacity witness");
  const PnrIndex witnessCell = witnesses.front().capacityCell;
  const std::vector<PnrIndex> affected = witnessLegs(*first.state, witnessCell);
  if (affected.size() != 2) {
    std::string detail;
    llvm::raw_string_ostream stream(detail);
    const auto usage = selectedUsage(*first.state);
    for (PnrIndex cell = 0; cell < usage.size(); ++cell) {
      const auto legs = witnessLegs(*first.state, cell);
      if (!legs.empty())
        stream << cell << ':' << legs.size() << ':' << usage[cell] << '/'
               << first.state->problem()
                      .routingTopology()
                      .capacityCells()[cell]
                      .capacity
               << ':';
      for (PnrIndex leg : legs)
        stream << leg << '.';
      if (!legs.empty())
        stream << ',';
    }
    fail("shared bottleneck witness covers " + llvm::Twine(affected.size()) +
         " selected service legs instead of two; " + stream.str());
  }
  require(findCapacityFeasibleAlternatePath(*first.state, affected.front(),
                                            witnessCell) ||
              findCapacityFeasibleAlternatePath(*first.state, affected.back(),
                                                witnessCell),
          "fixture has no structural capacity-feasible multi-hop repair path");
  auto temporaryDraft =
      take(materializeSystemCandidateDraft(*first.state, context));
  auto temporaryVerification = mapping::verifySystemMappingBase(
      mlir::cast<::mapping::SystemOp>(temporaryDraft.get()), dataflow, system,
      store);
  const auto *temporaryRejection =
      std::get_if<mapping::RejectedSystemMappingBase>(&temporaryVerification);
  require(temporaryRejection && llvm::StringRef(temporaryRejection->diagnostic)
                                    .contains("CapacityOveruse"),
          "artifact verifier admitted the temporary overcapacity iterate");

  SystemActionDomainScratch limitedDomain;
  if (llvm::Error error = limitedDomain.rebuild(*first.state))
    fail(llvm::toString(std::move(error)));
  const auto witnessAction = llvm::find_if(
      limitedDomain.view().routingChoices,
      [&](const SystemTransportRoutingAction &action) {
        const auto *witness =
            std::get_if<SystemWitnessRegionRoutingAction>(&action);
        return witness &&
               witness->witnessKind ==
                   ResolvedPnrViolationKind::CapacityOveruse &&
               witness->witnessOrdinal == witnessCell;
      });
  require(witnessAction != limitedDomain.view().routingChoices.end(),
          "live capacity witness did not activate WitnessRegion");

  auto limitedObjective =
      take(limited.problem->objectiveProgram().evaluate(*first.state));
  SystemActionProbeAccounting strictFailureWork;
  auto strictFailure = probeSystemAction(
      first.state, limitedObjective,
      SystemMappingAction{
          SystemTransportRoutingAction{SystemGlobalRoutingAction{}}},
      strictFailureWork, SystemActionExecutionContext::FinalClosure);
  require(!strictFailure,
          "strict final closure retained a temporary overcapacity iterate");
  bool typedWorkLimit = false;
  llvm::Error remaining = llvm::handleErrors(
      strictFailure.takeError(),
      [&](const SystemActionTransitionFailure &failure) {
        typedWorkLimit =
            failure.kind() == SystemActionTransitionFailureKind::WorkLimit;
      });
  if (remaining)
    fail(llvm::toString(std::move(remaining)));
  require(typedWorkLimit && strictFailureWork.negotiationIterations == 1,
          "strict final nonclosure lost its typed work-limit outcome");

  auto sufficient = buildProblem(resolved, 8, dataflow, system, constraints,
                                 partition, spatialMapping, store);
  auto conflicted = take(SystemCandidateState::create(
      sufficient.problem,
      {first.state->threadChoices(), first.state->graphChoices(),
       first.state->serviceRoutes(), first.state->serviceRouteNodes(),
       first.state->serviceRouteSinks(), first.state->serviceTargets(),
       first.state->instructionResourceUses(),
       first.state->serviceResourceUses()}));
  SystemActionDomainScratch sufficientDomain;
  if (llvm::Error error = sufficientDomain.rebuild(*conflicted))
    fail(llvm::toString(std::move(error)));
  const auto focusedAction = llvm::find_if(
      sufficientDomain.view().routingChoices,
      [&](const SystemTransportRoutingAction &action) {
        const auto *witness =
            std::get_if<SystemWitnessRegionRoutingAction>(&action);
        return witness && witness->witnessOrdinal == witnessCell;
      });
  require(focusedAction != sufficientDomain.view().routingChoices.end(),
          "transplanted live witness lost its focused Action");
  auto conflictedObjective =
      take(sufficient.problem->objectiveProgram().evaluate(*conflicted));
  SystemActionProbeAccounting focusedWork;
  auto focused =
      take(probeSystemAction(conflicted, conflictedObjective,
                             SystemMappingAction{*focusedAction}, focusedWork));
  require(focusedWork.negotiationIterations != 0,
          "WitnessRegion consumed no negotiated routing work");
  for (const auto &route : conflicted->serviceRoutes())
    if (!llvm::is_contained(affected, route.leg))
      require(sameRoute(*conflicted, *focused.candidate, route.leg),
              "WitnessRegion changed a leg outside the witness closure");

  const auto close = [&](const SystemCandidateStateHandle &candidate) {
    auto objective =
        take(candidate->problem().objectiveProgram().evaluate(*candidate));
    SystemActionProbeAccounting work;
    auto result = take(
        probeSystemAction(candidate, objective,
                          SystemMappingAction{SystemTransportRoutingAction{
                              SystemGlobalRoutingAction{}}},
                          work, SystemActionExecutionContext::FinalClosure));
    require(result.candidate->capacityOveruse() == 0 &&
                result.candidate->routeCapacityOveruse() == 0 &&
                result.candidate->routeCapacityOveruseWitnesses().empty(),
            "sufficient strict global closure left capacity overuse");
    return result.candidate;
  };
  auto firstClosed = close(conflicted);
  const auto retainedNonRoute = projectCapacity(
      *firstClosed, firstClosed->serviceRoutes(),
      firstClosed->serviceRouteNodes(), instructionOveruse, serviceOveruse);
  require(firstClosed->routeCapacityOveruse() == 0 &&
              retainedNonRoute.total != 0,
          "final capacity fixture did not isolate non-route overuse");
  auto closedDraft =
      take(materializeSystemCandidateDraft(*firstClosed, context));
  require(std::holds_alternative<mapping::VerifiedSystemMappingBase>(
              mapping::verifySystemMappingBase(
                  mlir::cast<::mapping::SystemOp>(closedDraft.get()), dataflow,
                  system, store)),
          "artifact verifier rejected the strict capacity closure");
  auto replayConflict = take(SystemCandidateState::create(
      sufficient.problem,
      {first.state->threadChoices(), first.state->graphChoices(),
       first.state->serviceRoutes(), first.state->serviceRouteNodes(),
       first.state->serviceRouteSinks(), first.state->serviceTargets(),
       first.state->instructionResourceUses(),
       first.state->serviceResourceUses()}));
  auto secondClosed = close(replayConflict);
  for (const auto &route : firstClosed->serviceRoutes())
    require(sameRoute(*firstClosed, *secondClosed, route.leg),
            "sufficient negotiated closure changed on replay");
}
