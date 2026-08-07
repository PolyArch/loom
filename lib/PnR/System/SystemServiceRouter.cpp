#include "SystemServiceRouter.h"

#include "SystemCandidateServiceResolver.h"

#include "PnR/EndpointRouter.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::pnr;

namespace {

constexpr llvm::StringLiteral nativeOwner = "SystemCandidateState";
constexpr PnrCapacityContext nodeOffsetContext{
    nativeOwner, "service_routes", "node", PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext sinkOffsetContext{
    nativeOwner, "service_routes", "sink", PnrCapacityMeasure::Offset};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_service_route_invalid: " + message);
}

llvm::Expected<PnrIndex> checked(PnrCapacityContext context,
                                 std::size_t value) {
  return checkedPnrIndex(context, static_cast<std::uint64_t>(value));
}

bool contains(llvm::ArrayRef<PnrIndex> values, PnrIndex value) {
  return std::binary_search(values.begin(), values.end(), value);
}

struct MutableNode final {
  PnrIndex endpoint = 0;
  PnrIndex parent = getInvalidPnrIndex();
  PnrIndex incomingTraversal = getInvalidPnrIndex();
  bool hasOutgoing = false;
  PnrIndex outgoingReplicationGroup = getInvalidPnrIndex();
};

struct MutableSink final {
  PnrIndex terminal = 0;
  PnrIndex node = 0;
};

struct AtomicPatternCatalog final {
  std::vector<std::vector<PnrIndex>> traversalsByGroup;
  std::vector<std::uint64_t> unicastEligibility;
};

void admitTraversal(std::vector<std::uint64_t> &eligibility,
                    PnrIndex traversal) {
  eligibility[traversal / 64] |= std::uint64_t{1} << (traversal % 64);
}

llvm::Expected<AtomicPatternCatalog>
buildAtomicPatternCatalog(const FrozenEndpointRoutingTopology &topology) {
  AtomicPatternCatalog catalog;
  catalog.traversalsByGroup.resize(topology.traversals().size());
  catalog.unicastEligibility.resize((topology.traversals().size() + 63) / 64);
  for (PnrIndex traversal = 0; traversal < topology.traversals().size();
       ++traversal) {
    if (!std::holds_alternative<
            ::loom::fabric::FabricTransferPatternLegPayload>(
            topology.traversals()[traversal].reference.payload))
      continue;
    const PnrIndex group = topology.traversalReplicationGroups()[traversal];
    if (group == getInvalidPnrIndex() ||
        group >= catalog.traversalsByGroup.size())
      return invalid("an atomic transfer-pattern leg has no replication group");
    catalog.traversalsByGroup[group].push_back(traversal);
  }
  for (PnrIndex traversal = 0; traversal < topology.traversals().size();
       ++traversal) {
    const PnrIndex group = topology.traversalReplicationGroups()[traversal];
    if (group == getInvalidPnrIndex() ||
        catalog.traversalsByGroup[group].empty() ||
        catalog.traversalsByGroup[group].size() == 1)
      admitTraversal(catalog.unicastEligibility, traversal);
  }
  return catalog;
}

llvm::Expected<std::pair<PnrIndex, PnrIndex>>
traversalEndpoints(const FrozenEndpointRoutingTopology &topology,
                   PnrIndex traversal) {
  if (traversal >= topology.traversals().size())
    return invalid("an atomic pattern names an invalid traversal");
  const auto &view = topology.traversals()[traversal];
  if (view.sourceCount != 1 || view.destinationCount != 1)
    return invalid("an atomic transfer-pattern leg is not point-to-point");
  return std::pair<PnrIndex, PnrIndex>{
      topology.traversalEndpoints()[view.sourceOffset],
      topology.traversalEndpoints()[view.destinationOffset]};
}

std::vector<std::uint64_t>
routeEligibility(const FrozenEndpointRoutingTopology &topology,
                 const AtomicPatternCatalog &catalog,
                 llvm::ArrayRef<MutableNode> nodes) {
  std::vector<std::uint64_t> eligibility = catalog.unicastEligibility;
  for (const MutableNode &node : nodes) {
    if (node.incomingTraversal == getInvalidPnrIndex())
      continue;
    const PnrIndex group =
        topology.traversalReplicationGroups()[node.incomingTraversal];
    if (group == getInvalidPnrIndex() ||
        catalog.traversalsByGroup[group].size() <= 1)
      continue;
    for (PnrIndex traversal : catalog.traversalsByGroup[group])
      admitTraversal(eligibility, traversal);
  }
  return eligibility;
}

struct AtomicPatternUpgrade final {
  PnrIndex node = 0;
  PnrIndex group = 0;
  PnrIndex extraTraversal = 0;
  std::vector<std::pair<PnrIndex, PnrIndex>> childReplacements;
};

llvm::Expected<std::vector<AtomicPatternUpgrade>> findAtomicPatternUpgrades(
    const FrozenEndpointRoutingTopology &topology,
    const AtomicPatternCatalog &catalog, llvm::ArrayRef<MutableNode> nodes,
    const llvm::DenseMap<PnrIndex, PnrIndex> &nodeByEndpoint) {
  std::vector<AtomicPatternUpgrade> upgrades;
  for (PnrIndex nodeOrdinal = 0; nodeOrdinal < nodes.size(); ++nodeOrdinal) {
    std::vector<PnrIndex> children;
    for (PnrIndex child = 1; child < nodes.size(); ++child)
      if (nodes[child].parent == nodeOrdinal)
        children.push_back(child);
    if (children.empty())
      continue;
    for (PnrIndex group = 0; group < catalog.traversalsByGroup.size();
         ++group) {
      const auto &pattern = catalog.traversalsByGroup[group];
      if (pattern.size() != children.size() + 1 ||
          group == nodes[nodeOrdinal].outgoingReplicationGroup)
        continue;
      AtomicPatternUpgrade upgrade{
          nodeOrdinal, group, getInvalidPnrIndex(), {}};
      std::vector<std::uint8_t> matched(pattern.size(), 0);
      bool compatible = true;
      for (PnrIndex child : children) {
        bool found = false;
        for (std::size_t position = 0; position < pattern.size(); ++position) {
          auto endpoints = traversalEndpoints(topology, pattern[position]);
          if (!endpoints)
            return endpoints.takeError();
          if (endpoints->first == nodes[nodeOrdinal].endpoint &&
              endpoints->second == nodes[child].endpoint &&
              !matched[position]) {
            matched[position] = 1;
            upgrade.childReplacements.emplace_back(child, pattern[position]);
            found = true;
            break;
          }
        }
        if (!found) {
          compatible = false;
          break;
        }
      }
      if (!compatible)
        continue;
      for (std::size_t position = 0; position < pattern.size(); ++position) {
        if (matched[position])
          continue;
        auto endpoints = traversalEndpoints(topology, pattern[position]);
        if (!endpoints)
          return endpoints.takeError();
        if (endpoints->first != nodes[nodeOrdinal].endpoint ||
            nodeByEndpoint.contains(endpoints->second)) {
          compatible = false;
          break;
        }
        upgrade.extraTraversal = pattern[position];
      }
      if (compatible && upgrade.extraTraversal != getInvalidPnrIndex())
        upgrades.push_back(std::move(upgrade));
    }
  }
  llvm::sort(upgrades, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.node, lhs.group, lhs.extraTraversal) <
           std::tie(rhs.node, rhs.group, rhs.extraTraversal);
  });
  return upgrades;
}

struct RouteProbe final {
  PnrIndex source = 0;
  PnrIndex target = 0;
  RouteCost cost = 0;
  std::vector<PnrIndex> forwardArcs;
  std::optional<AtomicPatternUpgrade> upgrade;
};

bool isBetterProbe(const RouteProbe &candidate, const RouteProbe &current) {
  if (candidate.cost != current.cost)
    return candidate.cost < current.cost;
  if (candidate.upgrade.has_value() != current.upgrade.has_value())
    return !candidate.upgrade.has_value();
  if (std::tie(candidate.source, candidate.target) !=
      std::tie(current.source, current.target))
    return std::tie(candidate.source, candidate.target) <
           std::tie(current.source, current.target);
  if (candidate.forwardArcs != current.forwardArcs)
    return candidate.forwardArcs < current.forwardArcs;
  if (!candidate.upgrade)
    return false;
  return std::tie(candidate.upgrade->node, candidate.upgrade->group,
                  candidate.upgrade->extraTraversal) <
         std::tie(current.upgrade->node, current.upgrade->group,
                  current.upgrade->extraTraversal);
}

llvm::Error noteOutgoing(const FrozenEndpointRoutingTopology &topology,
                         std::vector<MutableNode> &nodes, PnrIndex parent,
                         PnrIndex traversal) {
  if (parent >= nodes.size() ||
      traversal >= topology.traversalReplicationGroups().size())
    return invalid("route edge is outside the frozen topology");
  MutableNode &node = nodes[parent];
  const PnrIndex group = topology.traversalReplicationGroups()[traversal];
  if (!node.hasOutgoing) {
    node.hasOutgoing = true;
    node.outgoingReplicationGroup = group;
    return llvm::Error::success();
  }
  if (group == getInvalidPnrIndex() ||
      node.outgoingReplicationGroup == getInvalidPnrIndex() ||
      group != node.outgoingReplicationGroup)
    return invalid("route branches without one Fabric replication group");
  return llvm::Error::success();
}

llvm::Expected<PnrIndex>
appendPath(const FrozenEndpointRoutingTopology &topology,
           llvm::ArrayRef<PnrIndex> forwardArcs, PnrIndex sourceNode,
           std::vector<MutableNode> &nodes,
           llvm::DenseMap<PnrIndex, PnrIndex> &nodeByEndpoint) {
  PnrIndex lastExistingPathPosition = getInvalidPnrIndex();
  PnrIndex existingNode = sourceNode;
  for (auto [position, arcOrdinal] : llvm::enumerate(forwardArcs)) {
    if (arcOrdinal >= topology.arcs().size())
      return invalid("A-star returned an out-of-range arc");
    auto found = nodeByEndpoint.find(topology.arcs()[arcOrdinal].target);
    if (found != nodeByEndpoint.end()) {
      lastExistingPathPosition = static_cast<PnrIndex>(position);
      existingNode = found->second;
    }
  }
  std::size_t begin = 0;
  PnrIndex current = sourceNode;
  if (lastExistingPathPosition != getInvalidPnrIndex()) {
    begin = static_cast<std::size_t>(lastExistingPathPosition) + 1;
    current = existingNode;
  }
  for (std::size_t position = begin; position < forwardArcs.size();
       ++position) {
    const PnrIndex arcOrdinal = forwardArcs[position];
    const EndpointRoutingArc &arc = topology.arcs()[arcOrdinal];
    if (topology.arcSources()[arcOrdinal] != nodes[current].endpoint)
      return invalid("A-star path is disconnected from the selected tree");
    if (nodeByEndpoint.contains(arc.target))
      return invalid("A-star path repeats an existing route endpoint");
    if (llvm::Error error =
            noteOutgoing(topology, nodes, current, arc.traversal))
      return std::move(error);
    auto node = checked(nodeOffsetContext, nodes.size());
    if (!node)
      return node.takeError();
    nodes.push_back(
        {arc.target, current, arc.traversal, false, getInvalidPnrIndex()});
    nodeByEndpoint.try_emplace(arc.target, *node);
    current = *node;
  }
  return current;
}

llvm::Expected<PnrIndex>
findNodeForEndpoint(const llvm::DenseMap<PnrIndex, PnrIndex> &nodeByEndpoint,
                    PnrIndex endpoint) {
  auto found = nodeByEndpoint.find(endpoint);
  if (found == nodeByEndpoint.end())
    return invalid("selected target endpoint is absent from the route tree");
  return found->second;
}

llvm::Error canonicalizeTree(std::vector<MutableNode> &nodes,
                             std::vector<MutableSink> &sinks) {
  if (nodes.empty())
    return invalid("cannot canonicalize an empty service route");
  std::vector<std::vector<PnrIndex>> children(nodes.size());
  for (PnrIndex node = 1; node < nodes.size(); ++node) {
    if (nodes[node].parent >= node)
      return invalid("service route is not a rooted acyclic tree");
    children[nodes[node].parent].push_back(node);
  }
  for (auto &siblings : children)
    llvm::sort(siblings, [&](PnrIndex lhs, PnrIndex rhs) {
      return std::tie(nodes[lhs].incomingTraversal, nodes[lhs].endpoint) <
             std::tie(nodes[rhs].incomingTraversal, nodes[rhs].endpoint);
    });

  std::vector<PnrIndex> preorder;
  preorder.reserve(nodes.size());
  std::vector<PnrIndex> stack{0};
  while (!stack.empty()) {
    const PnrIndex node = stack.back();
    stack.pop_back();
    preorder.push_back(node);
    for (auto child = children[node].rbegin(); child != children[node].rend();
         ++child)
      stack.push_back(*child);
  }
  if (preorder.size() != nodes.size())
    return invalid("service route contains an unreachable node");
  std::vector<PnrIndex> remap(nodes.size(), getInvalidPnrIndex());
  for (auto [newOrdinal, oldOrdinal] : llvm::enumerate(preorder))
    remap[oldOrdinal] = static_cast<PnrIndex>(newOrdinal);
  std::vector<MutableNode> ordered;
  ordered.reserve(nodes.size());
  for (PnrIndex oldOrdinal : preorder) {
    MutableNode node = nodes[oldOrdinal];
    if (node.parent != getInvalidPnrIndex())
      node.parent = remap[node.parent];
    ordered.push_back(node);
  }
  for (MutableSink &sink : sinks)
    sink.node = remap[sink.node];
  llvm::sort(sinks, [](const MutableSink &lhs, const MutableSink &rhs) {
    return std::tie(lhs.terminal, lhs.node) < std::tie(rhs.terminal, rhs.node);
  });
  nodes = std::move(ordered);
  return llvm::Error::success();
}

struct ActiveServiceSink final {
  PnrIndex terminal = 0;
  std::vector<std::uint8_t> terminalKey;
  std::vector<PnrIndex> endpoints;
};

llvm::Expected<std::vector<ActiveServiceSink>>
activeServiceSinks(const FrozenSystemPnrProblem &problem, PnrIndex legOrdinal,
                   llvm::ArrayRef<PnrIndex> threadChoices,
                   llvm::ArrayRef<PnrIndex> graphChoices) {
  std::vector<ActiveServiceSink> active;
  for (PnrIndex terminal : problem.serviceLegSinkTerminals(legOrdinal)) {
    auto endpoints = detail::resolveSystemServiceTerminalDomain(
        problem, legOrdinal, terminal, threadChoices, graphChoices);
    if (!endpoints)
      return endpoints.takeError();
    auto key = ::loom::mapping::encodeSystemTransferTerminalKey(
        problem.dataflowIdentity(), problem.serviceTerminals()[terminal].key);
    if (!key)
      return key.takeError();
    const auto duplicate = llvm::find_if(active, [&](const auto &candidate) {
      return candidate.terminalKey == *key && candidate.endpoints == *endpoints;
    });
    if (duplicate != active.end())
      continue;
    active.push_back({terminal, std::move(*key), std::move(*endpoints)});
  }
  llvm::sort(active, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.terminalKey, lhs.endpoints) <
           std::tie(rhs.terminalKey, rhs.endpoints);
  });
  return active;
}

} // namespace

llvm::Expected<detail::CanonicalSystemServiceRoutes>
loom::pnr::detail::buildCanonicalSystemServiceRoutes(
    const FrozenSystemPnrProblem &problem,
    llvm::ArrayRef<PnrIndex> threadChoices,
    llvm::ArrayRef<PnrIndex> graphChoices) {
  CanonicalSystemServiceRoutes result;
  const FrozenEndpointRoutingTopology &topology = problem.routingTopology();
  auto atomicPatterns = buildAtomicPatternCatalog(topology);
  if (!atomicPatterns)
    return atomicPatterns.takeError();
  EndpointRouteSearchScratch search;
  if (llvm::Error error = search.prepare(endpointRoutingGraphView(topology)))
    return std::move(error);
  std::vector<RouteCost> arcCosts(topology.arcs().size(), 1);

  for (PnrIndex legOrdinal = 0; legOrdinal < problem.serviceLegs().size();
       ++legOrdinal) {
    const FrozenSystemServiceLeg &leg = problem.serviceLegs()[legOrdinal];
    auto activeSinks =
        activeServiceSinks(problem, legOrdinal, threadChoices, graphChoices);
    if (!activeSinks)
      return activeSinks.takeError();
    if (activeSinks->empty())
      return invalid("a frozen service leg has no sink terminal");
    auto sourceDomain = resolveSystemServiceTerminalDomain(
        problem, legOrdinal, leg.sourceTerminal, threadChoices, graphChoices);
    if (!sourceDomain)
      return sourceDomain.takeError();

    std::vector<MutableNode> nodes;
    std::vector<MutableSink> sinks;
    llvm::DenseMap<PnrIndex, PnrIndex> nodeByEndpoint;
    for (const ActiveServiceSink &activeSink : *activeSinks) {
      std::vector<PnrIndex> sourceEndpoints;
      std::vector<PnrIndex> sourceReplicationGroups;
      if (nodes.empty()) {
        sourceEndpoints.assign(sourceDomain->begin(), sourceDomain->end());
        sourceReplicationGroups.assign(sourceEndpoints.size(),
                                       getInvalidPnrIndex());
      } else {
        std::vector<std::pair<PnrIndex, PnrIndex>> frontier;
        frontier.reserve(nodes.size());
        for (const MutableNode &node : nodes)
          if (!node.hasOutgoing ||
              node.outgoingReplicationGroup != getInvalidPnrIndex())
            frontier.emplace_back(
                node.endpoint, node.hasOutgoing ? node.outgoingReplicationGroup
                                                : getInvalidPnrIndex());
        llvm::sort(frontier);
        for (const auto &[endpoint, group] : frontier) {
          sourceEndpoints.push_back(endpoint);
          sourceReplicationGroups.push_back(group);
        }
      }
      std::vector<PnrIndex> targetRanks(activeSink.endpoints.size());
      for (PnrIndex rank = 0; rank < targetRanks.size(); ++rank)
        targetRanks[rank] = rank;
      auto eligibility = routeEligibility(topology, *atomicPatterns, nodes);
      std::optional<RouteProbe> bestProbe;
      std::string lastRouteDiagnostic;
      const auto tryProbe =
          [&](llvm::ArrayRef<PnrIndex> sources,
              llvm::ArrayRef<PnrIndex> sourceGroups,
              llvm::ArrayRef<std::uint64_t> eligibleTraversals,
              std::optional<AtomicPatternUpgrade> upgrade) -> llvm::Error {
        auto routed = search.search(
            {sources, sourceGroups, activeSink.endpoints, targetRanks, arcCosts,
             arcCosts, leg.requiredPayloadWidthBits, 0,
             problem.config().policy().search.routing.endpointExpansionLimit,
             eligibleTraversals});
        if (!routed)
          return llvm::handleErrors(
              routed.takeError(),
              [&](const EndpointRouteSearchFailure &failure) -> llvm::Error {
                lastRouteDiagnostic.clear();
                llvm::raw_string_ostream stream(lastRouteDiagnostic);
                failure.log(stream);
                stream.flush();
                if (failure.kind() ==
                    EndpointRouteSearchFailureKind::Unreachable)
                  return llvm::Error::success();
                return llvm::make_error<EndpointRouteSearchFailure>(
                    failure.kind(), lastRouteDiagnostic);
              });
        RouteProbe probe{
            routed->source,
            routed->target,
            routed->cost,
            {routed->forwardArcs.begin(), routed->forwardArcs.end()},
            std::move(upgrade)};
        if (!bestProbe || isBetterProbe(probe, *bestProbe))
          bestProbe = std::move(probe);
        return llvm::Error::success();
      };
      if (llvm::Error error = tryProbe(sourceEndpoints, sourceReplicationGroups,
                                       eligibility, std::nullopt))
        return std::move(error);
      if (!nodes.empty()) {
        auto upgrades = findAtomicPatternUpgrades(topology, *atomicPatterns,
                                                  nodes, nodeByEndpoint);
        if (!upgrades)
          return upgrades.takeError();
        for (const AtomicPatternUpgrade &upgrade : *upgrades) {
          auto upgradedEligibility = eligibility;
          admitTraversal(upgradedEligibility, upgrade.extraTraversal);
          const std::array<PnrIndex, 1> source = {nodes[upgrade.node].endpoint};
          const std::array<PnrIndex, 1> group = {upgrade.group};
          if (llvm::Error error =
                  tryProbe(source, group, upgradedEligibility, upgrade))
            return std::move(error);
        }
      }
      if (!bestProbe)
        return llvm::make_error<detail::SystemCandidateInfeasible>(
            ("cannot route frozen service leg " + llvm::Twine(legOrdinal) +
             " in context " + llvm::Twine(leg.serviceContext) + ": " +
             lastRouteDiagnostic)
                .str());

      PnrIndex sourceNode = 0;
      if (nodes.empty()) {
        nodes.push_back({bestProbe->source, getInvalidPnrIndex(),
                         getInvalidPnrIndex(), false, getInvalidPnrIndex()});
        nodeByEndpoint.try_emplace(bestProbe->source, 0);
      } else if (bestProbe->upgrade) {
        sourceNode = bestProbe->upgrade->node;
        MutableNode &source = nodes[sourceNode];
        source.hasOutgoing = true;
        source.outgoingReplicationGroup = bestProbe->upgrade->group;
        for (const auto &[child, traversal] :
             bestProbe->upgrade->childReplacements)
          nodes[child].incomingTraversal = traversal;
      } else {
        auto found = findNodeForEndpoint(nodeByEndpoint, bestProbe->source);
        if (!found)
          return found.takeError();
        sourceNode = *found;
      }
      auto targetNode = appendPath(topology, bestProbe->forwardArcs, sourceNode,
                                   nodes, nodeByEndpoint);
      if (!targetNode)
        return targetNode.takeError();
      if (nodes[*targetNode].endpoint != bestProbe->target)
        return invalid("A-star result target disagrees with the route tree");
      sinks.push_back({activeSink.terminal, *targetNode});
    }
    if (llvm::Error error = canonicalizeTree(nodes, sinks))
      return std::move(error);

    auto nodeOffset = checked(nodeOffsetContext, result.nodes.size());
    auto nodeCount = checked(nodeOffsetContext, nodes.size());
    auto sinkOffset = checked(sinkOffsetContext, result.sinks.size());
    auto sinkCount = checked(sinkOffsetContext, sinks.size());
    if (!nodeOffset)
      return nodeOffset.takeError();
    if (!nodeCount)
      return nodeCount.takeError();
    if (!sinkOffset)
      return sinkOffset.takeError();
    if (!sinkCount)
      return sinkCount.takeError();
    for (const MutableNode &node : nodes)
      result.nodes.push_back(
          {node.endpoint, node.parent, node.incomingTraversal});
    for (const MutableSink &sink : sinks)
      result.sinks.push_back({sink.terminal, sink.node});
    result.routes.push_back({legOrdinal, nodes.front().endpoint, *nodeOffset,
                             *nodeCount, *sinkOffset, *sinkCount});
  }
  if (llvm::Error error =
          verifySystemServiceRoutes(problem, threadChoices, graphChoices,
                                    result.routes, result.nodes, result.sinks))
    return std::move(error);
  return result;
}

llvm::Error loom::pnr::detail::verifySystemServiceRoutes(
    const FrozenSystemPnrProblem &problem,
    llvm::ArrayRef<PnrIndex> threadChoices,
    llvm::ArrayRef<PnrIndex> graphChoices,
    llvm::ArrayRef<SystemServiceRouteSelection> routes,
    llvm::ArrayRef<SystemServiceRouteNodeSelection> nodes,
    llvm::ArrayRef<SystemServiceRouteSinkSelection> sinks) {
  if (routes.size() != problem.serviceLegs().size())
    return invalid("service route count does not match the frozen legs");
  const FrozenEndpointRoutingTopology &topology = problem.routingTopology();
  auto atomicPatterns = buildAtomicPatternCatalog(topology);
  if (!atomicPatterns)
    return atomicPatterns.takeError();
  PnrIndex expectedNodeOffset = 0;
  PnrIndex expectedSinkOffset = 0;
  for (PnrIndex routeOrdinal = 0; routeOrdinal < routes.size();
       ++routeOrdinal) {
    const SystemServiceRouteSelection &route = routes[routeOrdinal];
    if (route.leg != routeOrdinal || route.nodeOffset != expectedNodeOffset ||
        route.sinkOffset != expectedSinkOffset || route.nodeCount == 0)
      return invalid("service routes are not in canonical flat order");
    if (route.nodeOffset > nodes.size() ||
        route.nodeCount > nodes.size() - route.nodeOffset ||
        route.sinkOffset > sinks.size() ||
        route.sinkCount > sinks.size() - route.sinkOffset)
      return invalid("service route flat range is out of bounds");
    const auto routeNodes = nodes.slice(route.nodeOffset, route.nodeCount);
    const auto routeSinks = sinks.slice(route.sinkOffset, route.sinkCount);
    const FrozenSystemServiceLeg &leg = problem.serviceLegs()[route.leg];
    auto activeSinks =
        activeServiceSinks(problem, route.leg, threadChoices, graphChoices);
    if (!activeSinks)
      return activeSinks.takeError();
    if (route.sinkCount != activeSinks->size())
      return invalid(
          "service route does not cover the applicable sink-owner set");
    auto sourceDomain = resolveSystemServiceTerminalDomain(
        problem, route.leg, leg.sourceTerminal, threadChoices, graphChoices);
    if (!sourceDomain)
      return sourceDomain.takeError();
    if (routeNodes.front().endpoint != route.rootEndpoint ||
        routeNodes.front().parentNode != getInvalidPnrIndex() ||
        routeNodes.front().incomingTraversal != getInvalidPnrIndex() ||
        !contains(*sourceDomain, route.rootEndpoint))
      return invalid("service route root is not an admitted source endpoint");

    llvm::DenseMap<PnrIndex, PnrIndex> nodeByEndpoint;
    std::vector<PnrIndex> outgoingGroup(routeNodes.size(),
                                        getInvalidPnrIndex());
    std::vector<std::uint8_t> hasOutgoing(routeNodes.size(), 0);
    for (PnrIndex nodeOrdinal = 0; nodeOrdinal < routeNodes.size();
         ++nodeOrdinal) {
      const auto &node = routeNodes[nodeOrdinal];
      if (node.endpoint >= topology.endpoints().size() ||
          !nodeByEndpoint.try_emplace(node.endpoint, nodeOrdinal).second)
        return invalid("service route has an invalid or duplicate endpoint");
      if (nodeOrdinal == 0)
        continue;
      if (node.parentNode >= nodeOrdinal ||
          node.incomingTraversal >= topology.traversals().size())
        return invalid("service route node has an invalid parent edge");
      const auto &parent = routeNodes[node.parentNode];
      bool foundArc = false;
      const PnrIndex begin = topology.adjacencyOffsets()[parent.endpoint];
      const PnrIndex end = topology.adjacencyOffsets()[parent.endpoint + 1];
      for (PnrIndex arc = begin; arc < end; ++arc)
        if (topology.arcs()[arc].target == node.endpoint &&
            topology.arcs()[arc].traversal == node.incomingTraversal) {
          if (topology.arcs()[arc].payloadCapacityBits <
              leg.requiredPayloadWidthBits)
            return invalid("service route traversal is too narrow");
          foundArc = true;
          break;
        }
      if (!foundArc)
        return invalid("service route node is not a Fabric traversal");
      const PnrIndex group =
          topology.traversalReplicationGroups()[node.incomingTraversal];
      if (hasOutgoing[node.parentNode] &&
          (group == getInvalidPnrIndex() ||
           outgoingGroup[node.parentNode] == getInvalidPnrIndex() ||
           group != outgoingGroup[node.parentNode]))
        return invalid("service route branch lacks one replication group");
      hasOutgoing[node.parentNode] = 1;
      outgoingGroup[node.parentNode] = group;
    }

    for (PnrIndex parent = 0; parent < routeNodes.size(); ++parent) {
      const PnrIndex group = outgoingGroup[parent];
      if (group == getInvalidPnrIndex() ||
          atomicPatterns->traversalsByGroup[group].empty())
        continue;
      std::vector<std::uint8_t> selected(
          atomicPatterns->traversalsByGroup[group].size(), 0);
      for (PnrIndex child = 1; child < routeNodes.size(); ++child) {
        if (routeNodes[child].parentNode != parent)
          continue;
        const auto found = llvm::find(atomicPatterns->traversalsByGroup[group],
                                      routeNodes[child].incomingTraversal);
        if (found == atomicPatterns->traversalsByGroup[group].end())
          return invalid("service route mixes atomic transfer patterns");
        selected[found - atomicPatterns->traversalsByGroup[group].begin()] = 1;
      }
      if (llvm::any_of(selected, [](std::uint8_t value) { return value == 0; }))
        return invalid("service route splits an atomic transfer pattern");
    }

    std::vector<std::uint8_t> sinkSeen(activeSinks->size(), 0);
    std::vector<std::uint8_t> nodeHasSink(routeNodes.size(), 0);
    for (const auto &sink : routeSinks) {
      auto found = llvm::find_if(*activeSinks, [&](const auto &candidate) {
        return candidate.terminal == sink.terminal;
      });
      if (found == activeSinks->end() || sink.node >= routeNodes.size())
        return invalid("service route sink is outside its exact H domain");
      const std::size_t sinkOrdinal = found - activeSinks->begin();
      if (sinkSeen[sinkOrdinal]++)
        return invalid("service route binds one sink more than once");
      auto terminalDomain = resolveSystemServiceTerminalDomain(
          problem, route.leg, sink.terminal, threadChoices, graphChoices);
      if (!terminalDomain)
        return terminalDomain.takeError();
      if (!contains(*terminalDomain, routeNodes[sink.node].endpoint))
        return invalid("service route sink endpoint is not admitted by H");
      nodeHasSink[sink.node] = 1;
    }
    if (llvm::any_of(sinkSeen, [](std::uint8_t seen) { return seen != 1; }))
      return invalid("service route omitted a sink terminal");
    for (PnrIndex node = 0; node < routeNodes.size(); ++node)
      if (!hasOutgoing[node] && !nodeHasSink[node])
        return invalid("service route contains a non-sink leaf");
    expectedNodeOffset += route.nodeCount;
    expectedSinkOffset += route.sinkCount;
  }
  if (expectedNodeOffset != nodes.size() || expectedSinkOffset != sinks.size())
    return invalid("service route flat arrays contain trailing records");
  return llvm::Error::success();
}
