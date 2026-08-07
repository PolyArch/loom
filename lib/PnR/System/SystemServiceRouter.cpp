#include "SystemServiceRouter.h"

#include "SystemCandidateServiceResolver.h"

#include "PnR/EndpointRouter.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <limits>
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

} // namespace

llvm::Expected<detail::CanonicalSystemServiceRoutes>
loom::pnr::detail::buildCanonicalSystemServiceRoutes(
    const FrozenSystemPnrProblem &problem,
    llvm::ArrayRef<PnrIndex> threadChoices,
    llvm::ArrayRef<PnrIndex> graphChoices) {
  CanonicalSystemServiceRoutes result;
  const FrozenEndpointRoutingTopology &topology = problem.routingTopology();
  EndpointRouteSearchScratch search;
  if (llvm::Error error = search.prepare(endpointRoutingGraphView(topology)))
    return std::move(error);
  std::vector<RouteCost> arcCosts(topology.arcs().size(), 1);

  for (PnrIndex legOrdinal = 0; legOrdinal < problem.serviceLegs().size();
       ++legOrdinal) {
    const FrozenSystemServiceLeg &leg = problem.serviceLegs()[legOrdinal];
    const auto sinkTerminals = problem.serviceLegSinkTerminals(legOrdinal);
    if (sinkTerminals.empty())
      return invalid("a frozen service leg has no sink terminal");
    auto sourceDomain = resolveSystemServiceTerminalDomain(
        problem, legOrdinal, leg.sourceTerminal, threadChoices, graphChoices);
    if (!sourceDomain)
      return sourceDomain.takeError();

    std::vector<MutableNode> nodes;
    std::vector<MutableSink> sinks;
    llvm::DenseMap<PnrIndex, PnrIndex> nodeByEndpoint;
    for (PnrIndex sinkTerminal : sinkTerminals) {
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
      auto targetEndpoints = resolveSystemServiceTerminalDomain(
          problem, legOrdinal, sinkTerminal, threadChoices, graphChoices);
      if (!targetEndpoints)
        return targetEndpoints.takeError();
      std::vector<PnrIndex> targetRanks(targetEndpoints->size());
      for (PnrIndex rank = 0; rank < targetRanks.size(); ++rank)
        targetRanks[rank] = rank;
      auto routed = search.search(
          {sourceEndpoints,
           sourceReplicationGroups,
           *targetEndpoints,
           targetRanks,
           arcCosts,
           arcCosts,
           leg.requiredPayloadWidthBits,
           0,
           problem.config().policy().search.routing.endpointExpansionLimit,
           {}});
      if (!routed)
        return llvm::handleErrors(
            routed.takeError(),
            [&](const EndpointRouteSearchFailure &failure) -> llvm::Error {
              std::string routeDiagnostic;
              llvm::raw_string_ostream stream(routeDiagnostic);
              failure.log(stream);
              stream.flush();
              if (failure.kind() == EndpointRouteSearchFailureKind::Unreachable)
                return llvm::make_error<detail::SystemCandidateInfeasible>(
                    ("cannot route frozen service leg " +
                     llvm::Twine(legOrdinal) + " in context " +
                     llvm::Twine(leg.serviceContext) + ": " + routeDiagnostic)
                        .str());
              return llvm::make_error<EndpointRouteSearchFailure>(
                  failure.kind(), std::move(routeDiagnostic));
            });

      PnrIndex sourceNode = 0;
      if (nodes.empty()) {
        nodes.push_back({routed->source, getInvalidPnrIndex(),
                         getInvalidPnrIndex(), false, getInvalidPnrIndex()});
        nodeByEndpoint.try_emplace(routed->source, 0);
      } else {
        auto found = findNodeForEndpoint(nodeByEndpoint, routed->source);
        if (!found)
          return found.takeError();
        sourceNode = *found;
      }
      auto targetNode = appendPath(topology, routed->forwardArcs, sourceNode,
                                   nodes, nodeByEndpoint);
      if (!targetNode)
        return targetNode.takeError();
      if (nodes[*targetNode].endpoint != routed->target)
        return invalid("A-star result target disagrees with the route tree");
      sinks.push_back({sinkTerminal, *targetNode});
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
    const auto expectedSinks = problem.serviceLegSinkTerminals(route.leg);
    if (route.sinkCount != expectedSinks.size())
      return invalid("service route does not cover every sink terminal");
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

    std::vector<std::uint8_t> sinkSeen(expectedSinks.size(), 0);
    for (const auto &sink : routeSinks) {
      auto found = std::lower_bound(expectedSinks.begin(), expectedSinks.end(),
                                    sink.terminal);
      if (found == expectedSinks.end() || *found != sink.terminal ||
          sink.node >= routeNodes.size())
        return invalid("service route sink is outside its exact H domain");
      const std::size_t sinkOrdinal = found - expectedSinks.begin();
      if (sinkSeen[sinkOrdinal]++)
        return invalid("service route binds one sink more than once");
      auto terminalDomain = resolveSystemServiceTerminalDomain(
          problem, route.leg, sink.terminal, threadChoices, graphChoices);
      if (!terminalDomain)
        return terminalDomain.takeError();
      if (!contains(*terminalDomain, routeNodes[sink.node].endpoint))
        return invalid("service route sink endpoint is not admitted by H");
    }
    if (llvm::any_of(sinkSeen, [](std::uint8_t seen) { return seen != 1; }))
      return invalid("service route omitted a sink terminal");
    expectedNodeOffset += route.nodeCount;
    expectedSinkOffset += route.sinkCount;
  }
  if (expectedNodeOffset != nodes.size() || expectedSinkOffset != sinks.size())
    return invalid("service route flat arrays contain trailing records");
  return llvm::Error::success();
}
