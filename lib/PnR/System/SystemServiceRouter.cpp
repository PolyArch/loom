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
#include <map>
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

llvm::Error arithmeticOverflow(const llvm::Twine &message) {
  return llvm::make_error<RoutingNegotiationError>(
      RoutingNegotiationError::Kind::ArithmeticOverflow,
      ("routing negotiation arithmetic overflow: " + message).str());
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

void rejectTraversal(std::vector<std::uint64_t> &eligibility,
                     PnrIndex traversal) {
  eligibility[traversal / 64] &= ~(std::uint64_t{1} << (traversal % 64));
}

llvm::Error
applyCapacityEligibility(const FrozenEndpointRoutingTopology &topology,
                         llvm::ArrayRef<std::uint64_t> usage,
                         std::vector<std::uint64_t> &eligibility) {
  if (usage.size() != topology.capacityCells().size())
    return invalid("route capacity usage has the wrong shape");
  for (const auto &[traversalOrdinal, traversal] :
       llvm::enumerate(topology.traversals())) {
    if (traversal.capacityClaimOffset > topology.capacityClaims().size() ||
        traversal.capacityClaimCount >
            topology.capacityClaims().size() - traversal.capacityClaimOffset)
      return invalid("a traversal capacity range is out of bounds");
    for (const auto &claim : topology.capacityClaims().slice(
             traversal.capacityClaimOffset, traversal.capacityClaimCount)) {
      if (claim.cell >= usage.size())
        return invalid("a traversal capacity claim names an invalid cell");
      const auto &cell = topology.capacityCells()[claim.cell];
      if (usage[claim.cell] > cell.capacity ||
          claim.amount > cell.capacity - usage[claim.cell]) {
        rejectTraversal(eligibility, static_cast<PnrIndex>(traversalOrdinal));
        break;
      }
    }
  }
  return llvm::Error::success();
}

llvm::Error
commitRouteCapacityTraversals(const FrozenEndpointRoutingTopology &topology,
                              llvm::ArrayRef<PnrIndex> selectedTraversals,
                              std::vector<std::uint64_t> &usage,
                              bool enforceCapacity = true) {
  std::map<std::pair<PnrIndex, PnrIndex>, std::uint64_t> selectedClaims;
  for (PnrIndex selectedTraversal : selectedTraversals) {
    if (selectedTraversal == getInvalidPnrIndex())
      continue;
    if (selectedTraversal >= topology.traversals().size())
      return invalid("a route node has an invalid capacity traversal");
    const auto &traversal = topology.traversals()[selectedTraversal];
    if (traversal.capacityClaimOffset > topology.capacityClaims().size() ||
        traversal.capacityClaimCount >
            topology.capacityClaims().size() - traversal.capacityClaimOffset)
      return invalid("a route capacity range is out of bounds");
    for (const auto &claim : topology.capacityClaims().slice(
             traversal.capacityClaimOffset, traversal.capacityClaimCount)) {
      auto [position, inserted] = selectedClaims.try_emplace(
          std::make_pair(claim.activation, claim.cell), claim.amount);
      if (!inserted && position->second != claim.amount)
        return invalid("one route activation has inconsistent capacity claims");
    }
  }

  std::map<PnrIndex, std::uint64_t> additions;
  for (const auto &[key, amount] : selectedClaims) {
    std::uint64_t &addition = additions[key.second];
    if (amount > std::numeric_limits<std::uint64_t>::max() - addition)
      return arithmeticOverflow("route capacity addition exceeds uint64_t");
    addition += amount;
  }
  for (const auto &[cellOrdinal, amount] : additions) {
    if (cellOrdinal >= usage.size())
      return invalid("a selected route claim names an invalid capacity cell");
    const auto &cell = topology.capacityCells()[cellOrdinal];
    if (enforceCapacity && (usage[cellOrdinal] > cell.capacity ||
                            amount > cell.capacity - usage[cellOrdinal]))
      return llvm::make_error<detail::SystemCandidateInfeasible>(
          "selected service routes exceed Fabric capacity");
    if (amount > std::numeric_limits<std::uint64_t>::max() - usage[cellOrdinal])
      return arithmeticOverflow("route capacity usage exceeds uint64_t");
  }
  for (const auto &[cellOrdinal, amount] : additions)
    usage[cellOrdinal] += amount;
  return llvm::Error::success();
}

llvm::Error
removeRouteCapacityTraversals(const FrozenEndpointRoutingTopology &topology,
                              llvm::ArrayRef<PnrIndex> selectedTraversals,
                              std::vector<std::uint64_t> &usage) {
  std::vector<std::uint64_t> routeUsage(topology.capacityCells().size(), 0);
  if (llvm::Error error = commitRouteCapacityTraversals(
          topology, selectedTraversals, routeUsage, false))
    return error;
  for (PnrIndex cell = 0; cell < routeUsage.size(); ++cell) {
    if (routeUsage[cell] > usage[cell] ||
        usage[cell] - routeUsage[cell] <
            topology.capacityCells()[cell].initialOccupancy)
      return invalid("prior route capacity removal underflows occupancy");
    usage[cell] -= routeUsage[cell];
  }
  return llvm::Error::success();
}

llvm::Error commitRouteCapacity(const FrozenEndpointRoutingTopology &topology,
                                llvm::ArrayRef<MutableNode> nodes,
                                std::vector<std::uint64_t> &usage,
                                bool enforceCapacity = true) {
  std::vector<PnrIndex> traversals;
  traversals.reserve(nodes.size());
  for (const MutableNode &node : nodes)
    traversals.push_back(node.incomingTraversal);
  return commitRouteCapacityTraversals(topology, traversals, usage,
                                       enforceCapacity);
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

llvm::Expected<std::vector<RouteCost>>
computeLowerBoundArcCosts(const FrozenEndpointRoutingTopology &topology) {
  std::vector<RouteCost> traversalCosts(topology.traversals().size(), 0);
  for (const auto &[ordinal, traversal] :
       llvm::enumerate(topology.traversals())) {
    if (traversal.capacityClaimOffset > topology.capacityClaims().size() ||
        traversal.capacityClaimCount >
            topology.capacityClaims().size() - traversal.capacityClaimOffset)
      return invalid("a traversal capacity range is out of bounds");
    RouteCost cost = 0;
    for (const EndpointRoutingCapacityClaim &claim :
         topology.capacityClaims().slice(traversal.capacityClaimOffset,
                                         traversal.capacityClaimCount)) {
      auto accumulated = accumulateRouteCost(cost, claim.qCost);
      if (!accumulated)
        return accumulated.takeError();
      cost = *accumulated;
    }
    traversalCosts[ordinal] = cost;
  }
  std::vector<RouteCost> arcCosts;
  arcCosts.reserve(topology.arcs().size());
  for (const EndpointRoutingArc &arc : topology.arcs()) {
    if (arc.traversal >= traversalCosts.size())
      return invalid("a routing arc names an invalid traversal");
    arcCosts.push_back(traversalCosts[arc.traversal]);
  }
  return arcCosts;
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

struct PreparedRepairRegion final {
  std::vector<MutableNode> nodes;
  std::vector<MutableSink> retainedSinks;
  std::vector<ActiveServiceSink> reroutedSinks;
  std::optional<PnrIndex> rootedSubtreeNode;
  std::size_t retainedNodeCount = 0;
};

struct BuiltLegRoute final {
  PnrIndex rootEndpoint = 0;
  std::vector<MutableNode> nodes;
  std::vector<MutableSink> sinks;
};

llvm::Expected<std::vector<PnrIndex>>
selectedRouteTraversals(const detail::SystemServiceRoutesView &routes,
                        const SystemServiceRouteSelection &route) {
  if (route.nodeOffset > routes.nodes.size() ||
      route.nodeCount > routes.nodes.size() - route.nodeOffset)
    return invalid("prior route node range is out of bounds");
  std::vector<PnrIndex> traversals;
  traversals.reserve(route.nodeCount);
  for (const auto &node : routes.nodes.slice(route.nodeOffset, route.nodeCount))
    traversals.push_back(node.incomingTraversal);
  return traversals;
}

llvm::Error verifyLegOrder(llvm::ArrayRef<PnrIndex> order, std::size_t legCount,
                           bool requireComplete) {
  if ((requireComplete && order.size() != legCount) || order.empty())
    return invalid("service route leg order has the wrong width");
  std::vector<std::uint8_t> seen(legCount, 0);
  for (PnrIndex leg : order) {
    if (leg >= legCount || seen[leg]++)
      return invalid("service route leg order is not a permutation");
  }
  return llvm::Error::success();
}

llvm::Expected<BuiltLegRoute>
copyPriorLegRoute(const FrozenEndpointRoutingTopology &topology,
                  const detail::SystemServiceRoutesView &routes,
                  const SystemServiceRouteSelection &route) {
  if (route.nodeOffset > routes.nodes.size() ||
      route.nodeCount > routes.nodes.size() - route.nodeOffset ||
      route.sinkOffset > routes.sinks.size() ||
      route.sinkCount > routes.sinks.size() - route.sinkOffset ||
      route.nodeCount == 0)
    return invalid("prior route flat range is out of bounds");
  BuiltLegRoute result;
  result.rootEndpoint = route.rootEndpoint;
  for (const auto &node : routes.nodes.slice(route.nodeOffset, route.nodeCount))
    result.nodes.push_back({node.endpoint, node.parentNode,
                            node.incomingTraversal, false,
                            getInvalidPnrIndex()});
  for (PnrIndex child = 1; child < result.nodes.size(); ++child) {
    MutableNode &node = result.nodes[child];
    if (node.parent >= result.nodes.size() ||
        node.incomingTraversal >= topology.traversals().size())
      return invalid("prior route node is out of range");
    MutableNode &parent = result.nodes[node.parent];
    const PnrIndex group =
        topology.traversalReplicationGroups()[node.incomingTraversal];
    if (parent.hasOutgoing && parent.outgoingReplicationGroup != group)
      return invalid("prior route branch mixes replication groups");
    parent.hasOutgoing = true;
    parent.outgoingReplicationGroup = group;
  }
  for (const auto &sink : routes.sinks.slice(route.sinkOffset, route.sinkCount))
    result.sinks.push_back({sink.terminal, sink.node});
  return result;
}

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

llvm::Expected<PreparedRepairRegion>
prepareRepairRegion(const FrozenEndpointRoutingTopology &topology,
                    const BuiltLegRoute &prior,
                    llvm::ArrayRef<ActiveServiceSink> activeSinks,
                    detail::SystemServiceRouteRepairRegion region) {
  if (prior.nodes.empty())
    return invalid("repair region has no prior route tree");
  std::vector<std::uint8_t> keep(prior.nodes.size(), 0);
  std::vector<std::uint8_t> rerouteSink(prior.sinks.size(), 0);
  std::optional<PnrIndex> oldRootedNode;
  if (region.kind == detail::SystemServiceRouteRepairRegionKind::SingleSink) {
    if (region.anchor >= prior.sinks.size())
      return invalid("SingleSink repair anchor is out of range");
    rerouteSink[region.anchor] = 1;
    for (PnrIndex sink = 0; sink < prior.sinks.size(); ++sink) {
      if (rerouteSink[sink])
        continue;
      PnrIndex node = prior.sinks[sink].node;
      if (node >= prior.nodes.size())
        return invalid("prior repair sink node is out of range");
      while (node != getInvalidPnrIndex() && !keep[node]) {
        keep[node] = 1;
        node = prior.nodes[node].parent;
      }
    }
    keep[0] = 1;
  } else {
    const auto root = llvm::find_if(prior.nodes, [&](const MutableNode &node) {
      return node.endpoint == region.anchor;
    });
    if (root == prior.nodes.end())
      return invalid("RootedSubtree repair anchor is outside the route tree");
    oldRootedNode = static_cast<PnrIndex>(root - prior.nodes.begin());
    for (PnrIndex node = 0; node < prior.nodes.size(); ++node) {
      bool strictDescendant = false;
      PnrIndex ancestor = node;
      while (ancestor != getInvalidPnrIndex()) {
        if (ancestor == *oldRootedNode) {
          strictDescendant = node != *oldRootedNode;
          break;
        }
        ancestor = prior.nodes[ancestor].parent;
      }
      keep[node] = !strictDescendant;
    }
    for (PnrIndex sink = 0; sink < prior.sinks.size(); ++sink) {
      PnrIndex node = prior.sinks[sink].node;
      if (node >= prior.nodes.size())
        return invalid("prior repair sink node is out of range");
      while (node != getInvalidPnrIndex() && node != *oldRootedNode)
        node = prior.nodes[node].parent;
      rerouteSink[sink] = node == *oldRootedNode;
    }
  }

  PreparedRepairRegion result;
  std::vector<PnrIndex> remap(prior.nodes.size(), getInvalidPnrIndex());
  for (PnrIndex oldNode = 0; oldNode < prior.nodes.size(); ++oldNode) {
    if (!keep[oldNode])
      continue;
    remap[oldNode] = static_cast<PnrIndex>(result.nodes.size());
    const MutableNode &node = prior.nodes[oldNode];
    const PnrIndex parent = node.parent == getInvalidPnrIndex()
                                ? getInvalidPnrIndex()
                                : remap[node.parent];
    if (node.parent != getInvalidPnrIndex() && parent == getInvalidPnrIndex())
      return invalid("repair region retained a node without its parent");
    result.nodes.push_back({node.endpoint, parent, node.incomingTraversal,
                            false, getInvalidPnrIndex()});
  }
  for (PnrIndex child = 1; child < result.nodes.size(); ++child) {
    MutableNode &node = result.nodes[child];
    if (node.incomingTraversal >= topology.traversals().size())
      return invalid("repair region retained an invalid traversal");
    MutableNode &parent = result.nodes[node.parent];
    const PnrIndex group =
        topology.traversalReplicationGroups()[node.incomingTraversal];
    if (parent.hasOutgoing && parent.outgoingReplicationGroup != group)
      return invalid("repair region retained mixed replication groups");
    parent.hasOutgoing = true;
    parent.outgoingReplicationGroup = group;
  }
  for (PnrIndex sink = 0; sink < prior.sinks.size(); ++sink) {
    const MutableSink &record = prior.sinks[sink];
    if (!rerouteSink[sink]) {
      if (record.node >= remap.size() ||
          remap[record.node] == getInvalidPnrIndex())
        return invalid("repair region removed a retained sink node");
      result.retainedSinks.push_back({record.terminal, remap[record.node]});
      continue;
    }
    const auto active =
        llvm::find_if(activeSinks, [&](const ActiveServiceSink &candidate) {
          return candidate.terminal == record.terminal;
        });
    if (active == activeSinks.end())
      return invalid("repair region sink is outside the active H domain");
    result.reroutedSinks.push_back(*active);
  }
  if (result.reroutedSinks.empty())
    return invalid("repair region contains no sink obligation");
  llvm::sort(result.reroutedSinks, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.terminalKey, lhs.endpoints) <
           std::tie(rhs.terminalKey, rhs.endpoints);
  });
  if (oldRootedNode)
    result.rootedSubtreeNode = remap[*oldRootedNode];
  result.retainedNodeCount = result.nodes.size();
  return result;
}

} // namespace

llvm::Expected<detail::BuiltSystemServiceRoutes>
loom::pnr::detail::buildSystemServiceRoutes(
    const FrozenSystemPnrProblem &problem,
    llvm::ArrayRef<PnrIndex> threadChoices,
    llvm::ArrayRef<PnrIndex> graphChoices,
    const SystemServiceRouteBuildRequest &request,
    std::uint64_t &endpointExpansions) {
  endpointExpansions = 0;
  BuiltSystemServiceRoutes result;
  const FrozenEndpointRoutingTopology &topology = problem.routingTopology();
  if (llvm::Error error = verifyLegOrder(
          request.legOrder, problem.serviceLegs().size(), !request.priorRoutes))
    return std::move(error);
  if (request.lowerBoundArcCosts.size() != topology.arcs().size())
    return invalid("lower-bound arc-cost vector has the wrong width");
  if (request.exclusion &&
      (request.exclusion->leg >= problem.serviceLegs().size() ||
       request.exclusion->traversal >= topology.traversals().size()))
    return invalid("a route traversal exclusion is out of range");
  if (request.repairRegion &&
      (!request.priorRoutes ||
       request.repairRegion->leg >= problem.serviceLegs().size() ||
       !llvm::is_contained(request.legOrder, request.repairRegion->leg)))
    return invalid("a route repair region has no valid prior route");
  auto atomicPatterns = buildAtomicPatternCatalog(topology);
  if (!atomicPatterns)
    return atomicPatterns.takeError();
  EndpointRouteSearchScratch search;
  if (llvm::Error error = search.prepare(endpointRoutingGraphView(topology)))
    return std::move(error);
  std::vector<std::uint64_t> capacityUsage;
  capacityUsage.reserve(topology.capacityCells().size());
  for (const auto &cell : topology.capacityCells())
    capacityUsage.push_back(cell.initialOccupancy);

  if (request.priorRoutes) {
    if (request.priorRoutes->routes.size() != problem.serviceLegs().size())
      return invalid("prior service route count has the wrong width");
    for (PnrIndex leg = 0; leg < request.priorRoutes->routes.size(); ++leg) {
      const auto &prior = request.priorRoutes->routes[leg];
      if (prior.leg != leg)
        return invalid("prior service routes are not in canonical leg order");
      auto traversals = selectedRouteTraversals(*request.priorRoutes, prior);
      if (!traversals)
        return traversals.takeError();
      if (llvm::Error error = commitRouteCapacityTraversals(
              topology, *traversals, capacityUsage, false))
        return std::move(error);
    }
  }

  std::vector<std::optional<BuiltLegRoute>> builtLegs(
      problem.serviceLegs().size());
  if (request.priorRoutes)
    for (PnrIndex leg = 0; leg < request.priorRoutes->routes.size(); ++leg) {
      auto copied = copyPriorLegRoute(topology, *request.priorRoutes,
                                      request.priorRoutes->routes[leg]);
      if (!copied)
        return copied.takeError();
      builtLegs[leg] = std::move(*copied);
    }

  for (PnrIndex legOrdinal : request.legOrder) {
    if (request.priorRoutes) {
      auto traversals = selectedRouteTraversals(
          *request.priorRoutes, request.priorRoutes->routes[legOrdinal]);
      if (!traversals)
        return traversals.takeError();
      if (llvm::Error error = removeRouteCapacityTraversals(
              topology, *traversals, capacityUsage))
        return std::move(error);
    }
    auto currentArcCosts = request.currentArcCosts(capacityUsage);
    if (!currentArcCosts)
      return currentArcCosts.takeError();
    if (currentArcCosts->size() != topology.arcs().size())
      return invalid("current arc-cost vector has the wrong width");
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
    std::vector<ActiveServiceSink> sinksToRoute = *activeSinks;
    std::optional<PnrIndex> rootedSubtreeNode;
    std::size_t retainedNodeCount = 0;
    if (request.repairRegion && request.repairRegion->leg == legOrdinal) {
      if (!builtLegs[legOrdinal])
        return invalid("repair region has no prior route selection");
      auto prepared = prepareRepairRegion(topology, *builtLegs[legOrdinal],
                                          *activeSinks, *request.repairRegion);
      if (!prepared)
        return prepared.takeError();
      nodes = std::move(prepared->nodes);
      sinks = std::move(prepared->retainedSinks);
      sinksToRoute = std::move(prepared->reroutedSinks);
      rootedSubtreeNode = prepared->rootedSubtreeNode;
      retainedNodeCount = prepared->retainedNodeCount;
    }
    llvm::DenseMap<PnrIndex, PnrIndex> nodeByEndpoint;
    for (PnrIndex node = 0; node < nodes.size(); ++node)
      if (!nodeByEndpoint.try_emplace(nodes[node].endpoint, node).second)
        return invalid("repair region retained a duplicate endpoint");
    for (const ActiveServiceSink &activeSink : sinksToRoute) {
      std::vector<PnrIndex> sourceEndpoints;
      std::vector<PnrIndex> sourceReplicationGroups;
      if (nodes.empty()) {
        sourceEndpoints.assign(sourceDomain->begin(), sourceDomain->end());
        sourceReplicationGroups.assign(sourceEndpoints.size(),
                                       getInvalidPnrIndex());
      } else {
        std::vector<std::pair<PnrIndex, PnrIndex>> frontier;
        frontier.reserve(nodes.size());
        for (const auto &[nodeOrdinal, node] : llvm::enumerate(nodes))
          if ((!rootedSubtreeNode || nodeOrdinal == *rootedSubtreeNode ||
               nodeOrdinal >= retainedNodeCount) &&
              (!node.hasOutgoing ||
               node.outgoingReplicationGroup != getInvalidPnrIndex()))
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
      if (request.exclusion && request.exclusion->leg == legOrdinal)
        rejectTraversal(eligibility, request.exclusion->traversal);
      if (request.enforceCapacity)
        if (llvm::Error error =
                applyCapacityEligibility(topology, capacityUsage, eligibility))
          return std::move(error);
      std::optional<RouteProbe> bestProbe;
      std::string lastRouteDiagnostic;
      const auto tryProbe =
          [&](llvm::ArrayRef<PnrIndex> sources,
              llvm::ArrayRef<PnrIndex> sourceGroups,
              llvm::ArrayRef<std::uint64_t> eligibleTraversals,
              std::optional<AtomicPatternUpgrade> upgrade) -> llvm::Error {
        auto routed = search.search(
            {sources,
             sourceGroups,
             activeSink.endpoints,
             targetRanks,
             request.lowerBoundArcCosts,
             *currentArcCosts,
             leg.requiredPayloadWidthBits,
             0,
             problem.config().policy().search.routing.endpointExpansionLimit,
             eligibleTraversals,
             std::nullopt,
             {},
             false,
             {},
             {}});
        const std::uint64_t consumed = search.endpointExpansionCount();
        if (consumed >
            std::numeric_limits<std::uint64_t>::max() - endpointExpansions)
          return arithmeticOverflow(
              "endpoint expansion accounting exceeds uint64_t");
        endpointExpansions += consumed;
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
          if (rootedSubtreeNode && upgrade.node != *rootedSubtreeNode &&
              upgrade.node < retainedNodeCount)
            continue;
          auto upgradedEligibility = eligibility;
          admitTraversal(upgradedEligibility, upgrade.extraTraversal);
          if (request.exclusion && request.exclusion->leg == legOrdinal)
            rejectTraversal(upgradedEligibility, request.exclusion->traversal);
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
    if (llvm::Error error = commitRouteCapacity(topology, nodes, capacityUsage,
                                                request.enforceCapacity))
      return std::move(error);
    builtLegs[legOrdinal] = BuiltLegRoute{nodes.front().endpoint,
                                          std::move(nodes), std::move(sinks)};
  }

  for (PnrIndex legOrdinal = 0; legOrdinal < builtLegs.size(); ++legOrdinal) {
    if (!builtLegs[legOrdinal])
      return invalid("service route leg order omitted a leg");
    BuiltLegRoute &leg = *builtLegs[legOrdinal];
    auto nodeOffset =
        checked(nodeOffsetContext, result.selections.nodes.size());
    auto nodeCount = checked(nodeOffsetContext, leg.nodes.size());
    auto sinkOffset =
        checked(sinkOffsetContext, result.selections.sinks.size());
    auto sinkCount = checked(sinkOffsetContext, leg.sinks.size());
    if (!nodeOffset)
      return nodeOffset.takeError();
    if (!nodeCount)
      return nodeCount.takeError();
    if (!sinkOffset)
      return sinkOffset.takeError();
    if (!sinkCount)
      return sinkCount.takeError();
    for (const MutableNode &node : leg.nodes)
      result.selections.nodes.push_back(
          {node.endpoint, node.parent, node.incomingTraversal});
    for (const MutableSink &sink : leg.sinks)
      result.selections.sinks.push_back({sink.terminal, sink.node});
    result.selections.routes.push_back({legOrdinal, leg.rootEndpoint,
                                        *nodeOffset, *nodeCount, *sinkOffset,
                                        *sinkCount});
  }
  if (request.enforceCapacity)
    if (llvm::Error error = verifySystemServiceRoutes(
            problem, threadChoices, graphChoices, result.selections.routes,
            result.selections.nodes, result.selections.sinks))
      return std::move(error);
  result.capacityUsage = std::move(capacityUsage);
  return result;
}

llvm::Expected<std::vector<RouteCost>>
loom::pnr::detail::buildSystemServiceRouteLowerBoundArcCosts(
    const FrozenEndpointRoutingTopology &topology) {
  return computeLowerBoundArcCosts(topology);
}

llvm::Expected<std::vector<std::uint64_t>>
loom::pnr::detail::measureSystemServiceRouteCapacityUsage(
    const FrozenEndpointRoutingTopology &topology,
    SystemServiceRoutesView routes, bool enforceCapacity) {
  std::vector<std::uint64_t> usage;
  usage.reserve(topology.capacityCells().size());
  for (const auto &cell : topology.capacityCells())
    usage.push_back(cell.initialOccupancy);
  for (const auto &route : routes.routes) {
    auto traversals = selectedRouteTraversals(routes, route);
    if (!traversals)
      return traversals.takeError();
    if (llvm::Error error = commitRouteCapacityTraversals(
            topology, *traversals, usage, enforceCapacity))
      return std::move(error);
  }
  return usage;
}

llvm::Expected<std::uint64_t>
loom::pnr::detail::measureSystemServiceRouteTraversalClaim(
    const FrozenEndpointRoutingTopology &topology,
    SystemServiceRoutesView routes) {
  std::uint64_t total = 0;
  for (const SystemServiceRouteSelection &route : routes.routes) {
    if (route.nodeOffset > routes.nodes.size() ||
        route.nodeCount > routes.nodes.size() - route.nodeOffset)
      return invalid("route traversal-claim node range is out of bounds");
    std::set<std::pair<PnrIndex, PnrIndex>> selectedClaims;
    for (const SystemServiceRouteNodeSelection &node :
         routes.nodes.slice(route.nodeOffset, route.nodeCount)) {
      if (node.incomingTraversal == getInvalidPnrIndex())
        continue;
      if (node.incomingTraversal >= topology.traversals().size())
        return invalid(
            "route traversal-claim projection has a foreign traversal");
      const EndpointRoutingTraversal &traversal =
          topology.traversals()[node.incomingTraversal];
      if (traversal.capacityClaimOffset > topology.capacityClaims().size() ||
          traversal.capacityClaimCount >
              topology.capacityClaims().size() - traversal.capacityClaimOffset)
        return invalid("route traversal-claim capacity range is out of bounds");
      for (const EndpointRoutingCapacityClaim &claim :
           topology.capacityClaims().slice(traversal.capacityClaimOffset,
                                           traversal.capacityClaimCount)) {
        if (!selectedClaims.emplace(claim.activation, claim.cell).second)
          continue;
        if (claim.qCost > std::numeric_limits<std::uint64_t>::max() - total)
          return invalid("route traversal-claim total exceeds u64");
        total += claim.qCost;
      }
    }
  }
  return total;
}

llvm::Expected<std::vector<detail::SystemFixedTerminalCapacityConflict>>
loom::pnr::detail::analyzeSystemFixedTerminalCapacityConflicts(
    const FrozenSystemPnrProblem &problem, SystemServiceRoutesView routes,
    llvm::ArrayRef<std::uint64_t> capacityUsage) {
  const FrozenEndpointRoutingTopology &topology = problem.routingTopology();
  if (capacityUsage.size() != topology.capacityCells().size())
    return invalid("capacity-cut usage has the wrong width");
  if (routes.routes.size() != problem.serviceLegs().size())
    return invalid("capacity-cut route count has the wrong width");

  std::vector<std::uint32_t> traversalPayloadCapacity(
      topology.traversals().size(), 0);
  for (const EndpointRoutingArc &arc : topology.arcs()) {
    if (arc.traversal >= traversalPayloadCapacity.size())
      return invalid("capacity-cut arc names an invalid traversal");
    traversalPayloadCapacity[arc.traversal] = std::max(
        traversalPayloadCapacity[arc.traversal], arc.payloadCapacityBits);
  }

  std::vector<SystemFixedTerminalCapacityConflict> result;
  std::vector<std::uint8_t> blockedTraversal(topology.traversals().size(), 0);
  std::vector<std::uint64_t> minimumTraversalClaim(
      topology.traversals().size(), std::numeric_limits<std::uint64_t>::max());
  std::vector<std::uint8_t> reachable(topology.endpoints().size(), 0);
  std::vector<PnrIndex> worklist;
  worklist.reserve(topology.endpoints().size());

  for (PnrIndex capacityCell = 0;
       capacityCell < topology.capacityCells().size(); ++capacityCell) {
    const EndpointRoutingCapacityCell &cell =
        topology.capacityCells()[capacityCell];
    if (capacityUsage[capacityCell] <= cell.capacity)
      continue;

    std::fill(blockedTraversal.begin(), blockedTraversal.end(), 0);
    std::fill(minimumTraversalClaim.begin(), minimumTraversalClaim.end(),
              std::numeric_limits<std::uint64_t>::max());
    for (PnrIndex traversalOrdinal = 0;
         traversalOrdinal < topology.traversals().size(); ++traversalOrdinal) {
      const EndpointRoutingTraversal &traversal =
          topology.traversals()[traversalOrdinal];
      if (traversal.capacityClaimOffset > topology.capacityClaims().size() ||
          traversal.capacityClaimCount >
              topology.capacityClaims().size() - traversal.capacityClaimOffset)
        return invalid("capacity-cut traversal claim range is out of bounds");
      for (const EndpointRoutingCapacityClaim &claim :
           topology.capacityClaims().slice(traversal.capacityClaimOffset,
                                           traversal.capacityClaimCount)) {
        if (claim.cell != capacityCell || claim.amount == 0)
          continue;
        blockedTraversal[traversalOrdinal] = 1;
        minimumTraversalClaim[traversalOrdinal] =
            std::min(minimumTraversalClaim[traversalOrdinal], claim.amount);
      }
    }

    SystemFixedTerminalCapacityConflict conflict;
    conflict.capacityCell = capacityCell;
    conflict.usage = capacityUsage[capacityCell];
    conflict.capacity = cell.capacity;
    conflict.mandatoryUsage = cell.initialOccupancy;

    for (const SystemServiceRouteSelection &route : routes.routes) {
      if (route.leg >= problem.serviceLegs().size() ||
          route.nodeOffset > routes.nodes.size() ||
          route.nodeCount > routes.nodes.size() - route.nodeOffset ||
          route.sinkOffset > routes.sinks.size() ||
          route.sinkCount > routes.sinks.size() - route.sinkOffset ||
          route.rootEndpoint >= topology.endpoints().size())
        return invalid("capacity-cut route range is out of bounds");
      const auto routeNodes =
          routes.nodes.slice(route.nodeOffset, route.nodeCount);
      const bool contributes = llvm::any_of(
          routeNodes, [&](const SystemServiceRouteNodeSelection &node) {
            return node.incomingTraversal != getInvalidPnrIndex() &&
                   node.incomingTraversal < blockedTraversal.size() &&
                   blockedTraversal[node.incomingTraversal] != 0;
          });
      if (!contributes)
        continue;

      SystemFixedTerminalCapacityLegEvidence evidence;
      evidence.leg = route.leg;
      evidence.sourceEndpoint = route.rootEndpoint;
      for (const SystemServiceRouteNodeSelection &node : routeNodes)
        if (node.incomingTraversal != getInvalidPnrIndex() &&
            node.incomingTraversal < blockedTraversal.size() &&
            blockedTraversal[node.incomingTraversal] != 0)
          evidence.claimingTraversals.push_back(node.incomingTraversal);
      llvm::sort(evidence.claimingTraversals);
      evidence.claimingTraversals.erase(
          std::unique(evidence.claimingTraversals.begin(),
                      evidence.claimingTraversals.end()),
          evidence.claimingTraversals.end());

      const std::uint32_t requiredPayloadWidth =
          problem.serviceLegs()[route.leg].requiredPayloadWidthBits;
      std::uint64_t minimumClaim = std::numeric_limits<std::uint64_t>::max();
      for (PnrIndex traversal = 0; traversal < blockedTraversal.size();
           ++traversal)
        if (blockedTraversal[traversal] &&
            traversalPayloadCapacity[traversal] >= requiredPayloadWidth)
          minimumClaim =
              std::min(minimumClaim, minimumTraversalClaim[traversal]);
      if (minimumClaim == std::numeric_limits<std::uint64_t>::max())
        return invalid("capacity-cut contributing leg has no compatible claim");
      evidence.minimumClaim = minimumClaim;

      std::fill(reachable.begin(), reachable.end(), 0);
      worklist.clear();
      reachable[route.rootEndpoint] = 1;
      worklist.push_back(route.rootEndpoint);
      for (std::size_t cursor = 0; cursor < worklist.size(); ++cursor) {
        const PnrIndex endpoint = worklist[cursor];
        if (endpoint + 1 >= topology.adjacencyOffsets().size())
          return invalid("capacity-cut endpoint has no adjacency range");
        const PnrIndex begin = topology.adjacencyOffsets()[endpoint];
        const PnrIndex end = topology.adjacencyOffsets()[endpoint + 1];
        if (begin > end || end > topology.arcs().size())
          return invalid("capacity-cut adjacency range is out of bounds");
        for (PnrIndex arcOrdinal = begin; arcOrdinal < end; ++arcOrdinal) {
          const EndpointRoutingArc &arc = topology.arcs()[arcOrdinal];
          if (arc.traversal >= blockedTraversal.size() ||
              arc.target >= reachable.size())
            return invalid("capacity-cut arc is outside the frozen topology");
          if (blockedTraversal[arc.traversal] ||
              arc.payloadCapacityBits < requiredPayloadWidth ||
              reachable[arc.target])
            continue;
          reachable[arc.target] = 1;
          worklist.push_back(arc.target);
        }
      }
      evidence.reachableEndpointCount = worklist.size();

      for (const SystemServiceRouteSinkSelection &sink :
           routes.sinks.slice(route.sinkOffset, route.sinkCount)) {
        if (sink.node >= routeNodes.size() ||
            routeNodes[sink.node].endpoint >= reachable.size())
          return invalid("capacity-cut sink is outside its route tree");
        const PnrIndex endpoint = routeNodes[sink.node].endpoint;
        evidence.sinkEndpoints.push_back(endpoint);
        if (!reachable[endpoint])
          evidence.unreachableSinkEndpoints.push_back(endpoint);
      }
      if (!evidence.isForced()) {
        conflict.logicalNets.push_back(std::move(evidence));
        continue;
      }
      if (minimumClaim >
          std::numeric_limits<std::uint64_t>::max() - conflict.mandatoryUsage)
        return arithmeticOverflow(
            "capacity-cut mandatory usage exceeds uint64_t");
      conflict.mandatoryUsage += minimumClaim;
      conflict.logicalNets.push_back(std::move(evidence));
    }
    result.push_back(std::move(conflict));
  }
  return result;
}

llvm::Expected<std::vector<PnrIndex>>
loom::pnr::detail::buildSystemServiceRouteLegOrder(
    const FrozenEndpointRoutingTopology &topology,
    SystemServiceRoutesView routes,
    llvm::ArrayRef<std::uint64_t> capacityUsage) {
  if (capacityUsage.size() != topology.capacityCells().size())
    return invalid("route-order capacity usage has the wrong width");
  struct OrderKey final {
    PnrIndex leg = 0;
    std::uint8_t routeStateRank = 2;
    RouteCost conflictPressure = 0;
  };
  std::vector<OrderKey> keys;
  keys.reserve(routes.routes.size());
  for (const auto &route : routes.routes) {
    auto traversals = selectedRouteTraversals(routes, route);
    if (!traversals)
      return traversals.takeError();
    std::map<std::pair<PnrIndex, PnrIndex>, std::pair<std::uint64_t, RouteCost>>
        selectedClaims;
    for (PnrIndex traversalOrdinal : *traversals) {
      if (traversalOrdinal == getInvalidPnrIndex())
        continue;
      if (traversalOrdinal >= topology.traversals().size())
        return invalid("route-order traversal is out of range");
      const auto &traversal = topology.traversals()[traversalOrdinal];
      if (traversal.capacityClaimOffset > topology.capacityClaims().size() ||
          traversal.capacityClaimCount >
              topology.capacityClaims().size() - traversal.capacityClaimOffset)
        return invalid("route-order capacity range is out of bounds");
      for (const auto &claim : topology.capacityClaims().slice(
               traversal.capacityClaimOffset, traversal.capacityClaimCount)) {
        const auto key = std::make_pair(claim.activation, claim.cell);
        auto [position, inserted] = selectedClaims.try_emplace(
            key,
            std::make_pair(claim.amount, static_cast<RouteCost>(claim.qCost)));
        if (!inserted &&
            position->second !=
                std::make_pair(claim.amount,
                               static_cast<RouteCost>(claim.qCost)))
          return invalid("one route activation has inconsistent priced claims");
      }
    }
    RouteCost pressure = 0;
    for (const auto &[key, claim] : selectedClaims) {
      if (key.second >= capacityUsage.size())
        return invalid("route-order claim names an invalid capacity cell");
      auto overuse = normalizedRouteOveruseCost(
          capacityUsage[key.second], 0,
          topology.capacityCells()[key.second].capacity);
      if (!overuse)
        return overuse.takeError();
      auto contribution = scaledRouteProduct(claim.second, *overuse);
      if (!contribution)
        return contribution.takeError();
      auto accumulated = accumulateRouteCost(pressure, *contribution);
      if (!accumulated)
        return accumulated.takeError();
      pressure = *accumulated;
    }
    keys.push_back({route.leg,
                    pressure == 0 ? std::uint8_t{2} : std::uint8_t{1},
                    pressure});
  }
  llvm::sort(keys, [](const OrderKey &lhs, const OrderKey &rhs) {
    if (lhs.routeStateRank != rhs.routeStateRank)
      return lhs.routeStateRank < rhs.routeStateRank;
    if (lhs.conflictPressure != rhs.conflictPressure)
      return lhs.conflictPressure > rhs.conflictPressure;
    return lhs.leg < rhs.leg;
  });
  std::vector<PnrIndex> order;
  order.reserve(keys.size());
  for (const OrderKey &key : keys)
    order.push_back(key.leg);
  return order;
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
  std::vector<std::uint64_t> capacityUsage;
  capacityUsage.reserve(topology.capacityCells().size());
  for (const auto &cell : topology.capacityCells())
    capacityUsage.push_back(cell.initialOccupancy);
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
    std::vector<PnrIndex> selectedTraversals;
    selectedTraversals.reserve(routeNodes.size());
    for (const auto &node : routeNodes)
      selectedTraversals.push_back(node.incomingTraversal);
    if (llvm::Error error = commitRouteCapacityTraversals(
            topology, selectedTraversals, capacityUsage,
            /*enforceCapacity=*/false))
      return error;
    expectedNodeOffset += route.nodeCount;
    expectedSinkOffset += route.sinkCount;
  }
  if (expectedNodeOffset != nodes.size() || expectedSinkOffset != sinks.size())
    return invalid("service route flat arrays contain trailing records");
  return llvm::Error::success();
}
