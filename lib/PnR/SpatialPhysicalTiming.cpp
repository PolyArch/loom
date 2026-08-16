#include "SpatialPhysicalTiming.h"

#include "StaticSchedulePressure.h"

#include "Fabric/Identity/FabricRefBytes.h"
#include "PnR/RouteTreeState.h"
#include "PnR/SpatialPnrProblem.h"

#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

using namespace loom::pnr;

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "spatial_physical_timing_invalid: " + message);
}

llvm::Expected<std::uint64_t> add(std::uint64_t lhs, std::uint64_t rhs,
                                  llvm::StringRef what) {
  if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs)
    return invalid(what + " exceeds u64");
  return lhs + rhs;
}

} // namespace

llvm::Error detail::observeSpatialPhysicalTimingEndpoint(
    std::uint64_t arrival, std::uint64_t required,
    SpatialLogicalNetPhysicalTiming &timing) {
  if (required == 0)
    return invalid("physical timing provider has a zero delay budget");
  timing.worstArrivalDelayQuanta =
      std::max(timing.worstArrivalDelayQuanta, arrival);
  if (arrival <= required)
    return llvm::Error::success();
  auto total = add(timing.totalNegativeSlackQuanta, arrival - required,
                   "logical-net negative slack");
  if (!total)
    return total.takeError();
  timing.totalNegativeSlackQuanta = *total;
  return llvm::Error::success();
}

llvm::Expected<std::uint64_t> detail::advanceSpatialPhysicalTiming(
    std::uint64_t delayQuanta,
    ::loom::fabric::FabricPhysicalTimingBoundaryKind boundary,
    std::uint64_t arrival, std::uint64_t required,
    SpatialLogicalNetPhysicalTiming &timing) {
  auto reached = add(arrival, delayQuanta, "combinational route arrival");
  if (!reached)
    return reached.takeError();
  switch (boundary) {
  case ::loom::fabric::FabricPhysicalTimingBoundaryKind::Combinational:
    return *reached;
  case ::loom::fabric::FabricPhysicalTimingBoundaryKind::RegisteredDestination:
    if (llvm::Error error =
            observeSpatialPhysicalTimingEndpoint(*reached, required, timing))
      return std::move(error);
    return std::uint64_t{0};
  }
  llvm_unreachable("unknown Fabric physical timing boundary");
}

llvm::Expected<RouteCost>
detail::physicalTimingDrivenTraversalCost(std::uint64_t delayQuanta,
                                          std::uint64_t requiredQuanta,
                                          std::uint64_t structuralCriticality) {
  if (delayQuanta == 0)
    return invalid("physical traversal delay is zero");
  if (requiredQuanta == 0)
    return invalid("physical timing provider has a zero delay budget");
  const unsigned __int128 scaled =
      static_cast<unsigned __int128>(delayQuanta) * routeCostScale;
  const unsigned __int128 normalized =
      (scaled + requiredQuanta - 1) / requiredQuanta;
  const unsigned __int128 critical =
      normalized * (static_cast<unsigned __int128>(structuralCriticality) + 1);
  if (critical > maxFiniteRouteCost)
    return invalid("critical physical traversal cost exceeds the finite route "
                   "domain");
  return static_cast<RouteCost>(critical);
}

llvm::Expected<RouteCost> detail::physicalTimingDrivenNegativeSlackCost(
    std::uint64_t excessDeltaQuanta, std::uint64_t requiredQuanta,
    std::uint64_t structuralCriticality) {
  if (excessDeltaQuanta == 0)
    return RouteCost{0};
  if (requiredQuanta == 0)
    return invalid("physical timing provider has a zero delay budget");
  const unsigned __int128 scaled =
      static_cast<unsigned __int128>(excessDeltaQuanta) * routeCostScale;
  const unsigned __int128 normalized =
      (scaled + requiredQuanta - 1) / requiredQuanta;
  const unsigned __int128 critical =
      normalized * (static_cast<unsigned __int128>(structuralCriticality) + 1);
  if (critical > maxFiniteRouteCost)
    return invalid("critical negative-slack cost exceeds the finite route "
                   "domain");
  return static_cast<RouteCost>(critical);
}

namespace {

llvm::Expected<std::uint64_t>
logicalNetStructuralCriticality(const FrozenSpatialPnrProblem &problem,
                                PnrIndex logicalNet) {
  const auto nets = problem.transfers().logicalNets();
  if (logicalNet >= nets.size())
    return invalid("logical net is out of range");
  const FrozenSpatialLogicalNet &net = nets[logicalNet];
  const auto sinks = problem.transfers().logicalNetSinks();
  if (net.sinkOffset > sinks.size() ||
      net.sinkCount > sinks.size() - net.sinkOffset)
    return invalid("logical-net sink range is malformed");
  const auto *producer =
      std::get_if<::dataflow::ActorTokenResultRef>(&net.producer);
  if (!producer)
    return std::uint64_t{0};
  std::uint64_t criticality = 0;
  for (const auto &sink : sinks.slice(net.sinkOffset, net.sinkCount))
    if (const auto *consumer =
            std::get_if<::dataflow::ActorTokenOperandRef>(&sink))
      criticality =
          std::max(criticality,
                   problem.schedulePressure().edgeWeight(*producer, *consumer));
  return criticality;
}

llvm::Expected<std::optional<PnrIndex>>
selectedLocalTraversal(const FrozenSpatialPnrProblem &problem,
                       FrozenSpatialTerminalBinding binding,
                       llvm::ArrayRef<PnrIndex> portAttachments,
                       llvm::ArrayRef<PnrIndex> graphBoundaryAttachments) {
  PnrIndex option = getInvalidPnrIndex();
  switch (binding.kind) {
  case FrozenSpatialTerminalBindingKind::PortDemand:
    if (binding.index >= portAttachments.size())
      return invalid("physical timing PortDemand selection is out of range");
    option = portAttachments[binding.index];
    break;
  case FrozenSpatialTerminalBindingKind::GraphBoundary:
    if (binding.index >= graphBoundaryAttachments.size())
      return invalid(
          "physical timing graph-boundary selection is out of range");
    option = graphBoundaryAttachments[binding.index];
    break;
  }
  if (option >= problem.ports().attachmentOptions().size())
    return invalid("physical timing attachment option is out of range");
  return problem.ports().attachmentOptions()[option].localTraversal;
}

llvm::Expected<std::uint64_t>
traversePhysicalTiming(const FrozenSpatialRoutingGraph &routing,
                       PnrIndex traversalOrdinal, std::uint64_t arrival,
                       std::uint64_t required,
                       detail::SpatialLogicalNetPhysicalTiming &result) {
  if (traversalOrdinal >= routing.traversals().size())
    return invalid("physical timing traversal is out of range");
  const FrozenSpatialTraversal &traversal =
      routing.traversals()[traversalOrdinal];
  return detail::advanceSpatialPhysicalTiming(traversal.physicalDelayQuanta,
                                              traversal.physicalTimingBoundary,
                                              arrival, required, result);
}

llvm::Expected<std::uint64_t>
deriveSourceArrival(const FrozenSpatialPnrProblem &problem, PnrIndex logicalNet,
                    llvm::ArrayRef<PnrIndex> portAttachments,
                    llvm::ArrayRef<PnrIndex> graphBoundaryAttachments,
                    detail::SpatialLogicalNetPhysicalTiming &timing) {
  if (logicalNet >= problem.transfers().logicalNetSourceBindings().size())
    return invalid("logical-net source timing domain is out of range");
  auto rootLocalTraversal = selectedLocalTraversal(
      problem, problem.transfers().logicalNetSourceBindings()[logicalNet],
      portAttachments, graphBoundaryAttachments);
  if (!rootLocalTraversal)
    return rootLocalTraversal.takeError();
  if (!*rootLocalTraversal)
    return std::uint64_t{0};
  return traversePhysicalTiming(
      problem.routing(), **rootLocalTraversal, 0,
      problem.routing().requiredCombinationalDelayQuanta(), timing);
}

llvm::Error deriveRouteNodeArrivals(
    const FrozenSpatialPnrProblem &problem, PnrIndex logicalNet,
    const RouteTreeState &route, llvm::ArrayRef<PnrIndex> portAttachments,
    llvm::ArrayRef<PnrIndex> graphBoundaryAttachments,
    detail::SpatialLogicalNetPhysicalTiming &timing,
    std::vector<std::uint64_t> &arrivals,
    std::vector<std::pair<PnrIndex, std::uint64_t>> &worklist) {
  const FrozenSpatialRoutingGraph &routing = problem.routing();
  if (logicalNet >= problem.transfers().logicalNets().size() ||
      logicalNet >= problem.transfers().logicalNetSourceBindings().size())
    return invalid("logical net timing domain is out of range");
  if (!route.isRouted())
    return invalid("RouteTree timing arrivals require a routed net");
  if (&route.routingGraph() != &routing)
    return invalid("RouteTree belongs to another frozen routing graph");
  const std::optional<PnrIndex> sourceEndpoint = route.sourceEndpoint();
  if (!sourceEndpoint)
    return invalid("routed net has no source endpoint");
  const std::optional<PnrIndex> sourceSlot = route.findNode(*sourceEndpoint);
  if (!sourceSlot)
    return invalid("routed net source has no RouteTree node");
  const std::uint64_t required = routing.requiredCombinationalDelayQuanta();
  if (required == 0)
    return invalid("physical timing provider has a zero delay budget");

  auto rootArrival = deriveSourceArrival(problem, logicalNet, portAttachments,
                                         graphBoundaryAttachments, timing);
  if (!rootArrival)
    return rootArrival.takeError();

  worklist.clear();
  arrivals.resize(route.nodeStorage().size());
  worklist.reserve(route.activeNodeCount());
  worklist.emplace_back(*sourceSlot, *rootArrival);
  for (std::size_t cursor = 0; cursor < worklist.size(); ++cursor) {
    const auto [slot, arrival] = worklist[cursor];
    if (slot >= route.nodeStorage().size())
      return invalid("RouteTree timing walk reached a foreign node");
    const RouteTreeNode &node = route.node(slot);
    if (!node.isActive())
      return invalid("RouteTree timing walk reached an inactive node");
    arrivals[slot] = arrival;
    for (PnrIndex child = node.firstChild; child != getInvalidPnrIndex();
         child = route.node(child).nextSibling) {
      if (child >= route.nodeStorage().size())
        return invalid("RouteTree timing child is out of range");
      const RouteTreeNode &childNode = route.node(child);
      if (!childNode.isActive() ||
          childNode.parentArc >= routing.routingArcs().size())
        return invalid("RouteTree timing child has an invalid parent arc");
      const PnrIndex traversalOrdinal =
          routing.routingArcs()[childNode.parentArc].traversal;
      auto reached = traversePhysicalTiming(routing, traversalOrdinal, arrival,
                                            required, timing);
      if (!reached)
        return reached.takeError();
      worklist.emplace_back(child, *reached);
    }
  }
  if (worklist.size() != route.activeNodeCount())
    return invalid("RouteTree timing walk did not cover every active node");
  return llvm::Error::success();
}

} // namespace

llvm::Expected<std::optional<PnrIndex>>
detail::projectSelectedSpatialTerminalTraversal(
    const FrozenSpatialPnrProblem &problem,
    FrozenSpatialTerminalBinding binding,
    llvm::ArrayRef<PnrIndex> portAttachments,
    llvm::ArrayRef<PnrIndex> graphBoundaryAttachments) {
  return selectedLocalTraversal(problem, binding, portAttachments,
                                graphBoundaryAttachments);
}

llvm::Expected<std::vector<std::uint64_t>>
detail::projectSpatialLogicalNetRouteNodeArrivals(
    const FrozenSpatialPnrProblem &problem, PnrIndex logicalNet,
    const RouteTreeState &route, llvm::ArrayRef<PnrIndex> portAttachments,
    llvm::ArrayRef<PnrIndex> graphBoundaryAttachments) {
  SpatialLogicalNetPhysicalTiming timing;
  std::vector<std::uint64_t> arrivals;
  std::vector<std::pair<PnrIndex, std::uint64_t>> worklist;
  if (llvm::Error error = deriveRouteNodeArrivals(
          problem, logicalNet, route, portAttachments, graphBoundaryAttachments,
          timing, arrivals, worklist))
    return std::move(error);
  return arrivals;
}

llvm::Expected<std::uint64_t> detail::projectSpatialLogicalNetSourceArrival(
    const FrozenSpatialPnrProblem &problem, PnrIndex logicalNet,
    llvm::ArrayRef<PnrIndex> portAttachments,
    llvm::ArrayRef<PnrIndex> graphBoundaryAttachments) {
  SpatialLogicalNetPhysicalTiming timing;
  return deriveSourceArrival(problem, logicalNet, portAttachments,
                             graphBoundaryAttachments, timing);
}

llvm::Expected<detail::SpatialLogicalNetPhysicalTiming>
detail::projectSpatialLogicalNetPhysicalTiming(
    const FrozenSpatialPnrProblem &problem, PnrIndex logicalNet,
    const RouteTreeState &route, PnrIndex registerFifoTransfer,
    llvm::ArrayRef<PnrIndex> portAttachments,
    llvm::ArrayRef<PnrIndex> graphBoundaryAttachments,
    std::vector<std::uint64_t> *routeNodeArrivals,
    std::vector<std::pair<PnrIndex, std::uint64_t>> *routeNodeWorklist) {
  SpatialLogicalNetPhysicalTiming result;
  if (routeNodeArrivals)
    routeNodeArrivals->clear();
  auto criticality = logicalNetStructuralCriticality(problem, logicalNet);
  if (!criticality)
    return criticality.takeError();
  result.structuralCriticality = *criticality;
  const FrozenSpatialRoutingGraph &routing = problem.routing();
  const std::uint64_t required = routing.requiredCombinationalDelayQuanta();
  if (required == 0)
    return invalid("physical timing provider has a zero delay budget");

  if (registerFifoTransfer != getInvalidPnrIndex()) {
    if (!route.isUnrouted())
      return invalid("register-FIFO net also has an external route");
    const auto options = problem.localTransfers().options();
    if (registerFifoTransfer >= options.size())
      return invalid("register-FIFO selection is out of range");
    const FrozenSpatialRegisterFifoTransferOption &option =
        options[registerFifoTransfer];
    for (PnrIndex traversalOrdinal :
         {option.writeTraversal, option.readTraversal}) {
      auto reached = traversePhysicalTiming(routing, traversalOrdinal, 0,
                                            required, result);
      if (!reached)
        return reached.takeError();
      if (llvm::Error error = detail::observeSpatialPhysicalTimingEndpoint(
              *reached, required, result))
        return std::move(error);
    }
    return result;
  }

  if (route.isUnrouted())
    return result;
  std::vector<std::uint64_t> localArrivals;
  std::vector<std::pair<PnrIndex, std::uint64_t>> localWorklist;
  std::vector<std::uint64_t> &arrivalStorage =
      routeNodeArrivals ? *routeNodeArrivals : localArrivals;
  std::vector<std::pair<PnrIndex, std::uint64_t>> &worklistStorage =
      routeNodeWorklist ? *routeNodeWorklist : localWorklist;
  if (llvm::Error error = deriveRouteNodeArrivals(
          problem, logicalNet, route, portAttachments, graphBoundaryAttachments,
          result, arrivalStorage, worklistStorage))
    return std::move(error);
  const llvm::ArrayRef<std::uint64_t> arrivalValues = arrivalStorage;

  const FrozenSpatialLogicalNet &net =
      problem.transfers().logicalNets()[logicalNet];
  const auto sinkBindings = problem.transfers().logicalNetSinkBindings();
  for (PnrIndex sink = 0; sink < net.sinkCount; ++sink) {
    const auto sinkSlot = route.sinkNode(sink);
    if (!sinkSlot)
      continue;
    if (*sinkSlot >= arrivalValues.size())
      return invalid("timing sink RouteTree node is out of range");
    std::uint64_t arrival = arrivalValues[*sinkSlot];
    const FrozenSpatialTerminalBinding binding =
        sinkBindings[net.sinkOffset + sink];
    auto localTraversal = selectedLocalTraversal(
        problem, binding, portAttachments, graphBoundaryAttachments);
    if (!localTraversal)
      return localTraversal.takeError();
    if (*localTraversal) {
      auto reached = traversePhysicalTiming(routing, **localTraversal, arrival,
                                            required, result);
      if (!reached)
        return reached.takeError();
      arrival = *reached;
    }
    if (llvm::Error error = detail::observeSpatialPhysicalTimingEndpoint(
            arrival, required, result))
      return std::move(error);
  }
  return result;
}

llvm::Expected<detail::SpatialPhysicalTimingProjection>
detail::projectSpatialPhysicalTiming(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<const RouteTreeState *> routes,
    llvm::ArrayRef<PnrIndex> registerFifoTransfers,
    llvm::ArrayRef<PnrIndex> portAttachments,
    llvm::ArrayRef<PnrIndex> graphBoundaryAttachments,
    std::vector<std::uint64_t> *netWorstArrivals,
    std::vector<std::uint64_t> *netNegativeSlacks) {
  if (routes.size() != problem.transfers().logicalNets().size() ||
      registerFifoTransfers.size() != routes.size())
    return invalid("physical timing projection has the wrong net domain");
  if ((netWorstArrivals == nullptr) != (netNegativeSlacks == nullptr))
    return invalid(
        "physical timing per-net outputs must be requested together");
  if (netWorstArrivals) {
    netWorstArrivals->clear();
    netNegativeSlacks->clear();
    netWorstArrivals->reserve(routes.size());
    netNegativeSlacks->reserve(routes.size());
  }
  SpatialPhysicalTimingProjection result;
  for (PnrIndex logicalNet = 0; logicalNet < routes.size(); ++logicalNet) {
    if (!routes[logicalNet])
      return invalid("physical timing projection has a null RouteTree");
    auto timing = projectSpatialLogicalNetPhysicalTiming(
        problem, logicalNet, *routes[logicalNet],
        registerFifoTransfers[logicalNet], portAttachments,
        graphBoundaryAttachments);
    if (!timing)
      return timing.takeError();
    if (netWorstArrivals) {
      netWorstArrivals->push_back(timing->worstArrivalDelayQuanta);
      netNegativeSlacks->push_back(timing->totalNegativeSlackQuanta);
    }
    result.worstArrivalDelayQuanta = std::max(result.worstArrivalDelayQuanta,
                                              timing->worstArrivalDelayQuanta);
    auto total =
        add(result.totalNegativeSlackQuanta, timing->totalNegativeSlackQuanta,
            "Spatial total negative slack");
    if (!total)
      return total.takeError();
    result.totalNegativeSlackQuanta = *total;
  }
  return result;
}

llvm::Expected<detail::SpatialPhysicalTimingProjection>
detail::projectSpatialMappingPhysicalTiming(
    const ::loom::mapping::SpatialMappingView &mapping,
    const ::loom::fabric::FabricPhysicalTimingProfileView &profile) {
  if (mapping.fabricIdentity() != profile.fabricIdentity())
    return invalid("SpatialMapping and physical timing profile owners differ");
  const std::uint64_t required = profile.requiredCombinationalDelayQuanta();
  if (required == 0)
    return invalid("physical timing provider has a zero delay budget");

  struct TimingRecord final {
    std::uint64_t delay = 0;
    ::loom::fabric::FabricPhysicalTimingBoundaryKind boundary =
        ::loom::fabric::FabricPhysicalTimingBoundaryKind::Combinational;
  };
  std::map<std::vector<std::uint8_t>, TimingRecord> records;
  for (const auto &timing : profile.traversals())
    if (!records
             .emplace(::loom::fabric::canonicalFabricBytes(timing.traversal),
                      TimingRecord{timing.delayQuanta, timing.boundary})
             .second)
      return invalid("physical timing profile repeats a traversal");

  SpatialPhysicalTimingProjection result;
  SpatialLogicalNetPhysicalTiming routeTiming;
  const auto traverse =
      [&](const ::loom::fabric::FabricPhysicalTraversalRef &traversal,
          std::uint64_t arrival, SpatialLogicalNetPhysicalTiming &timing)
      -> llvm::Expected<std::uint64_t> {
    const auto found =
        records.find(::loom::fabric::canonicalFabricBytes(traversal));
    if (found == records.end())
      return invalid("SpatialMapping traversal has no timing record");
    return advanceSpatialPhysicalTiming(
        found->second.delay, found->second.boundary, arrival, required, timing);
  };
  const auto mergeRoute =
      [&](const SpatialLogicalNetPhysicalTiming &timing) -> llvm::Error {
    result.worstArrivalDelayQuanta = std::max(result.worstArrivalDelayQuanta,
                                              timing.worstArrivalDelayQuanta);
    auto total =
        add(result.totalNegativeSlackQuanta, timing.totalNegativeSlackQuanta,
            "persistent Mapping total negative slack");
    if (!total)
      return total.takeError();
    result.totalNegativeSlackQuanta = *total;
    return llvm::Error::success();
  };

  for (const auto &route : mapping.routeTrees()) {
    routeTiming = {};
    if (route.nodes.empty())
      return invalid("persistent route has no nodes");
    std::map<std::uint64_t, std::size_t> nodePositions;
    for (auto [position, node] : llvm::enumerate(route.nodes))
      if (!nodePositions.emplace(node.ordinal, position).second)
        return invalid("persistent route repeats a node ordinal");
    std::uint64_t rootArrival = 0;
    if (route.localTraversal) {
      auto reached = traverse(*route.localTraversal, 0, routeTiming);
      if (!reached)
        return reached.takeError();
      rootArrival = *reached;
    }
    std::vector<std::uint64_t> arrivals(route.nodes.size(), 0);
    std::vector<std::uint8_t> states(route.nodes.size(), 0);
    const auto deriveArrival =
        [&](auto &&self,
            std::size_t position) -> llvm::Expected<std::uint64_t> {
      if (position >= route.nodes.size())
        return invalid("persistent timing node is out of range");
      if (states[position] == 2)
        return arrivals[position];
      if (states[position] == 1)
        return invalid("persistent timing route contains a parent cycle");
      states[position] = 1;
      const auto &node = route.nodes[position];
      if (!node.parentOrdinal) {
        if (node.incomingTraversal)
          return invalid("route root has an incoming traversal");
        arrivals[position] = rootArrival;
      } else {
        const auto parent = nodePositions.find(*node.parentOrdinal);
        if (parent == nodePositions.end() || !node.incomingTraversal)
          return invalid("route node has an incomplete parent relation");
        auto parentArrival = self(self, parent->second);
        if (!parentArrival)
          return parentArrival.takeError();
        auto reached =
            traverse(*node.incomingTraversal, *parentArrival, routeTiming);
        if (!reached)
          return reached.takeError();
        arrivals[position] = *reached;
      }
      states[position] = 2;
      return arrivals[position];
    };
    for (std::size_t position = 0; position < route.nodes.size(); ++position) {
      auto arrival = deriveArrival(deriveArrival, position);
      if (!arrival)
        return arrival.takeError();
    }
    for (const auto &sink : route.sinks) {
      const auto node = nodePositions.find(sink.nodeOrdinal);
      if (node == nodePositions.end())
        return invalid("persistent timing sink names a foreign node");
      std::uint64_t arrival = arrivals[node->second];
      if (sink.localTraversal) {
        auto reached = traverse(*sink.localTraversal, arrival, routeTiming);
        if (!reached)
          return reached.takeError();
        arrival = *reached;
      }
      if (llvm::Error error = observeSpatialPhysicalTimingEndpoint(
              arrival, required, routeTiming))
        return std::move(error);
    }
    if (llvm::Error error = mergeRoute(routeTiming))
      return std::move(error);
  }

  for (const auto &transfer : mapping.registerFifoTransfers()) {
    routeTiming = {};
    for (const auto &traversal :
         {transfer.writeTraversal, transfer.readTraversal}) {
      auto reached = traverse(traversal, 0, routeTiming);
      if (!reached)
        return reached.takeError();
      if (llvm::Error error = observeSpatialPhysicalTimingEndpoint(
              *reached, required, routeTiming))
        return std::move(error);
    }
    if (llvm::Error error = mergeRoute(routeTiming))
      return std::move(error);
  }
  return result;
}
