#include "PnR/SpatialNetRouter.h"

#include "SpatialProgressAnalysis.h"
#include "SpatialRouteConstraintModel.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <system_error>
#include <tuple>
#include <utility>

using namespace loom::pnr;

namespace {

llvm::Error netRouterError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("invalid Spatial whole-net route: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

llvm::Error unreachable(const llvm::Twine &message) {
  return llvm::make_error<EndpointRouteSearchFailure>(
      EndpointRouteSearchFailureKind::Unreachable,
      ("endpoint route search: " + message).str());
}

template <typename T> std::size_t retainedBytes(const std::vector<T> &values) {
  return values.capacity() * sizeof(T);
}

} // namespace

SpatialNetRouterScratch::SpatialNetRouterScratch()
    : routeConstraints_(
          std::make_unique<detail::SpatialRouteConstraintScratch>()) {}

SpatialNetRouterScratch::~SpatialNetRouterScratch() = default;

llvm::Error
SpatialNetRouterScratch::prepare(const FrozenSpatialPnrProblem &problem) {
  if (llvm::Error error = endpointSearch_.prepare(
          endpointRoutingGraphView(problem.routing().topology())))
    return error;

  const std::size_t endpointCount = problem.routing().routingEndpoints().size();
  std::size_t maximumSinkCount = 0;
  for (const FrozenSpatialLogicalNet &net : problem.transfers().logicalNets())
    maximumSinkCount =
        std::max(maximumSinkCount, static_cast<std::size_t>(net.sinkCount));

  sourceCandidates_.clear();
  sourceCandidates_.reserve(endpointCount);
  sourceEndpoints_.clear();
  sourceEndpoints_.reserve(endpointCount);
  sourceReplicationGroups_.clear();
  sourceReplicationGroups_.reserve(endpointCount);
  targetCandidates_.clear();
  targetCandidates_.reserve(maximumSinkCount);
  targetEndpoints_.clear();
  targetEndpoints_.reserve(maximumSinkCount);
  targetPreferenceRanks_.clear();
  targetPreferenceRanks_.reserve(maximumSinkCount);
  targetObligationByEndpoint_.assign(endpointCount, getInvalidPnrIndex());
  unresolvedSinks_.assign(maximumSinkCount, 0);
  prospectiveClaimBits_.assign(
      (problem.routing().routeClaims().size() + 63) / 64, 0);
  bufferedTraversalBits_.assign(
      (problem.routing().traversals().size() + 63) / 64, 0);
  for (auto [traversal, record] :
       llvm::enumerate(problem.routing().traversals())) {
    const auto *fifo = std::get_if<::loom::fabric::FabricFifoTraversalPayload>(
        &record.reference.payload);
    if (fifo && fifo->mode == ::loom::fabric::FabricFifoTraversalMode::Buffered)
      bufferedTraversalBits_[traversal / 64] |= std::uint64_t{1}
                                                << (traversal % 64);
  }
  endpointMarks_.assign(endpointCount, 0);
  subtreeWorklist_.clear();
  subtreeWorklist_.reserve(endpointCount);
  endpointMarkEpoch_ = 0;
  if (llvm::Error error = routeConstraints_->prepare(problem))
    return error;
  preparedProblem_ = &problem;
  return llvm::Error::success();
}

llvm::Error SpatialNetRouterScratch::beginConstraintSweep(
    llvm::ArrayRef<PnrIndex> logicalNets) {
  return routeConstraints_->beginSweep(logicalNets);
}

llvm::Error SpatialNetRouterScratch::finishConstraintNet(PnrIndex logicalNet) {
  return routeConstraints_->finishNet(logicalNet);
}

llvm::Error
SpatialNetRouterScratch::collectSourceFrontier(const RouteTreeState &tree,
                                               PnrIndex unroutedSource) {
  sourceCandidates_.clear();
  sourceEndpoints_.clear();
  sourceReplicationGroups_.clear();
  const FrozenSpatialRoutingGraph &routing = preparedProblem_->routing();

  if (tree.isUnrouted()) {
    sourceCandidates_.push_back({unroutedSource, getInvalidPnrIndex()});
  } else {
    for (const RouteTreeNode &node : tree.nodeStorage()) {
      if (!node.isActive())
        continue;
      PnrIndex replicationGroup = getInvalidPnrIndex();
      if (node.firstChild != getInvalidPnrIndex()) {
        if (node.firstChild >= tree.nodeStorage().size())
          return netRouterError("RouteTree child is out of range");
        const PnrIndex parentArc = tree.node(node.firstChild).parentArc;
        if (parentArc >= routing.routingArcs().size())
          return netRouterError("RouteTree parent arc is out of range");
        const PnrIndex traversal = routing.routingArcs()[parentArc].traversal;
        if (traversal >= routing.traversalReplicationGroups().size())
          return netRouterError("RouteTree traversal is out of range");
        replicationGroup = routing.traversalReplicationGroups()[traversal];
        if (replicationGroup == getInvalidPnrIndex())
          continue;
      }
      sourceCandidates_.push_back({node.endpoint, replicationGroup});
    }
  }

  llvm::sort(sourceCandidates_,
             [](const SourceCandidate &lhs, const SourceCandidate &rhs) {
               return lhs.endpoint < rhs.endpoint;
             });
  for (const SourceCandidate &source : sourceCandidates_) {
    sourceEndpoints_.push_back(source.endpoint);
    sourceReplicationGroups_.push_back(source.replicationGroup);
  }
  if (sourceEndpoints_.empty())
    return unreachable("the current RouteTree has no legal branch point");
  return llvm::Error::success();
}

llvm::Expected<bool> SpatialNetRouterScratch::collectTargetFrontier(
    const SpatialCandidateState &candidate, PnrIndex logicalNet,
    PnrIndex sinkCount) {
  targetCandidates_.clear();
  targetEndpoints_.clear();
  targetPreferenceRanks_.clear();
  std::optional<PnrIndex> selectedSink;
  bool requiresBufferedTraversal = false;
  for (PnrIndex sink = 0; sink < sinkCount; ++sink) {
    if (!unresolvedSinks_[sink])
      continue;
    auto prerequisites =
        spatialSinkProgressDependencies(candidate.problem(), logicalNet, sink);
    if (!prerequisites)
      return prerequisites.takeError();
    bool ready = true;
    for (PnrIndex prerequisite : *prerequisites) {
      if (prerequisite >= sinkCount)
        return netRouterError("sink progress prerequisite is out of range");
      if (unresolvedSinks_[prerequisite]) {
        ready = false;
        break;
      }
    }
    if (!ready)
      continue;
    selectedSink = sink;
    requiresBufferedTraversal = !prerequisites->empty();
    if (requiresBufferedTraversal) {
      const FrozenSpatialLogicalNet &net =
          candidate.problem().transfers().logicalNets()[logicalNet];
      auto localBoundary = spatialTerminalProvidesLocalProgressBoundary(
          candidate,
          candidate.problem().transfers().logicalNetSinkBindings()[
              net.sinkOffset + sink]);
      if (!localBoundary)
        return localBoundary.takeError();
      requiresBufferedTraversal = !*localBoundary;
    }
    break;
  }
  if (!selectedSink)
    return netRouterError(
        "unresolved sink dependencies contain no routable frontier");
  const PnrIndex endpoint =
      candidate.logicalNetSinkEndpoint(logicalNet, *selectedSink);
  targetCandidates_.push_back({endpoint, *selectedSink});
  targetEndpoints_.push_back(endpoint);
  targetPreferenceRanks_.push_back(*selectedSink);
  targetObligationByEndpoint_[endpoint] = *selectedSink;
  return requiresBufferedTraversal;
}

llvm::Error
SpatialNetRouterScratch::addPathClaims(const FrozenSpatialRoutingGraph &routing,
                                       llvm::ArrayRef<PnrIndex> forwardArcs) {
  for (PnrIndex arc : forwardArcs) {
    if (arc >= routing.routingArcs().size())
      return netRouterError("selected path arc is out of range");
    const PnrIndex traversal = routing.routingArcs()[arc].traversal;
    if (traversal >= routing.traversals().size())
      return netRouterError("selected path traversal is out of range");
    const FrozenSpatialTraversal &record = routing.traversals()[traversal];
    for (PnrIndex claim : routing.traversalClaimKeys().slice(
             record.routeClaimOffset, record.routeClaimCount)) {
      if (claim >= routing.routeClaims().size())
        return netRouterError("selected path claim is out of range");
      prospectiveClaimBits_[claim / 64] |= std::uint64_t{1} << (claim % 64);
    }
  }
  return llvm::Error::success();
}

llvm::Error
SpatialNetRouterScratch::collectCurrentClaims(const RouteTreeState &tree) {
  const FrozenSpatialRoutingGraph &routing = tree.routingGraph();
  std::fill(prospectiveClaimBits_.begin(), prospectiveClaimBits_.end(), 0);
  for (const RouteTreeNode &node : tree.nodeStorage()) {
    if (!node.isActive() || node.parentArc == getInvalidPnrIndex())
      continue;
    if (node.parentArc >= routing.routingArcs().size())
      return netRouterError("RouteTree parent arc is out of range");
    const PnrIndex traversal = routing.routingArcs()[node.parentArc].traversal;
    if (traversal >= routing.traversals().size())
      return netRouterError("RouteTree traversal is out of range");
    const FrozenSpatialTraversal &record = routing.traversals()[traversal];
    for (PnrIndex claim : routing.traversalClaimKeys().slice(
             record.routeClaimOffset, record.routeClaimCount)) {
      if (claim >= routing.routeClaims().size())
        return netRouterError("RouteTree claim is out of range");
      prospectiveClaimBits_[claim / 64] |= std::uint64_t{1} << (claim % 64);
    }
  }
  return llvm::Error::success();
}

void SpatialNetRouterScratch::beginEndpointMarks() {
  ++endpointMarkEpoch_;
  if (endpointMarkEpoch_ == 0) {
    std::fill(endpointMarks_.begin(), endpointMarks_.end(), 0);
    endpointMarkEpoch_ = 1;
  }
  subtreeWorklist_.clear();
}

llvm::Expected<RouteCost> SpatialNetRouterScratch::routeWholeNet(
    SpatialMoveTransaction &move, const SpatialCandidateState &candidate,
    SpatialRouteCostState &costs, PnrIndex logicalNet,
    std::uint64_t endpointExpansionLimit) {
  if (!preparedProblem_ || preparedProblem_ != &candidate.problem())
    return netRouterError("scratch is not prepared for the candidate freeze");
  if (logicalNet >= candidate.problem().transfers().logicalNets().size())
    return netRouterError("logical net is out of range");
  if (costs.selectedLogicalNet() != logicalNet)
    return netRouterError("route cost state does not select the logical net");
  if (endpointExpansionLimit == 0)
    return netRouterError("endpoint expansion limit must be positive");

  const FrozenSpatialLogicalNet &net =
      candidate.problem().transfers().logicalNets()[logicalNet];
  if (net.sinkCount == 0 || net.sinkCount > unresolvedSinks_.size())
    return netRouterError("logical net has no prepared sink domain");

  std::fill(unresolvedSinks_.begin(), unresolvedSinks_.begin() + net.sinkCount,
            1);
  if (llvm::Error error = move.ripUpWholeRoute(logicalNet))
    return std::move(error);

  const PnrIndex source = candidate.logicalNetSourceEndpoint(logicalNet);
  if (llvm::Error error = move.bindRouteSource(logicalNet, source))
    return std::move(error);
  return routeSelectedSinks(move, candidate, costs, logicalNet,
                            endpointExpansionLimit);
}

llvm::Expected<RouteCost> SpatialNetRouterScratch::routeSingleSink(
    SpatialMoveTransaction &move, const SpatialCandidateState &candidate,
    SpatialRouteCostState &costs, PnrIndex logicalNet, PnrIndex sinkObligation,
    std::uint64_t endpointExpansionLimit) {
  if (!preparedProblem_ || preparedProblem_ != &candidate.problem())
    return netRouterError("scratch is not prepared for the candidate freeze");
  if (logicalNet >= candidate.problem().transfers().logicalNets().size())
    return netRouterError("logical net is out of range");
  if (costs.selectedLogicalNet() != logicalNet)
    return netRouterError("route cost state does not select the logical net");
  const FrozenSpatialLogicalNet &net =
      candidate.problem().transfers().logicalNets()[logicalNet];
  if (sinkObligation >= net.sinkCount ||
      net.sinkCount > unresolvedSinks_.size())
    return netRouterError("sink obligation is out of range");
  if (!candidate.routeTree(logicalNet).isRouted())
    return netRouterError("SingleSink requires a complete current route");

  std::fill(unresolvedSinks_.begin(), unresolvedSinks_.begin() + net.sinkCount,
            0);
  unresolvedSinks_[sinkObligation] = 1;
  if (llvm::Error error = move.ripUpRouteSink(logicalNet, sinkObligation))
    return std::move(error);
  return routeSelectedSinks(move, candidate, costs, logicalNet,
                            endpointExpansionLimit);
}

llvm::Expected<RouteCost> SpatialNetRouterScratch::routeRootedSubtree(
    SpatialMoveTransaction &move, const SpatialCandidateState &candidate,
    SpatialRouteCostState &costs, PnrIndex logicalNet, PnrIndex rootEndpoint,
    std::uint64_t endpointExpansionLimit) {
  if (!preparedProblem_ || preparedProblem_ != &candidate.problem())
    return netRouterError("scratch is not prepared for the candidate freeze");
  if (logicalNet >= candidate.problem().transfers().logicalNets().size())
    return netRouterError("logical net is out of range");
  if (costs.selectedLogicalNet() != logicalNet)
    return netRouterError("route cost state does not select the logical net");
  const FrozenSpatialLogicalNet &net =
      candidate.problem().transfers().logicalNets()[logicalNet];
  const RouteTreeState &tree = candidate.routeTree(logicalNet);
  if (!tree.isRouted() || net.sinkCount > unresolvedSinks_.size())
    return netRouterError("RootedSubtree requires a complete current route");
  const auto source = tree.sourceEndpoint();
  const auto rootSlot = tree.findNode(rootEndpoint);
  if (!source || !rootSlot || rootEndpoint == *source)
    return netRouterError("subtree root is absent or names the source root");

  beginEndpointMarks();
  subtreeWorklist_.push_back(*rootSlot);
  for (std::size_t cursor = 0; cursor < subtreeWorklist_.size(); ++cursor) {
    const PnrIndex slot = subtreeWorklist_[cursor];
    if (slot >= tree.nodeStorage().size() || !tree.node(slot).isActive())
      return netRouterError("subtree contains an inactive RouteTree node");
    const RouteTreeNode &node = tree.node(slot);
    endpointMarks_[node.endpoint] = endpointMarkEpoch_;
    for (PnrIndex child = node.firstChild; child != getInvalidPnrIndex();
         child = tree.node(child).nextSibling)
      subtreeWorklist_.push_back(child);
  }

  std::fill(unresolvedSinks_.begin(), unresolvedSinks_.begin() + net.sinkCount,
            0);
  PnrIndex selectedCount = 0;
  for (PnrIndex sink = 0; sink < net.sinkCount; ++sink) {
    const PnrIndex endpoint =
        candidate.logicalNetSinkEndpoint(logicalNet, sink);
    if (endpointMarks_[endpoint] != endpointMarkEpoch_)
      continue;
    unresolvedSinks_[sink] = 1;
    ++selectedCount;
  }
  if (selectedCount == 0)
    return netRouterError("subtree contains no sink obligation");
  if (llvm::Error error = move.ripUpRouteSubtree(logicalNet, rootEndpoint))
    return std::move(error);
  return routeSelectedSinks(move, candidate, costs, logicalNet,
                            endpointExpansionLimit);
}

llvm::Expected<RouteCost> SpatialNetRouterScratch::routeSelectedSinks(
    SpatialMoveTransaction &move, const SpatialCandidateState &candidate,
    SpatialRouteCostState &costs, PnrIndex logicalNet,
    std::uint64_t endpointExpansionLimit) {
  if (endpointExpansionLimit == 0)
    return netRouterError("endpoint expansion limit must be positive");
  const FrozenSpatialLogicalNet &net =
      candidate.problem().transfers().logicalNets()[logicalNet];
  auto eligibleTraversals =
      routeConstraints_->eligibleTraversals(candidate, logicalNet);
  if (!eligibleTraversals)
    return eligibleTraversals.takeError();
  if (llvm::Error error = collectCurrentClaims(candidate.routeTree(logicalNet)))
    return std::move(error);
  if (llvm::Error error =
          costs.updateSelectedLogicalNetClaims(prospectiveClaimBits_))
    return std::move(error);

  PnrIndex unresolvedCount = 0;
  for (PnrIndex sink = 0; sink < net.sinkCount; ++sink)
    if (unresolvedSinks_[sink]) {
      ++unresolvedCount;
      if (llvm::Error error = move.bindRouteSink(
              logicalNet, sink,
              candidate.logicalNetSinkEndpoint(logicalNet, sink)))
        return std::move(error);
    }
  if (unresolvedCount == 0)
    return netRouterError("local route selected no sink obligation");

  RouteCost totalCost = 0;
  const PnrIndex source = candidate.logicalNetSourceEndpoint(logicalNet);
  const FrozenSpatialRoutingGraph &routing = candidate.problem().routing();
  while (unresolvedCount != 0) {
    const RouteTreeState &tree = candidate.routeTree(logicalNet);
    for (PnrIndex sink = 0; sink < net.sinkCount; ++sink) {
      if (!unresolvedSinks_[sink])
        continue;
      const PnrIndex target =
          candidate.logicalNetSinkEndpoint(logicalNet, sink);
      if (target != source && !tree.findNode(target))
        continue;
      auto prerequisites = spatialSinkProgressDependencies(candidate.problem(),
                                                           logicalNet, sink);
      if (!prerequisites)
        return prerequisites.takeError();
      bool progressSatisfied = true;
      for (PnrIndex prerequisite : *prerequisites) {
        if (prerequisite >= net.sinkCount)
          return netRouterError("sink progress prerequisite is out of range");
        if (unresolvedSinks_[prerequisite]) {
          progressSatisfied = false;
          break;
        }
        auto satisfied = spatialRouteProgressDependencySatisfied(
            candidate, logicalNet, prerequisite, sink);
        if (!satisfied)
          return satisfied.takeError();
        if (!*satisfied) {
          progressSatisfied = false;
          break;
        }
      }
      if (!progressSatisfied)
        continue;
      if (llvm::Error error =
              move.attachRoutePath(logicalNet, target, {}, sink))
        return std::move(error);
      unresolvedSinks_[sink] = 0;
      --unresolvedCount;
    }
    if (unresolvedCount == 0)
      break;

    if (llvm::Error error = collectSourceFrontier(tree, source))
      return std::move(error);
    auto requiresBufferedTraversal =
        collectTargetFrontier(candidate, logicalNet, net.sinkCount);
    if (!requiresBufferedTraversal)
      return requiresBufferedTraversal.takeError();
    if (*requiresBufferedTraversal &&
        llvm::all_of(bufferedTraversalBits_,
                     [](std::uint64_t word) { return word == 0; }))
      return unreachable(
          "a causal multicast branch requires buffered ingress, but the "
          "Fabric exposes no buffered FIFO traversal");
    const llvm::ArrayRef<std::uint64_t> requiredTraversals =
        *requiresBufferedTraversal
            ? llvm::ArrayRef<std::uint64_t>(bufferedTraversalBits_)
            : llvm::ArrayRef<std::uint64_t>();
    auto result = endpointSearch_.search(
        {sourceEndpoints_, sourceReplicationGroups_, targetEndpoints_,
         targetPreferenceRanks_, costs.lowerBoundArcCosts(),
         costs.currentArcCosts(), candidate.logicalNetPayloadWidth(logicalNet),
         0, endpointExpansionLimit, *eligibleTraversals,
         costs.lowerBoundCostRevision(), requiredTraversals,
         *requiresBufferedTraversal});
    if (!result)
      return result.takeError();

    PnrIndex attachment = result->source;
    std::size_t pathBegin = 0;
    for (auto [index, arc] : llvm::enumerate(result->forwardArcs)) {
      const PnrIndex target = routing.routingArcs()[arc].target;
      if (tree.findNode(target)) {
        attachment = target;
        pathBegin = index + 1;
      }
    }
    const llvm::ArrayRef<PnrIndex> branch =
        result->forwardArcs.drop_front(pathBegin);
    if (*requiresBufferedTraversal) {
      bool branchBuffered = false;
      for (PnrIndex arc : branch) {
        if (arc >= routing.routingArcs().size())
          return netRouterError("selected branch arc is out of range");
        const PnrIndex traversal = routing.routingArcs()[arc].traversal;
        branchBuffered |= traversal / 64 < bufferedTraversalBits_.size() &&
                          (bufferedTraversalBits_[traversal / 64] &
                           (std::uint64_t{1} << (traversal % 64))) != 0;
      }
      if (!branchBuffered)
        return netRouterError(
            "causal multicast branch lost its buffered traversal after "
            "route-tree attachment");
    }
    RouteCost branchCost = 0;
    for (PnrIndex arc : branch) {
      auto next = accumulateRouteCost(branchCost, costs.currentArcCosts()[arc]);
      if (!next)
        return next.takeError();
      branchCost = *next;
    }
    auto nextTotal = accumulateRouteCost(totalCost, branchCost);
    if (!nextTotal)
      return nextTotal.takeError();

    const PnrIndex sink = targetObligationByEndpoint_[result->target];
    if (sink == getInvalidPnrIndex() || sink >= net.sinkCount ||
        !unresolvedSinks_[sink])
      return netRouterError("route search selected no unresolved obligation");
    if (llvm::Error error =
            move.attachRoutePath(logicalNet, attachment, branch, sink))
      return std::move(error);
    if (llvm::Error error = addPathClaims(routing, branch))
      return std::move(error);
    if (llvm::Error error =
            costs.updateSelectedLogicalNetClaims(prospectiveClaimBits_))
      return std::move(error);
    unresolvedSinks_[sink] = 0;
    --unresolvedCount;
    totalCost = *nextTotal;
  }
  return totalCost;
}

std::size_t SpatialNetRouterScratch::retainedStorageBytes() const {
  return endpointSearch_.retainedStorageBytes() +
         retainedBytes(sourceCandidates_) + retainedBytes(sourceEndpoints_) +
         retainedBytes(sourceReplicationGroups_) +
         retainedBytes(targetCandidates_) + retainedBytes(targetEndpoints_) +
         retainedBytes(targetPreferenceRanks_) +
         retainedBytes(targetObligationByEndpoint_) +
         retainedBytes(unresolvedSinks_) +
         retainedBytes(prospectiveClaimBits_) +
         retainedBytes(bufferedTraversalBits_) + retainedBytes(endpointMarks_) +
         retainedBytes(subtreeWorklist_) +
         routeConstraints_->retainedStorageBytes();
}
