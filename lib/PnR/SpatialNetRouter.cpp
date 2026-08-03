#include "PnR/SpatialNetRouter.h"

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

llvm::Error
SpatialNetRouterScratch::prepare(const FrozenSpatialPnrProblem &problem) {
  if (llvm::Error error =
          endpointSearch_.prepare(endpointRoutingGraphView(problem.routing())))
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
  preparedProblem_ = &problem;
  return llvm::Error::success();
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

llvm::Error SpatialNetRouterScratch::collectTargetFrontier(
    const SpatialCandidateState &candidate, PnrIndex logicalNet,
    PnrIndex sinkCount) {
  targetCandidates_.clear();
  targetEndpoints_.clear();
  targetPreferenceRanks_.clear();
  for (PnrIndex sink = 0; sink < sinkCount; ++sink) {
    if (!unresolvedSinks_[sink])
      continue;
    targetCandidates_.push_back(
        {candidate.logicalNetSinkEndpoint(logicalNet, sink), sink});
  }
  llvm::sort(targetCandidates_,
             [](const TargetCandidate &lhs, const TargetCandidate &rhs) {
               return std::tie(lhs.endpoint, lhs.sinkObligation) <
                      std::tie(rhs.endpoint, rhs.sinkObligation);
             });
  for (const TargetCandidate &target : targetCandidates_) {
    if (!targetEndpoints_.empty() && targetEndpoints_.back() == target.endpoint)
      continue;
    targetEndpoints_.push_back(target.endpoint);
    targetPreferenceRanks_.push_back(target.sinkObligation);
    targetObligationByEndpoint_[target.endpoint] = target.sinkObligation;
  }
  if (targetEndpoints_.empty())
    return netRouterError("unresolved sink count has no target endpoint");
  return llvm::Error::success();
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
  std::fill(prospectiveClaimBits_.begin(), prospectiveClaimBits_.end(), 0);
  if (llvm::Error error = costs.updateSelectedLogicalNetClaims({}))
    return std::move(error);
  if (llvm::Error error = move.ripUpWholeRoute(logicalNet))
    return std::move(error);

  const PnrIndex source = candidate.logicalNetSourceEndpoint(logicalNet);
  if (llvm::Error error = move.bindRouteSource(logicalNet, source))
    return std::move(error);
  for (PnrIndex sink = 0; sink < net.sinkCount; ++sink)
    if (llvm::Error error = move.bindRouteSink(
            logicalNet, sink,
            candidate.logicalNetSinkEndpoint(logicalNet, sink)))
      return std::move(error);

  RouteCost totalCost = 0;
  PnrIndex unresolvedCount = net.sinkCount;
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
    if (llvm::Error error =
            collectTargetFrontier(candidate, logicalNet, net.sinkCount))
      return std::move(error);
    auto result = endpointSearch_.search(
        {sourceEndpoints_, sourceReplicationGroups_, targetEndpoints_,
         targetPreferenceRanks_, costs.lowerBoundArcCosts(),
         costs.currentArcCosts(), candidate.logicalNetPayloadWidth(logicalNet),
         0, endpointExpansionLimit});
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
         retainedBytes(unresolvedSinks_) + retainedBytes(prospectiveClaimBits_);
}
