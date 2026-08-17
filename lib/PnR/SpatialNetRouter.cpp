#include "PnR/SpatialNetRouter.h"

#include "SpatialPhysicalTiming.h"
#include "SpatialProgressAnalysis.h"
#include "SpatialRouteConstraintModel.h"
#include "SpatialRouteTreePruning.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <system_error>
#include <tuple>
#include <utility>

using namespace loom::pnr;

class loom::pnr::detail::SpatialNetRouterPrivate final {
public:
  SpatialRouteConstraintScratch routeConstraints;
  SpatialRouteTreePruningScratch routeTreePruning;
};

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
    : private_(std::make_unique<detail::SpatialNetRouterPrivate>()) {}

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
  sourceTimingArrivalQuanta_.clear();
  sourceTimingArrivalQuanta_.reserve(endpointCount);
  targetCandidates_.clear();
  targetCandidates_.reserve(maximumSinkCount);
  targetEndpoints_.clear();
  targetEndpoints_.reserve(maximumSinkCount);
  targetPreferenceRanks_.clear();
  targetPreferenceRanks_.reserve(maximumSinkCount);
  targetRequiresTraversal_.clear();
  targetRequiresTraversal_.reserve(maximumSinkCount);
  targetTimingDelayQuanta_.clear();
  targetTimingDelayQuanta_.reserve(maximumSinkCount);
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
  arcTimingDelayQuanta_.clear();
  arcTimingRegisteredDestination_.clear();
  arcTimingDelayQuanta_.reserve(problem.routing().routingArcs().size());
  arcTimingRegisteredDestination_.reserve(
      problem.routing().routingArcs().size());
  for (const EndpointRoutingArc &arc : problem.routing().routingArcs()) {
    if (arc.traversal >= problem.routing().traversals().size())
      return netRouterError("routing arc timing traversal is out of range");
    const FrozenSpatialTraversal &traversal =
        problem.routing().traversals()[arc.traversal];
    arcTimingDelayQuanta_.push_back(traversal.physicalDelayQuanta);
    arcTimingRegisteredDestination_.push_back(
        traversal.physicalTimingBoundary ==
                ::loom::fabric::FabricPhysicalTimingBoundaryKind::
                    RegisteredDestination
            ? 1
            : 0);
  }
  routeNodeTimingArrivals_.clear();
  routeNodeTimingArrivals_.reserve(endpointCount);
  routeNodeTimingWorklist_.clear();
  routeNodeTimingWorklist_.reserve(endpointCount);
  endpointMarks_.assign(endpointCount, 0);
  subtreeWorklist_.clear();
  subtreeWorklist_.reserve(endpointCount);
  endpointMarkEpoch_ = 0;
  if (llvm::Error error = private_->routeConstraints.prepare(problem))
    return error;
  if (llvm::Error error = private_->routeTreePruning.prepare(problem))
    return error;
  preparedProblem_ = &problem;
  return llvm::Error::success();
}

llvm::Error SpatialNetRouterScratch::beginConstraintSweep(
    llvm::ArrayRef<PnrIndex> logicalNets) {
  return private_->routeConstraints.beginSweep(logicalNets);
}

llvm::Error SpatialNetRouterScratch::finishConstraintNet(PnrIndex logicalNet) {
  return private_->routeConstraints.finishNet(logicalNet);
}

llvm::Expected<detail::SpatialNegotiatedRoutePlan>
SpatialNetRouterScratch::planNegotiatedRoute(
    const SpatialCandidateState &candidate, const SpatialRouteCostState &costs,
    PnrIndex logicalNet) {
  return private_->routeTreePruning.project(candidate, costs, logicalNet);
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
  targetRequiresTraversal_.clear();
  targetTimingDelayQuanta_.clear();
  for (PnrIndex sink = 0; sink < sinkCount; ++sink) {
    if (!unresolvedSinks_[sink])
      continue;
    auto prerequisites =
        spatialSinkProgressDependencies(candidate.problem(), logicalNet, sink);
    if (!prerequisites)
      return prerequisites.takeError();
    bool hasLocalBoundary = false;
    if (!prerequisites->empty()) {
      const FrozenSpatialLogicalNet &net =
          candidate.problem().transfers().logicalNets()[logicalNet];
      auto localBoundary = spatialTerminalProvidesLocalProgressBoundary(
          candidate, candidate.problem()
                         .transfers()
                         .logicalNetSinkBindings()[net.sinkOffset + sink]);
      if (!localBoundary)
        return localBoundary.takeError();
      hasLocalBoundary = *localBoundary;
    }
    bool ready = true;
    for (const FrozenSpatialProgressPrerequisite &prerequisite :
         *prerequisites) {
      const auto *external =
          std::get_if<FrozenSpatialExternalSinkPrerequisite>(&prerequisite);
      if (!external)
        continue;
      if (external->sink >= sinkCount)
        return netRouterError(
            "external sink progress prerequisite is out of range");
      if (!hasLocalBoundary && unresolvedSinks_[external->sink]) {
        ready = false;
        break;
      }
    }
    if (!ready)
      continue;
    const bool requiresBufferedTraversal =
        !prerequisites->empty() && !hasLocalBoundary;
    targetCandidates_.push_back(
        {candidate.logicalNetSinkEndpoint(logicalNet, sink), sink,
         requiresBufferedTraversal});
  }
  if (targetCandidates_.empty())
    return netRouterError(
        "unresolved sink dependencies contain no routable frontier");
  llvm::sort(targetCandidates_,
             [](const TargetCandidate &lhs, const TargetCandidate &rhs) {
               return std::make_tuple(lhs.endpoint, !lhs.requiresTraversal,
                                      lhs.sinkObligation) <
                      std::make_tuple(rhs.endpoint, !rhs.requiresTraversal,
                                      rhs.sinkObligation);
             });
  for (const TargetCandidate &target : targetCandidates_) {
    if (!targetEndpoints_.empty() && targetEndpoints_.back() == target.endpoint)
      continue;
    targetEndpoints_.push_back(target.endpoint);
    targetPreferenceRanks_.push_back(target.sinkObligation);
    targetRequiresTraversal_.push_back(target.requiresTraversal);
    const FrozenSpatialLogicalNet &net =
        candidate.problem().transfers().logicalNets()[logicalNet];
    const FrozenSpatialTerminalBinding binding =
        candidate.problem()
            .transfers()
            .logicalNetSinkBindings()[net.sinkOffset + target.sinkObligation];
    auto localTraversal = detail::projectSelectedSpatialTerminalTraversal(
        candidate.problem(), binding, candidate.portAttachmentSelections(),
        candidate.graphBoundaryAttachmentSelections());
    if (!localTraversal)
      return localTraversal.takeError();
    std::uint64_t terminalDelay = 0;
    if (*localTraversal) {
      if (**localTraversal >= candidate.problem().routing().traversals().size())
        return netRouterError(
            "target-local physical timing traversal is out of range");
      terminalDelay = candidate.problem()
                          .routing()
                          .traversals()[**localTraversal]
                          .physicalDelayQuanta;
    }
    targetTimingDelayQuanta_.push_back(terminalDelay);
    targetObligationByEndpoint_[target.endpoint] = target.sinkObligation;
  }
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

llvm::Error
SpatialNetRouterScratch::updateCurrentTagUses(const RouteTreeState &tree,
                                              SpatialRouteCostState &costs) {
  if (llvm::Error error = detail::rebuildSpatialTagContinuityUnchecked(
          tree, tagContinuity_, tagContinuityScratch_))
    return error;
  return costs.updateSelectedLogicalNetTagUses(tree, tagContinuity_);
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

llvm::Expected<RouteCost> SpatialNetRouterScratch::routeSinkSet(
    SpatialMoveTransaction &move, const SpatialCandidateState &candidate,
    SpatialRouteCostState &costs, PnrIndex logicalNet,
    llvm::ArrayRef<PnrIndex> sinkObligations,
    std::uint64_t endpointExpansionLimit) {
  if (!preparedProblem_ || preparedProblem_ != &candidate.problem())
    return netRouterError("scratch is not prepared for the candidate freeze");
  if (logicalNet >= candidate.problem().transfers().logicalNets().size())
    return netRouterError("logical net is out of range");
  if (costs.selectedLogicalNet() != logicalNet)
    return netRouterError("route cost state does not select the logical net");
  const FrozenSpatialLogicalNet &net =
      candidate.problem().transfers().logicalNets()[logicalNet];
  if (!candidate.routeTree(logicalNet).isRouted() ||
      net.sinkCount > unresolvedSinks_.size())
    return netRouterError("sink-set routing requires a complete current route");
  if (sinkObligations.empty())
    return netRouterError("sink-set routing selected no sink obligation");

  std::fill(unresolvedSinks_.begin(), unresolvedSinks_.begin() + net.sinkCount,
            0);
  PnrIndex previous = getInvalidPnrIndex();
  for (PnrIndex sink : sinkObligations) {
    if (sink >= net.sinkCount ||
        (previous != getInvalidPnrIndex() && previous >= sink))
      return netRouterError(
          "sink-set routing domain is not canonical and unique");
    unresolvedSinks_[sink] = 1;
    previous = sink;
  }
  for (PnrIndex sink : sinkObligations)
    if (llvm::Error error = move.ripUpRouteSink(logicalNet, sink))
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
      private_->routeConstraints.eligibleTraversals(candidate, logicalNet);
  if (!eligibleTraversals)
    return eligibleTraversals.takeError();
  if (llvm::Error error = collectCurrentClaims(candidate.routeTree(logicalNet)))
    return std::move(error);
  if (llvm::Error error =
          costs.updateSelectedLogicalNetClaims(prospectiveClaimBits_))
    return std::move(error);
  if (llvm::Error error =
          updateCurrentTagUses(candidate.routeTree(logicalNet), costs))
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
      bool hasLocalBoundary = false;
      if (!prerequisites->empty()) {
        auto localBoundary = spatialTerminalProvidesLocalProgressBoundary(
            candidate,
            candidate.problem()
                .transfers()
                .logicalNetSinkBindings()[net.sinkOffset + sink]);
        if (!localBoundary)
          return localBoundary.takeError();
        hasLocalBoundary = *localBoundary;
      }
      bool progressSatisfied = true;
      for (const FrozenSpatialProgressPrerequisite &prerequisite :
           *prerequisites) {
        const auto *external =
            std::get_if<FrozenSpatialExternalSinkPrerequisite>(&prerequisite);
        if (external && external->sink >= net.sinkCount)
          return netRouterError(
              "external sink progress prerequisite is out of range");
        if (!hasLocalBoundary && external &&
            unresolvedSinks_[external->sink]) {
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
    auto timing = detail::projectSpatialLogicalNetPhysicalTiming(
        candidate.problem(), logicalNet, tree,
        candidate.registerFifoTransfer(logicalNet),
        candidate.portAttachmentSelections(),
        candidate.graphBoundaryAttachmentSelections(),
        &routeNodeTimingArrivals_, &routeNodeTimingWorklist_);
    if (!timing)
      return timing.takeError();
    sourceTimingArrivalQuanta_.clear();
    if (tree.isUnrouted()) {
      auto arrival = detail::projectSpatialLogicalNetSourceArrival(
          candidate.problem(), logicalNet, candidate.portAttachmentSelections(),
          candidate.graphBoundaryAttachmentSelections());
      if (!arrival)
        return arrival.takeError();
      sourceTimingArrivalQuanta_.assign(sourceEndpoints_.size(), *arrival);
    } else {
      sourceTimingArrivalQuanta_.reserve(sourceEndpoints_.size());
      for (PnrIndex endpoint : sourceEndpoints_) {
        const auto slot = tree.findNode(endpoint);
        if (!slot || *slot >= routeNodeTimingArrivals_.size())
          return netRouterError(
              "route branch point has no physical timing arrival");
        sourceTimingArrivalQuanta_.push_back(routeNodeTimingArrivals_[*slot]);
      }
    }
    if (llvm::Error error =
            collectTargetFrontier(candidate, logicalNet, net.sinkCount))
      return std::move(error);
    const bool requiresBufferedTraversal =
        llvm::is_contained(targetRequiresTraversal_, std::uint8_t{1});
    if (requiresBufferedTraversal &&
        llvm::all_of(bufferedTraversalBits_,
                     [](std::uint64_t word) { return word == 0; }))
      return unreachable(
          "a causal multicast branch requires buffered ingress, but the "
          "Fabric exposes no buffered FIFO traversal");
    const llvm::ArrayRef<std::uint64_t> requiredTraversals =
        requiresBufferedTraversal
            ? llvm::ArrayRef<std::uint64_t>(bufferedTraversalBits_)
            : llvm::ArrayRef<std::uint64_t>();
    EndpointRouteSearchRequest routeRequest;
    routeRequest.sourceEndpoints = sourceEndpoints_;
    routeRequest.sourceReplicationGroups = sourceReplicationGroups_;
    routeRequest.targetEndpoints = targetEndpoints_;
    routeRequest.targetPreferenceRanks = targetPreferenceRanks_;
    routeRequest.lowerBoundArcCosts = costs.lowerBoundArcCosts();
    routeRequest.currentArcCosts = costs.currentArcCosts();
    routeRequest.requiredPayloadWidthBits =
        candidate.logicalNetPayloadWidth(logicalNet);
    routeRequest.endpointExpansionLimit = endpointExpansionLimit;
    routeRequest.eligibleTraversalBits = *eligibleTraversals;
    routeRequest.lowerBoundCostRevision = costs.lowerBoundCostRevision();
    routeRequest.requiredTraversalBits = requiredTraversals;
    routeRequest.forbidSourceReentry = requiresBufferedTraversal;
    routeRequest.targetRequiresTraversal = targetRequiresTraversal_;
    routeRequest.physicalTimingEnabled = true;
    routeRequest.arcTimingDelayQuanta = arcTimingDelayQuanta_;
    routeRequest.arcTimingRegisteredDestination =
        arcTimingRegisteredDestination_;
    routeRequest.sourceTimingArrivalQuanta = sourceTimingArrivalQuanta_;
    routeRequest.targetTimingDelayQuanta = targetTimingDelayQuanta_;
    routeRequest.requiredTimingQuanta =
        routing.requiredCombinationalDelayQuanta();
    routeRequest.timingCriticality = timing->structuralCriticality;
    auto result = endpointSearch_.search(routeRequest);
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
    const auto selectedTarget =
        llvm::lower_bound(targetEndpoints_, result->target);
    if (selectedTarget == targetEndpoints_.end() ||
        *selectedTarget != result->target)
      return netRouterError("route search selected a foreign target endpoint");
    const std::size_t targetOrdinal = selectedTarget - targetEndpoints_.begin();
    const PnrIndex sink = targetObligationByEndpoint_[result->target];
    if (sink == getInvalidPnrIndex() || sink >= net.sinkCount ||
        !unresolvedSinks_[sink])
      return netRouterError("route search selected no unresolved obligation");
    const bool selectedRequiresTraversal =
        targetRequiresTraversal_[targetOrdinal] != 0;
    if (selectedRequiresTraversal) {
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
    std::uint64_t branchArrival = 0;
    if (tree.isUnrouted()) {
      const auto sourcePosition =
          llvm::lower_bound(sourceEndpoints_, attachment);
      if (sourcePosition == sourceEndpoints_.end() ||
          *sourcePosition != attachment)
        return netRouterError(
            "unrouted attachment has no physical timing arrival");
      branchArrival =
          sourceTimingArrivalQuanta_[sourcePosition - sourceEndpoints_.begin()];
    } else {
      const auto attachmentSlot = tree.findNode(attachment);
      if (!attachmentSlot || *attachmentSlot >= routeNodeTimingArrivals_.size())
        return netRouterError(
            "routed attachment has no physical timing arrival");
      branchArrival = routeNodeTimingArrivals_[*attachmentSlot];
    }
    const std::uint64_t initialExcess =
        branchArrival > routing.requiredCombinationalDelayQuanta()
            ? branchArrival - routing.requiredCombinationalDelayQuanta()
            : 0;
    auto initialPenalty = detail::physicalTimingDrivenNegativeSlackCost(
        initialExcess, routing.requiredCombinationalDelayQuanta(),
        timing->structuralCriticality);
    if (!initialPenalty)
      return initialPenalty.takeError();
    branchCost = *initialPenalty;
    for (PnrIndex arc : branch) {
      auto traversalCost = detail::physicalTimingDrivenTraversalCost(
          arcTimingDelayQuanta_[arc],
          routing.requiredCombinationalDelayQuanta(),
          timing->structuralCriticality);
      if (!traversalCost)
        return traversalCost.takeError();
      auto arcCost =
          accumulateRouteCost(costs.currentArcCosts()[arc], *traversalCost);
      if (!arcCost)
        return arcCost.takeError();
      auto next = accumulateRouteCost(branchCost, *arcCost);
      if (!next)
        return next.takeError();
      branchCost = *next;
      if (arcTimingDelayQuanta_[arc] >
          std::numeric_limits<std::uint64_t>::max() - branchArrival)
        return netRouterError("route branch physical arrival exceeds u64");
      const std::uint64_t reached = branchArrival + arcTimingDelayQuanta_[arc];
      const std::uint64_t oldExcess =
          branchArrival > routing.requiredCombinationalDelayQuanta()
              ? branchArrival - routing.requiredCombinationalDelayQuanta()
              : 0;
      const std::uint64_t newExcess =
          reached > routing.requiredCombinationalDelayQuanta()
              ? reached - routing.requiredCombinationalDelayQuanta()
              : 0;
      auto penalty = detail::physicalTimingDrivenNegativeSlackCost(
          newExcess - oldExcess, routing.requiredCombinationalDelayQuanta(),
          timing->structuralCriticality);
      if (!penalty)
        return penalty.takeError();
      next = accumulateRouteCost(branchCost, *penalty);
      if (!next)
        return next.takeError();
      branchCost = *next;
      branchArrival = arcTimingRegisteredDestination_[arc] ? 0 : reached;
    }
    const std::uint64_t terminalDelay = targetTimingDelayQuanta_[targetOrdinal];
    if (terminalDelay >
        std::numeric_limits<std::uint64_t>::max() - branchArrival)
      return netRouterError("route target physical arrival exceeds u64");
    const std::uint64_t terminalArrival = branchArrival + terminalDelay;
    const std::uint64_t oldTerminalExcess =
        branchArrival > routing.requiredCombinationalDelayQuanta()
            ? branchArrival - routing.requiredCombinationalDelayQuanta()
            : 0;
    const std::uint64_t terminalExcess =
        terminalArrival > routing.requiredCombinationalDelayQuanta()
            ? terminalArrival - routing.requiredCombinationalDelayQuanta()
            : 0;
    auto terminalPenalty = detail::physicalTimingDrivenNegativeSlackCost(
        terminalExcess - oldTerminalExcess,
        routing.requiredCombinationalDelayQuanta(),
        timing->structuralCriticality);
    if (!terminalPenalty)
      return terminalPenalty.takeError();
    auto next = accumulateRouteCost(branchCost, *terminalPenalty);
    if (!next)
      return next.takeError();
    branchCost = *next;
    if (pathBegin == 0 && branchCost != result->cost)
      return netRouterError(
          "route branch timing cost disagrees with endpoint search");
    auto nextTotal = accumulateRouteCost(totalCost, branchCost);
    if (!nextTotal)
      return nextTotal.takeError();
    if (llvm::Error error =
            move.attachRoutePath(logicalNet, attachment, branch, sink))
      return std::move(error);
    if (llvm::Error error = addPathClaims(routing, branch))
      return std::move(error);
    if (llvm::Error error =
            costs.updateSelectedLogicalNetClaims(prospectiveClaimBits_))
      return std::move(error);
    if (llvm::Error error =
            updateCurrentTagUses(candidate.routeTree(logicalNet), costs))
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
         retainedBytes(sourceTimingArrivalQuanta_) +
         retainedBytes(targetCandidates_) + retainedBytes(targetEndpoints_) +
         retainedBytes(targetPreferenceRanks_) +
         retainedBytes(targetRequiresTraversal_) +
         retainedBytes(targetTimingDelayQuanta_) +
         retainedBytes(targetObligationByEndpoint_) +
         retainedBytes(unresolvedSinks_) +
         retainedBytes(prospectiveClaimBits_) +
         retainedBytes(bufferedTraversalBits_) +
         retainedBytes(arcTimingDelayQuanta_) +
         retainedBytes(arcTimingRegisteredDestination_) +
         retainedBytes(routeNodeTimingArrivals_) +
         retainedBytes(routeNodeTimingWorklist_) +
         retainedBytes(endpointMarks_) + retainedBytes(subtreeWorklist_) +
         tagContinuity_.retainedStorageBytes() +
         tagContinuityScratch_.retainedStorageBytes() +
         private_->routeConstraints.retainedStorageBytes() +
         private_->routeTreePruning.retainedStorageBytes();
}
