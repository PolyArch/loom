#include "SpatialRouteTreePruning.h"

#include "PnR/RouteTreeState.h"
#include "PnR/SpatialCandidateState.h"
#include "PnR/SpatialRouteCostState.h"
#include "SpatialProgressAnalysis.h"
#include "SpatialRouteConstraintModel.h"
#include "SpatialTagConstraintModel.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <system_error>

using namespace loom::pnr;
using namespace loom::pnr::detail;

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "spatial_route_tree_pruning_invalid: " + message);
}

template <typename T> std::size_t retainedBytes(const std::vector<T> &values) {
  return values.capacity() * sizeof(T);
}

} // namespace

llvm::Error SpatialRouteTreePruningScratch::prepare(
    const FrozenSpatialPnrProblem &problem) {
  const std::size_t traversalCount = problem.routing().traversals().size();
  const std::size_t endpointCount = problem.routing().routingEndpoints().size();
  std::size_t maximumSinkCount = 0;
  for (const FrozenSpatialLogicalNet &net : problem.transfers().logicalNets())
    maximumSinkCount =
        std::max(maximumSinkCount, static_cast<std::size_t>(net.sinkCount));

  traversalEpochs_.assign(traversalCount, 0);
  traversalOveruse_.assign(traversalCount, 0);
  nodeEpochs_.assign(endpointCount, 0);
  nodeAffected_.assign(endpointCount, 0);
  sinkAffected_.assign(maximumSinkCount, 0);
  selectedSinks_.clear();
  selectedSinks_.reserve(maximumSinkCount);
  pathSlots_.clear();
  pathSlots_.reserve(endpointCount);
  projectionEpoch_ = 0;
  preparedProblem_ = &problem;
  return llvm::Error::success();
}

bool SpatialRouteTreePruningScratch::traversalOverused(
    const SpatialCandidateState &candidate, const SpatialRouteCostState &costs,
    PnrIndex traversal) {
  if (traversalEpochs_[traversal] == projectionEpoch_)
    return traversalOveruse_[traversal] != 0;
  traversalEpochs_[traversal] = projectionEpoch_;
  traversalOveruse_[traversal] = 0;

  const FrozenSpatialRoutingGraph &routing = candidate.problem().routing();
  const FrozenSpatialTraversal &record = routing.traversals()[traversal];
  for (PnrIndex claim : routing.traversalClaimKeys().slice(
           record.routeClaimOffset, record.routeClaimCount)) {
    const FrozenSpatialRouteClaim &claimRecord = routing.routeClaims()[claim];
    if (claimRecord.amount == 0)
      continue;
    const PnrIndex capacity = claimRecord.capacityDimension;
    if (costs.workingCapacityUsageRaw(capacity) >
        candidate.problem()
            .resources()
            .capacityDimensions()[capacity]
            .capacity) {
      traversalOveruse_[traversal] = 1;
      break;
    }
  }
  if (!traversalOveruse_[traversal])
    for (PnrIndex arc : routing.traversalArcs().slice(
             routing.traversalArcOffsets()[traversal],
             routing.traversalArcOffsets()[traversal + 1] -
                 routing.traversalArcOffsets()[traversal]))
      if (costs.arcHasTagPressure(arc)) {
        traversalOveruse_[traversal] = 1;
        break;
      }
  return traversalOveruse_[traversal] != 0;
}

llvm::Expected<SpatialNegotiatedRoutePlan>
SpatialRouteTreePruningScratch::project(const SpatialCandidateState &candidate,
                                        const SpatialRouteCostState &costs,
                                        PnrIndex logicalNet) {
  if (!preparedProblem_ || preparedProblem_ != &candidate.problem())
    return invalid("scratch is not prepared for the candidate freeze");
  if (!costs.isBoundTo(candidate))
    return invalid("route costs are bound to another candidate");
  if (costs.selectedLogicalNet())
    return invalid("route costs already select a logical net");
  if (logicalNet >= candidate.problem().transfers().logicalNets().size())
    return invalid("logical net is out of range");

  const FrozenSpatialLogicalNet &net =
      candidate.problem().transfers().logicalNets()[logicalNet];
  const RouteTreeState &tree = candidate.routeTree(logicalNet);
  if (tree.isUnrouted() ||
      (!costs.hasCapacityOveruse() && !costs.hasTagPressureViolation()) ||
      !candidate.problem()
           .routeConstraints()
           .netRelations(logicalNet)
           .empty() ||
      candidate.problem().tagConstraints().netHasRelations(logicalNet) ||
      costs.logicalNetTagUnassignedCount(logicalNet) != 0)
    return SpatialNegotiatedRoutePlan{SpatialNegotiatedRouteScope::WholeNet,
                                      {}};
  if (tree.sourceEndpoint() != candidate.logicalNetSourceEndpoint(logicalNet))
    return SpatialNegotiatedRoutePlan{SpatialNegotiatedRouteScope::WholeNet,
                                      {}};
  for (PnrIndex sink = 0; sink < net.sinkCount; ++sink)
    if (tree.sinkEndpoint(sink) !=
        candidate.logicalNetSinkEndpoint(logicalNet, sink))
      return SpatialNegotiatedRoutePlan{SpatialNegotiatedRouteScope::WholeNet,
                                        {}};

  ++projectionEpoch_;
  if (projectionEpoch_ == 0) {
    std::fill(traversalEpochs_.begin(), traversalEpochs_.end(), 0);
    std::fill(nodeEpochs_.begin(), nodeEpochs_.end(), 0);
    projectionEpoch_ = 1;
  }
  std::fill(sinkAffected_.begin(), sinkAffected_.begin() + net.sinkCount, 0);

  const FrozenSpatialRoutingGraph &routing = candidate.problem().routing();
  const auto arcs = routing.routingArcs();
  for (PnrIndex sink = 0; sink < net.sinkCount; ++sink) {
    const auto sinkSlot =
        tree.findNode(candidate.logicalNetSinkEndpoint(logicalNet, sink));
    if (!sinkSlot)
      return invalid("routed sink endpoint is absent from its RouteTree");

    pathSlots_.clear();
    PnrIndex slot = *sinkSlot;
    while (slot < nodeEpochs_.size() && nodeEpochs_[slot] != projectionEpoch_) {
      if (pathSlots_.size() >= tree.activeNodeCount())
        return invalid("RouteTree ancestry is cyclic");
      pathSlots_.push_back(slot);
      const RouteTreeNode &node = tree.node(slot);
      if (node.parentArc == getInvalidPnrIndex()) {
        slot = getInvalidPnrIndex();
        break;
      }
      if (node.parentArc >= arcs.size())
        return invalid("RouteTree parent arc is out of range");
      const auto parent = tree.parentNodeSlot(slot);
      if (!parent)
        return invalid("RouteTree parent endpoint is absent");
      slot = *parent;
    }
    if (slot != getInvalidPnrIndex() && slot >= nodeEpochs_.size())
      return invalid("RouteTree node slot is out of range");

    bool affected = slot != getInvalidPnrIndex() && nodeAffected_[slot] != 0;
    for (PnrIndex pathSlot : llvm::reverse(pathSlots_)) {
      const RouteTreeNode &node = tree.node(pathSlot);
      if (node.parentArc != getInvalidPnrIndex()) {
        const PnrIndex traversal = arcs[node.parentArc].traversal;
        if (traversal >= routing.traversals().size())
          return invalid("RouteTree traversal is out of range");
        affected |= traversalOverused(candidate, costs, traversal);
      }
      nodeEpochs_[pathSlot] = projectionEpoch_;
      nodeAffected_[pathSlot] = affected ? 1 : 0;
    }
    sinkAffected_[sink] = nodeAffected_[*sinkSlot];
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (PnrIndex dependent = 0; dependent < net.sinkCount; ++dependent) {
      if (sinkAffected_[dependent])
        continue;
      auto prerequisites = spatialSinkProgressDependencies(
          candidate.problem(), logicalNet, dependent);
      if (!prerequisites)
        return prerequisites.takeError();
      bool prerequisiteAffected = false;
      for (const FrozenSpatialProgressPrerequisite &prerequisite :
           *prerequisites) {
        const auto *external =
            std::get_if<FrozenSpatialExternalSinkPrerequisite>(&prerequisite);
        if (!external)
          continue;
        if (external->sink >= net.sinkCount)
          return invalid("external sink progress prerequisite is out of range");
        prerequisiteAffected |= sinkAffected_[external->sink] != 0;
      }
      if (!prerequisiteAffected)
        continue;
      auto localBoundary = spatialTerminalProvidesLocalProgressBoundary(
          candidate, candidate.problem()
                         .transfers()
                         .logicalNetSinkBindings()[net.sinkOffset + dependent]);
      if (!localBoundary)
        return localBoundary.takeError();
      if (*localBoundary)
        continue;
      sinkAffected_[dependent] = 1;
      changed = true;
    }
  }

  selectedSinks_.clear();
  for (PnrIndex sink = 0; sink < net.sinkCount; ++sink)
    if (sinkAffected_[sink])
      selectedSinks_.push_back(sink);
  if (selectedSinks_.empty())
    return SpatialNegotiatedRoutePlan{SpatialNegotiatedRouteScope::Preserve,
                                      {}};
  if (selectedSinks_.size() == net.sinkCount)
    return SpatialNegotiatedRoutePlan{SpatialNegotiatedRouteScope::WholeNet,
                                      {}};
  return SpatialNegotiatedRoutePlan{SpatialNegotiatedRouteScope::SelectedSinks,
                                    selectedSinks_};
}

std::size_t SpatialRouteTreePruningScratch::retainedStorageBytes() const {
  return retainedBytes(traversalEpochs_) + retainedBytes(traversalOveruse_) +
         retainedBytes(nodeEpochs_) + retainedBytes(nodeAffected_) +
         retainedBytes(sinkAffected_) + retainedBytes(selectedSinks_) +
         retainedBytes(pathSlots_);
}
