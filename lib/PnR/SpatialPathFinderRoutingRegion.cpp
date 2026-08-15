#include "PnR/SpatialPathFinderRouter.h"

#include "SpatialRouteConstraintModel.h"
#include "SpatialTagConstraintModel.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstdint>
#include <system_error>
#include <utility>

namespace loom::pnr {
namespace {

llvm::Error routingRegionError(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "invalid Spatial PathFinder routing region: " + message);
}

} // namespace

llvm::Expected<bool>
SpatialPathFinderRouterScratch::expandRoutingRelationClosure(
    std::uint64_t logicalNetLimit) {
  if (!preparedProblem_)
    return routingRegionError("scratch is not prepared");
  const auto &routeConstraints = preparedProblem_->routeConstraints();
  const auto &tagConstraints = preparedProblem_->tagConstraints();
  bool expanded = false;
  const auto addNet = [&](PnrIndex logicalNet) -> llvm::Error {
    if (logicalNet >= routingRegionNetMarks_.size())
      return routingRegionError(
          "relation closure contains a foreign logical net");
    if (routingRegionNetMarks_[logicalNet])
      return llvm::Error::success();
    routingRegionNetMarks_[logicalNet] = 1;
    routingRegionNets_.push_back(logicalNet);
    expanded = true;
    if (logicalNetLimit != 0 && routingRegionNets_.size() > logicalNetLimit)
      return llvm::make_error<SpatialPathFinderClosureFailure>(
          SpatialPathFinderClosureFailure::Kind::RegionalLimit,
          "Spatial PathFinder relation closure exceeds its regional "
          "logical-net limit",
          SpatialFixedTerminalCutCertificate{}, 0, 0, routingRegionNets_.size(),
          logicalNetLimit);
    return llvm::Error::success();
  };

  for (std::size_t cursor = 0; cursor < routingRegionNets_.size(); ++cursor) {
    const PnrIndex logicalNet = routingRegionNets_[cursor];
    for (PnrIndex member : routeConstraints.equalityClosure(logicalNet))
      if (llvm::Error error = addNet(member))
        return std::move(error);
    const PnrIndex equalityClass = tagConstraints.classOfNet(logicalNet);
    for (PnrIndex member : tagConstraints.classMembers(equalityClass))
      if (llvm::Error error = addNet(member))
        return std::move(error);
    for (PnrIndex group : tagConstraints.classDisjointGroups(equalityClass))
      for (PnrIndex peerClass : tagConstraints.disjointGroupMembers(group))
        for (PnrIndex member : tagConstraints.classMembers(peerClass))
          if (llvm::Error error = addNet(member))
            return std::move(error);
  }
  if (expanded)
    llvm::sort(routingRegionNets_);
  return expanded;
}

llvm::Expected<bool>
SpatialPathFinderRouterScratch::expandExactRegionalConflictClosure(
    const SpatialCandidateState &candidate, const SpatialRouteCostState &costs,
    std::uint64_t logicalNetLimit) {
  if (!preparedProblem_)
    return routingRegionError("scratch is not prepared");
  const FrozenSpatialRoutingGraph &routing = preparedProblem_->routing();
  const FrozenSpatialResourceIndex &resources = preparedProblem_->resources();
  bool expanded = false;
  for (PnrIndex logicalNet = 0;
       logicalNet < preparedProblem_->transfers().logicalNets().size();
       ++logicalNet) {
    bool contributes = false;
    for (PnrIndex capacity = 0;
         capacity < resources.capacityDimensions().size(); ++capacity) {
      if (!regionalCapacityMarks_[capacity] ||
          costs.workingCapacityUsageRaw(capacity) <=
              resources.capacityDimensions()[capacity].capacity)
        continue;
      const auto offsets = routing.capacityRouteClaimOffsets();
      if (capacity + 1 >= offsets.size())
        return routingRegionError(
            "conflict closure capacity incidence is out of range");
      for (PnrIndex claim : routing.capacityRouteClaims().slice(
               offsets[capacity], offsets[capacity + 1] - offsets[capacity])) {
        if (claim >= routing.routeClaims().size())
          return routingRegionError(
              "conflict closure route claim is out of range");
        if (candidate.logicalNetRouteClaimRefcount(logicalNet, claim) != 0) {
          contributes = true;
          break;
        }
      }
      if (contributes)
        break;
    }
    if (!contributes) {
      for (const SpatialTagDomainUse &use :
           costs.logicalNetTagDomainUses(logicalNet)) {
        if (use.domain >= regionalTagDomainMarks_.size())
          return routingRegionError(
              "conflict closure tag domain is out of range");
        if (!regionalTagDomainMarks_[use.domain])
          continue;
        if (costs.tagDomainResidentOveruse(use.domain) != 0 ||
            costs.tagDomainConflictCount(use.domain) != 0 ||
            costs.workingTagDomainUsage(use.domain) >
                costs.tagDomainEncodingCapacity(use.domain)) {
          contributes = true;
          break;
        }
      }
    }
    if (!contributes || routingRegionNetMarks_[logicalNet])
      continue;
    routingRegionNetMarks_[logicalNet] = 1;
    routingRegionNets_.push_back(logicalNet);
    expanded = true;
    if (routingRegionNets_.size() > logicalNetLimit)
      return llvm::make_error<SpatialPathFinderClosureFailure>(
          SpatialPathFinderClosureFailure::Kind::RegionalLimit,
          "Spatial PathFinder conflict closure exceeds its regional "
          "logical-net limit",
          SpatialFixedTerminalCutCertificate{}, 0, 0, routingRegionNets_.size(),
          logicalNetLimit);
  }
  auto relationExpanded = expandRoutingRelationClosure(logicalNetLimit);
  if (!relationExpanded)
    return relationExpanded.takeError();
  const bool closureExpanded = expanded || *relationExpanded;
  if (closureExpanded)
    llvm::sort(routingRegionNets_);
  return closureExpanded;
}

} // namespace loom::pnr
