#include "PnR/SpatialPathFinderRouter.h"

#include "Common/MappingDebugLog.h"
#include "Fabric/Identity/FabricRefText.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace loom::pnr;

char SpatialPathFinderClosureFailure::ID;

SpatialPathFinderClosureFailure::SpatialPathFinderClosureFailure(
    Kind kind, std::string message, PnrIndex certificateCapacity,
    std::uint64_t mandatoryUsage, std::uint64_t physicalCapacity,
    std::vector<PnrIndex> forcedLogicalNets)
    : kind_(kind), message_(std::move(message)),
      certificateCapacity_(certificateCapacity),
      mandatoryUsage_(mandatoryUsage), physicalCapacity_(physicalCapacity),
      forcedLogicalNets_(std::move(forcedLogicalNets)) {}

void SpatialPathFinderClosureFailure::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code SpatialPathFinderClosureFailure::convertToErrorCode() const {
  switch (kind_) {
  case Kind::NonClosure:
  case Kind::NoProgress:
    return std::make_error_code(std::errc::resource_unavailable_try_again);
  case Kind::FixedTerminalCapacityCut:
    return std::make_error_code(std::errc::address_not_available);
  case Kind::SelectedCombinationalHandshakeCycle:
    return std::make_error_code(std::errc::state_not_recoverable);
  }
  llvm_unreachable("invalid Spatial PathFinder closure failure kind");
}

namespace {

enum class RankTrendTransition : std::uint8_t {
  Equal,
  Improved,
  Ineligible,
  Regressed,
};

llvm::Error pathFinderError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("invalid Spatial PathFinder route: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

std::string errorMessage(const llvm::ErrorInfoBase &error) {
  std::string message;
  llvm::raw_string_ostream stream(message);
  error.log(stream);
  return message;
}

template <typename T> std::size_t retainedBytes(const std::vector<T> &values) {
  return values.capacity() * sizeof(T);
}

llvm::Error rollbackIteration(SpatialMoveTransaction &move,
                              SpatialRouteCostState &costs,
                              llvm::Error failure) {
  move.rollback();
  if (llvm::Error reset = costs.resetFromCandidate())
    return llvm::joinErrors(std::move(failure), std::move(reset));
  return failure;
}

std::optional<PnrIndex>
resourceStateForCapacity(const FrozenSpatialResourceIndex &resources,
                         PnrIndex capacity) {
  for (auto [state, record] : llvm::enumerate(resources.resourceStates()))
    if (capacity >= record.capacityOffset &&
        capacity - record.capacityOffset < record.capacityCount)
      return static_cast<PnrIndex>(state);
  return std::nullopt;
}

std::optional<PnrIndex>
resourceOwnerForState(const FrozenSpatialResourceIndex &resources,
                      PnrIndex state) {
  for (auto [owner, record] : llvm::enumerate(resources.resourceOwners()))
    if (state >= record.stateOffset &&
        state - record.stateOffset < record.stateCount)
      return static_cast<PnrIndex>(owner);
  return std::nullopt;
}

llvm::json::Array
encodeSelectedOrdinalRanges(llvm::ArrayRef<std::uint8_t> selected) {
  llvm::json::Array ranges;
  for (std::size_t begin = 0; begin < selected.size();) {
    while (begin < selected.size() && !selected[begin])
      ++begin;
    if (begin == selected.size())
      break;
    std::size_t end = begin + 1;
    while (end < selected.size() && selected[end])
      ++end;
    llvm::json::Object range;
    range["begin"] = static_cast<std::uint64_t>(begin);
    range["end"] = static_cast<std::uint64_t>(end);
    ranges.push_back(std::move(range));
    begin = end;
  }
  return ranges;
}

struct ProjectionDisposition final {
  bool violationsZero = true;
  bool temporaryAdmitted = false;
};

llvm::Expected<ProjectionDisposition>
classifyProjection(const SpatialCandidateState &candidate,
                   const SpatialCandidateRouteProjection &projection,
                   SpatialRoutingClosureRequirement closureRequirement) {
  ProjectionDisposition result;
  result.temporaryAdmitted =
      closureRequirement ==
      SpatialRoutingClosureRequirement::PolicyAdmittedTemporary;
  const auto admitted =
      candidate.problem().config().policy().temporaryViolations.admitted;
  for (std::uint32_t ordinal = 0;
       ordinal != loom::resolvedPnrViolationKindCount; ++ordinal) {
    const auto kind = static_cast<loom::ResolvedPnrViolationKind>(ordinal);
    auto value = spatialMappingViolationValue(candidate, projection, kind);
    if (!value)
      return value.takeError();
    if (*value == 0)
      continue;
    result.violationsZero = false;
    result.temporaryAdmitted &= llvm::is_contained(admitted, kind);
  }
  return result;
}

} // namespace

llvm::Error SpatialPathFinderRouterScratch::prepare(
    const FrozenSpatialPnrProblem &problem) {
  if (llvm::Error error = netRouter_.prepare(problem))
    return error;
  const std::size_t logicalNetCount = problem.transfers().logicalNets().size();
  const std::size_t routeClaimCount = problem.routing().routeClaims().size();
  const std::size_t capacityCount =
      problem.resources().capacityDimensions().size();

  netOrder_.clear();
  netOrder_.reserve(logicalNetCount);
  activeClaimBits_.assign((routeClaimCount + 63) / 64, 0);
  claimEpochs_.assign(routeClaimCount, 0);
  capacityEpochs_.assign(capacityCount, 0);
  capacityNetQCosts_.assign(capacityCount, 0);
  touchedCapacities_.clear();
  touchedCapacities_.reserve(capacityCount);
  regionalCapacityMarks_.assign(capacityCount, 0);
  constraintSweepNets_.clear();
  constraintSweepNets_.reserve(logicalNetCount);
  capturedSinkPathOffsets_.clear();
  capturedSinkPathOffsets_.reserve(
      problem.transfers().logicalNetSinks().size() + 1);
  capturedForwardArcs_.clear();
  reversePath_.clear();
  reversePath_.reserve(problem.routing().routingEndpoints().size());
  cutBlockedTraversals_.assign(problem.routing().traversals().size(), 0);
  cutReachableEndpoints_.assign(problem.routing().routingEndpoints().size(), 0);
  cutSeenTraversals_.assign(problem.routing().traversals().size(), 0);
  cutSeenEndpoints_.assign(problem.routing().routingEndpoints().size(), 0);
  cutWorklist_.clear();
  cutWorklist_.reserve(problem.routing().routingEndpoints().size());
  cutContributingNets_.clear();
  cutContributingNets_.reserve(logicalNetCount);
  cutForcedNets_.clear();
  cutForcedNets_.reserve(logicalNetCount);
  cutCertificateForcedNets_.clear();
  cutCertificateForcedNets_.reserve(logicalNetCount);
  cutPayloadWidths_.clear();
  cutPayloadWidths_.reserve(logicalNetCount);
  cutMinimumClaims_.clear();
  cutMinimumClaims_.reserve(logicalNetCount);
  cutTouchedClaims_.clear();
  cutTouchedClaims_.reserve(routeClaimCount);
  cutNetClaimRefcounts_.assign(routeClaimCount, 0);
  cutClaimSelectionCounts_.assign(routeClaimCount, 0);
  cutClaimTraversalRefcounts_.assign(routeClaimCount, 0);
  rankTrendTransitions_.clear();
  projectionEpoch_ = 0;
  negotiationIterationCount_ = 0;
  preparedProblem_ = &problem;
  return llvm::Error::success();
}

llvm::Error SpatialPathFinderRouterScratch::captureCurrentRoutes(
    const SpatialCandidateState &candidate,
    llvm::ArrayRef<PnrIndex> logicalNets) {
  capturedSinkPathOffsets_.clear();
  capturedForwardArcs_.clear();
  capturedSinkPathOffsets_.push_back(0);
  const FrozenSpatialRoutingGraph &routing = candidate.problem().routing();
  const auto arcs = routing.routingArcs();
  const auto arcSources = routing.arcSources();
  const auto captureLogicalNet = [&](PnrIndex logicalNet) -> llvm::Error {
    if (logicalNet >= candidate.problem().transfers().logicalNets().size())
      return pathFinderError("temporary routing region contains a foreign net");
    const RouteTreeState &tree = candidate.routeTree(logicalNet);
    if (!tree.isRouted())
      return pathFinderError("temporary iterate contains an unrouted net");
    const PnrIndex sinkCount =
        candidate.problem().transfers().logicalNets()[logicalNet].sinkCount;
    for (PnrIndex sink = 0; sink < sinkCount; ++sink) {
      const auto endpoint = tree.sinkEndpoint(sink);
      if (!endpoint)
        return pathFinderError(
            "temporary iterate has an unattached sink obligation");
      auto slot = tree.findNode(*endpoint);
      if (!slot)
        return pathFinderError(
            "temporary iterate sink is absent from its RouteTree");
      reversePath_.clear();
      for (std::size_t depth = 0;; ++depth) {
        if (depth > tree.nodeStorage().size())
          return pathFinderError(
              "temporary iterate RouteTree contains a cycle");
        const RouteTreeNode &node = tree.node(*slot);
        if (node.parentArc == getInvalidPnrIndex())
          break;
        if (node.parentArc >= arcs.size() ||
            node.parentArc >= arcSources.size())
          return pathFinderError(
              "temporary iterate RouteTree arc is out of range");
        reversePath_.push_back(node.parentArc);
        slot = tree.findNode(arcSources[node.parentArc]);
        if (!slot)
          return pathFinderError(
              "temporary iterate RouteTree parent is absent");
      }
      capturedForwardArcs_.insert(capturedForwardArcs_.end(),
                                  reversePath_.rbegin(), reversePath_.rend());
      capturedSinkPathOffsets_.push_back(capturedForwardArcs_.size());
    }
    return llvm::Error::success();
  };
  if (logicalNets.empty()) {
    for (PnrIndex logicalNet = 0;
         logicalNet < candidate.problem().transfers().logicalNets().size();
         ++logicalNet)
      if (llvm::Error error = captureLogicalNet(logicalNet))
        return error;
  } else {
    for (PnrIndex logicalNet : logicalNets)
      if (llvm::Error error = captureLogicalNet(logicalNet))
        return error;
  }
  return llvm::Error::success();
}

llvm::Error SpatialPathFinderRouterScratch::restoreCapturedRoutes(
    SpatialMoveTransaction &move, SpatialCandidateState &candidate,
    SpatialRouteCostState &costs, llvm::ArrayRef<PnrIndex> logicalNets) {
  const FrozenSpatialRoutingGraph &routing = candidate.problem().routing();
  const auto arcs = routing.routingArcs();
  std::size_t sinkPath = 0;
  const auto restoreLogicalNet = [&](PnrIndex logicalNet) -> llvm::Error {
    if (logicalNet >= candidate.problem().transfers().logicalNets().size())
      return pathFinderError("captured routing region contains a foreign net");
    auto projection = projectLogicalNet(candidate, costs, logicalNet);
    if (!projection)
      return projection.takeError();
    if (llvm::Error error =
            costs.selectLogicalNet(logicalNet, activeClaimBits_))
      return error;
    if (llvm::Error error = move.ripUpWholeRoute(logicalNet))
      return error;
    const PnrIndex source = candidate.logicalNetSourceEndpoint(logicalNet);
    if (llvm::Error error = move.bindRouteSource(logicalNet, source))
      return error;
    const PnrIndex sinkCount =
        candidate.problem().transfers().logicalNets()[logicalNet].sinkCount;
    for (PnrIndex sink = 0; sink < sinkCount; ++sink, ++sinkPath) {
      if (llvm::Error error = move.bindRouteSink(
              logicalNet, sink,
              candidate.logicalNetSinkEndpoint(logicalNet, sink)))
        return error;
      if (sinkPath + 1 >= capturedSinkPathOffsets_.size())
        return pathFinderError("captured route path domain is truncated");
      const std::size_t begin = capturedSinkPathOffsets_[sinkPath];
      const std::size_t end = capturedSinkPathOffsets_[sinkPath + 1];
      if (begin > end || end > capturedForwardArcs_.size())
        return pathFinderError("captured route path range is invalid");
      const llvm::ArrayRef<PnrIndex> path(capturedForwardArcs_.data() + begin,
                                          end - begin);
      PnrIndex attachment = source;
      std::size_t pathBegin = 0;
      const RouteTreeState &tree = candidate.routeTree(logicalNet);
      for (auto [index, arc] : llvm::enumerate(path)) {
        if (arc >= arcs.size())
          return pathFinderError("captured route arc is out of range");
        if (tree.findNode(arcs[arc].target)) {
          attachment = arcs[arc].target;
          pathBegin = index + 1;
        }
      }
      if (llvm::Error error = move.attachRoutePath(
              logicalNet, attachment, path.drop_front(pathBegin), sink))
        return error;
    }
    auto restored = projectLogicalNet(candidate, costs, logicalNet);
    if (!restored)
      return restored.takeError();
    if (llvm::Error error =
            costs.updateSelectedLogicalNetClaims(activeClaimBits_))
      return error;
    if (llvm::Error error = costs.acceptSelectedLogicalNet())
      return error;
    return llvm::Error::success();
  };
  if (logicalNets.empty()) {
    for (PnrIndex logicalNet = 0;
         logicalNet < candidate.problem().transfers().logicalNets().size();
         ++logicalNet)
      if (llvm::Error error = restoreLogicalNet(logicalNet))
        return error;
  } else {
    for (PnrIndex logicalNet : logicalNets)
      if (llvm::Error error = restoreLogicalNet(logicalNet))
        return error;
  }
  if (sinkPath + 1 != capturedSinkPathOffsets_.size())
    return pathFinderError("captured route path domain has trailing entries");
  return llvm::Error::success();
}

llvm::Expected<SpatialPathFinderRouterScratch::CapacityConflictAnalysis>
SpatialPathFinderRouterScratch::analyzeCapacityConflicts(
    const SpatialCandidateState &candidate, const SpatialRouteCostState &costs,
    std::uint64_t iteration, std::uint64_t sessionIteration) {
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  const FrozenSpatialResourceIndex &resources = problem.resources();
  const FrozenSpatialRoutingGraph &routing = problem.routing();
  const auto arcs = routing.routingArcs();
  const auto arcSources = routing.arcSources();
  const auto adjacencyOffsets = routing.adjacencyOffsets();
  const bool emitDecisions =
      loom::mapping_debug::enabled(loom::mapping_debug::Level::Decision);
  const bool emitDetails =
      loom::mapping_debug::enabled(loom::mapping_debug::Level::Detail);
  CapacityConflictAnalysis analysis;
  cutCertificateForcedNets_.clear();

  std::optional<PnrIndex> selectedConflictCapacity;
  for (PnrIndex capacity = 0; capacity < resources.capacityDimensions().size();
       ++capacity) {
    if (!regionalCapacityMarks_[capacity])
      continue;
    const std::uint64_t usage = costs.workingCapacityUsageRaw(capacity);
    if (usage <= resources.capacityDimensions()[capacity].capacity)
      continue;
    if (analysis.conflictCount == std::numeric_limits<std::uint64_t>::max())
      return pathFinderError("capacity conflict count overflows u64");
    ++analysis.conflictCount;
    if (!selectedConflictCapacity ||
        usage < costs.workingCapacityUsageRaw(*selectedConflictCapacity) ||
        (usage == costs.workingCapacityUsageRaw(*selectedConflictCapacity) &&
         capacity < *selectedConflictCapacity))
      selectedConflictCapacity = capacity;
  }
  if (!selectedConflictCapacity)
    return analysis;

  const std::array<PnrIndex, 1> conflictCapacities{*selectedConflictCapacity};
  for (PnrIndex capacity : conflictCapacities) {
    const std::uint64_t usage = costs.workingCapacityUsageRaw(capacity);
    const auto &capacityRecord = resources.capacityDimensions()[capacity];
    const std::uint64_t physicalCapacity = capacityRecord.capacity;
    const auto capacityClaimOffsets = routing.capacityRouteClaimOffsets();
    const auto capacityClaims = routing.capacityRouteClaims();
    if (capacity + 1 >= capacityClaimOffsets.size())
      return pathFinderError("capacity-to-claim incidence is out of range");
    const auto activeCapacityClaims = capacityClaims.slice(
        capacityClaimOffsets[capacity],
        capacityClaimOffsets[capacity + 1] - capacityClaimOffsets[capacity]);

    std::fill(cutClaimSelectionCounts_.begin(), cutClaimSelectionCounts_.end(),
              0);
    std::fill(cutClaimTraversalRefcounts_.begin(),
              cutClaimTraversalRefcounts_.end(), 0);
    std::fill(cutSeenTraversals_.begin(), cutSeenTraversals_.end(), 0);
    std::fill(cutSeenEndpoints_.begin(), cutSeenEndpoints_.end(), 0);
    cutContributingNets_.clear();
    cutForcedNets_.clear();
    std::uint64_t derivedUsage = capacityRecord.initialOccupancy;
    for (PnrIndex logicalNet = 0;
         logicalNet < problem.transfers().logicalNets().size(); ++logicalNet) {
      cutTouchedClaims_.clear();
      const RouteTreeState &tree = candidate.routeTree(logicalNet);
      for (const RouteTreeNode &node : tree.nodeStorage()) {
        if (!node.isActive() || node.parentArc == getInvalidPnrIndex())
          continue;
        if (node.parentArc >= arcs.size() ||
            node.parentArc >= arcSources.size())
          return pathFinderError(
              "capacity analysis RouteTree arc is out of range");
        const PnrIndex traversal = arcs[node.parentArc].traversal;
        if (traversal >= routing.traversals().size())
          return pathFinderError("capacity analysis traversal is out of range");
        const FrozenSpatialTraversal &traversalRecord =
            routing.traversals()[traversal];
        bool traversalContributes = false;
        for (PnrIndex claim : routing.traversalClaimKeys().slice(
                 traversalRecord.routeClaimOffset,
                 traversalRecord.routeClaimCount)) {
          if (claim >= routing.routeClaims().size())
            return pathFinderError(
                "capacity analysis route claim is out of range");
          const FrozenSpatialRouteClaim &claimRecord =
              routing.routeClaims()[claim];
          if (claimRecord.capacityDimension != capacity ||
              claimRecord.amount == 0)
            continue;
          traversalContributes = true;
          if (cutNetClaimRefcounts_[claim] == 0) {
            cutTouchedClaims_.push_back(claim);
            if (cutClaimSelectionCounts_[claim] ==
                std::numeric_limits<PnrIndex>::max())
              return pathFinderError(
                  "route-tree claim selection count overflows PnrIndex");
            ++cutClaimSelectionCounts_[claim];
            if (claimRecord.amount >
                std::numeric_limits<std::uint64_t>::max() - derivedUsage)
              return pathFinderError("route-tree capacity usage exceeds u64");
            derivedUsage += claimRecord.amount;
          }
          if (cutNetClaimRefcounts_[claim] ==
              std::numeric_limits<PnrIndex>::max())
            return pathFinderError(
                "route-tree claim traversal refcount overflows PnrIndex");
          ++cutNetClaimRefcounts_[claim];
        }
        if (emitDecisions && traversalContributes) {
          cutSeenTraversals_[traversal] = 1;
          cutSeenEndpoints_[arcSources[node.parentArc]] = 1;
          cutSeenEndpoints_[arcs[node.parentArc].target] = 1;
        }
      }
      if (cutTouchedClaims_.empty())
        continue;
      cutContributingNets_.push_back(logicalNet);
      for (PnrIndex claim : cutTouchedClaims_) {
        if (cutNetClaimRefcounts_[claim] >
            std::numeric_limits<std::uint64_t>::max() -
                cutClaimTraversalRefcounts_[claim])
          return pathFinderError(
              "route-tree claim traversal count exceeds u64");
        cutClaimTraversalRefcounts_[claim] += cutNetClaimRefcounts_[claim];
        cutNetClaimRefcounts_[claim] = 0;
      }
    }
    if (derivedUsage != usage)
      return pathFinderError(
          "working capacity usage disagrees with the active RouteTrees at " +
          llvm::Twine(capacity) + ": working=" + llvm::Twine(usage) +
          " projected=" + llvm::Twine(derivedUsage));

    std::fill(cutBlockedTraversals_.begin(), cutBlockedTraversals_.end(), 0);
    const auto claimTraversalOffsets = routing.routeClaimTraversalOffsets();
    const auto claimTraversals = routing.routeClaimTraversals();
    const auto traversalArcOffsets = routing.traversalArcOffsets();
    const auto traversalArcs = routing.traversalArcs();
    for (PnrIndex entry = capacityClaimOffsets[capacity];
         entry < capacityClaimOffsets[capacity + 1]; ++entry) {
      if (entry >= capacityClaims.size())
        return pathFinderError("capacity-to-claim entry is out of range");
      const PnrIndex claim = capacityClaims[entry];
      if (claim >= routing.routeClaims().size() ||
          claim + 1 >= claimTraversalOffsets.size())
        return pathFinderError("claim-to-traversal incidence is out of range");
      if (routing.routeClaims()[claim].amount == 0)
        continue;
      for (PnrIndex traversalEntry = claimTraversalOffsets[claim];
           traversalEntry < claimTraversalOffsets[claim + 1];
           ++traversalEntry) {
        if (traversalEntry >= claimTraversals.size() ||
            claimTraversals[traversalEntry] >= cutBlockedTraversals_.size())
          return pathFinderError("claim traversal entry is out of range");
        cutBlockedTraversals_[claimTraversals[traversalEntry]] = 1;
      }
    }

    cutPayloadWidths_.clear();
    cutMinimumClaims_.clear();
    const auto minimumClaimForPayload =
        [&](std::uint32_t payloadWidth) -> llvm::Expected<std::uint64_t> {
      for (std::size_t index = 0; index < cutPayloadWidths_.size(); ++index)
        if (cutPayloadWidths_[index] == payloadWidth)
          return cutMinimumClaims_[index];
      std::uint64_t minimumClaim = std::numeric_limits<std::uint64_t>::max();
      for (PnrIndex claim : activeCapacityClaims) {
        if (claim + 1 >= claimTraversalOffsets.size())
          return pathFinderError(
              "claim-to-traversal incidence is out of range");
        bool payloadCompatible = false;
        for (PnrIndex traversalEntry = claimTraversalOffsets[claim];
             traversalEntry < claimTraversalOffsets[claim + 1] &&
             !payloadCompatible;
             ++traversalEntry) {
          if (traversalEntry >= claimTraversals.size())
            return pathFinderError("claim traversal entry is out of range");
          const PnrIndex traversal = claimTraversals[traversalEntry];
          if (traversal + 1 >= traversalArcOffsets.size())
            return pathFinderError(
                "traversal-to-arc incidence is out of range");
          for (PnrIndex arcEntry = traversalArcOffsets[traversal];
               arcEntry < traversalArcOffsets[traversal + 1]; ++arcEntry) {
            if (arcEntry >= traversalArcs.size() ||
                traversalArcs[arcEntry] >= arcs.size())
              return pathFinderError("traversal arc entry is out of range");
            if (arcs[traversalArcs[arcEntry]].payloadCapacityBits >=
                payloadWidth) {
              payloadCompatible = true;
              break;
            }
          }
        }
        const std::uint64_t amount = routing.routeClaims()[claim].amount;
        if (payloadCompatible && amount != 0)
          minimumClaim = std::min(minimumClaim, amount);
      }
      cutPayloadWidths_.push_back(payloadWidth);
      cutMinimumClaims_.push_back(minimumClaim);
      return minimumClaim;
    };

    std::uint64_t mandatoryUsage = capacityRecord.initialOccupancy;
    std::uint64_t forcedNetCount = 0;
    for (PnrIndex logicalNet : cutContributingNets_) {
      const std::uint32_t payloadWidth =
          candidate.logicalNetPayloadWidth(logicalNet);
      auto minimumClaimValue = minimumClaimForPayload(payloadWidth);
      if (!minimumClaimValue)
        return minimumClaimValue.takeError();
      const std::uint64_t minimumClaim = *minimumClaimValue;

      std::fill(cutReachableEndpoints_.begin(), cutReachableEndpoints_.end(),
                0);
      cutWorklist_.clear();
      const PnrIndex source = candidate.logicalNetSourceEndpoint(logicalNet);
      if (source >= cutReachableEndpoints_.size())
        return pathFinderError("cut source endpoint is out of range");
      cutReachableEndpoints_[source] = 1;
      cutWorklist_.push_back(source);
      for (std::size_t cursor = 0; cursor < cutWorklist_.size(); ++cursor) {
        const PnrIndex endpoint = cutWorklist_[cursor];
        if (endpoint + 1 >= adjacencyOffsets.size())
          return pathFinderError("cut adjacency entry is out of range");
        for (PnrIndex arc = adjacencyOffsets[endpoint];
             arc < adjacencyOffsets[endpoint + 1]; ++arc) {
          if (arc >= arcs.size())
            return pathFinderError("cut routing arc is out of range");
          const EndpointRoutingArc &record = arcs[arc];
          if (record.traversal >= cutBlockedTraversals_.size() ||
              record.target >= cutReachableEndpoints_.size())
            return pathFinderError("cut routing arc endpoint is out of range");
          if (cutBlockedTraversals_[record.traversal] ||
              record.payloadCapacityBits < payloadWidth ||
              cutReachableEndpoints_[record.target])
            continue;
          cutReachableEndpoints_[record.target] = 1;
          cutWorklist_.push_back(record.target);
        }
      }

      bool separatingCut = false;
      const PnrIndex sinkCount =
          problem.transfers().logicalNets()[logicalNet].sinkCount;
      for (PnrIndex sink = 0; sink < sinkCount; ++sink) {
        const PnrIndex endpoint =
            candidate.logicalNetSinkEndpoint(logicalNet, sink);
        if (endpoint >= cutReachableEndpoints_.size())
          return pathFinderError("cut sink endpoint is out of range");
        separatingCut |= !cutReachableEndpoints_[endpoint];
      }
      if (separatingCut) {
        if (minimumClaim == std::numeric_limits<std::uint64_t>::max())
          return pathFinderError(
              "separating capacity cut has no positive compatible claim");
        if (minimumClaim >
            std::numeric_limits<std::uint64_t>::max() - mandatoryUsage)
          return pathFinderError("mandatory capacity usage exceeds u64");
        mandatoryUsage += minimumClaim;
        ++forcedNetCount;
        cutForcedNets_.push_back(logicalNet);
      }

      if (emitDetails) {
        llvm::json::Array reachableSinks;
        llvm::json::Array unreachableSinks;
        for (PnrIndex sink = 0; sink < sinkCount; ++sink) {
          const PnrIndex endpoint =
              candidate.logicalNetSinkEndpoint(logicalNet, sink);
          llvm::json::Object row;
          row["endpoint"] = endpoint;
          row["sink"] = sink;
          (cutReachableEndpoints_[endpoint] ? reachableSinks : unreachableSinks)
              .push_back(std::move(row));
        }
        loom::mapping_debug::emit(
            loom::mapping_debug::Level::Detail,
            loom::mapping_debug::Stage::SpatialPnr,
            loom::mapping_debug::Event::CutAnalysis,
            [&](llvm::json::Object &fields) {
              fields["analysis_scope"] =
                  "payload_compatible_capacity_removed_frozen_graph";
              fields["iteration"] = iteration;
              fields["session_iteration"] = sessionIteration;
              fields["capacity_ref"] = capacity;
              fields["logical_net"] = logicalNet;
              fields["source_endpoint"] = source;
              fields["payload_width_bits"] = payloadWidth;
              fields["minimum_positive_claim"] =
                  minimumClaim == std::numeric_limits<std::uint64_t>::max()
                      ? 0
                      : minimumClaim;
              fields["reachable_endpoint_count"] = cutWorklist_.size();
              fields["reachable_endpoint_ranges"] =
                  encodeSelectedOrdinalRanges(cutReachableEndpoints_);
              fields["reachable_sinks"] = std::move(reachableSinks);
              fields["unreachable_sinks"] = std::move(unreachableSinks);
              fields["separating_cut"] = separatingCut;
            });
      }
      if (mandatoryUsage > physicalCapacity)
        break;
    }

    const bool certificate = mandatoryUsage > physicalCapacity;
    if (certificate && !analysis.hasCertificate()) {
      analysis.certificateCapacity = capacity;
      analysis.mandatoryUsage = mandatoryUsage;
      analysis.physicalCapacity = physicalCapacity;
      cutCertificateForcedNets_ = cutForcedNets_;
    }

    constexpr std::uint64_t decisionConflictEventLimit = 16;
    constexpr std::size_t decisionOrdinalSampleLimit = 8;
    const bool emitConflict =
        emitDetails ||
        (emitDecisions && (certificate || analysis.diagnosticConflictCount <
                                              decisionConflictEventLimit));
    if (emitConflict) {
      ++analysis.diagnosticConflictCount;
      const std::size_t sampleLimit =
          emitDetails ? std::numeric_limits<std::size_t>::max()
                      : decisionOrdinalSampleLimit;
      llvm::json::Array logicalNets;
      for (PnrIndex logicalNet : cutContributingNets_)
        if (logicalNets.size() < sampleLimit)
          logicalNets.push_back(logicalNet);
      llvm::json::Array claims;
      std::uint64_t claimCount = 0;
      for (PnrIndex claim = 0; claim < routing.routeClaims().size(); ++claim) {
        if (cutClaimSelectionCounts_[claim] == 0)
          continue;
        ++claimCount;
        if (claims.size() >= sampleLimit)
          continue;
        const FrozenSpatialRouteClaim &record = routing.routeClaims()[claim];
        llvm::json::Object row;
        row["amount"] = record.amount;
        row["claim"] = claim;
        row["active_logical_net_count"] = cutClaimSelectionCounts_[claim];
        row["route_tree_traversal_refcount"] =
            cutClaimTraversalRefcounts_[claim];
        row["committed_selection_count"] =
            candidate.routeClaimSelectionCount(claim);
        claims.push_back(std::move(row));
      }
      llvm::json::Array traversals;
      llvm::json::Array endpoints;
      std::uint64_t traversalCount = 0;
      std::uint64_t endpointCount = 0;
      for (PnrIndex traversal = 0; traversal < cutSeenTraversals_.size();
           ++traversal)
        if (cutSeenTraversals_[traversal]) {
          ++traversalCount;
          if (traversals.size() < sampleLimit)
            traversals.push_back(traversal);
        }
      for (PnrIndex endpoint = 0; endpoint < cutSeenEndpoints_.size();
           ++endpoint)
        if (cutSeenEndpoints_[endpoint]) {
          ++endpointCount;
          if (endpoints.size() < sampleLimit)
            endpoints.push_back(endpoint);
        }
      const std::optional<PnrIndex> state =
          resourceStateForCapacity(resources, capacity);
      const std::optional<PnrIndex> owner =
          state ? resourceOwnerForState(resources, *state) : std::nullopt;
      loom::mapping_debug::emit(
          emitDetails ? loom::mapping_debug::Level::Detail
                      : loom::mapping_debug::Level::Decision,
          loom::mapping_debug::Stage::SpatialPnr,
          loom::mapping_debug::Event::CapacityConflict,
          [&](llvm::json::Object &fields) {
            fields["iteration"] = iteration;
            fields["session_iteration"] = sessionIteration;
            fields["capacity_ref"] = capacity;
            fields["usage"] = usage;
            fields["capacity"] = physicalCapacity;
            fields["overuse"] = usage - physicalCapacity;
            fields["initial_occupancy"] = capacityRecord.initialOccupancy;
            fields["route_tree_derived_usage"] = derivedUsage;
            fields["committed_candidate_usage"] =
                candidate.routeCapacityUsageRaw(capacity);
            fields["working_matches_route_tree"] = true;
            fields["present_pressure"] = costs.presentPressure();
            fields["mandatory_usage_lower_bound"] = mandatoryUsage;
            fields["forced_logical_net_count"] = forcedNetCount;
            fields["fixed_terminal_cut_certificate"] = certificate;
            fields["contributing_logical_net_count"] =
                cutContributingNets_.size();
            fields["active_claim_count"] = claimCount;
            fields["active_traversal_count"] = traversalCount;
            fields["active_endpoint_count"] = endpointCount;
            fields["diagnostic_sample_limit"] =
                emitDetails ? 0 : decisionOrdinalSampleLimit;
            if (state) {
              fields["resource_state"] = *state;
              fields["resource_state_ref"] = loom::fabric::printFabricRef(
                  resources.resourceStates()[*state].reference);
              fields["capacity_dimension"] =
                  capacity - resources.resourceStates()[*state].capacityOffset;
            }
            if (owner) {
              fields["resource_owner"] = *owner;
              fields["resource_owner_ref"] = loom::fabric::printFabricRef(
                  resources.resourceOwners()[*owner].reference);
            }
            if (emitDetails) {
              fields["logical_nets"] = std::move(logicalNets);
              fields["claims"] = std::move(claims);
              fields["traversals"] = std::move(traversals);
              fields["endpoints"] = std::move(endpoints);
            } else {
              fields["logical_net_sample"] = std::move(logicalNets);
              fields["claim_sample"] = std::move(claims);
              fields["traversal_sample"] = std::move(traversals);
              fields["endpoint_sample"] = std::move(endpoints);
            }
          });
    }
    if (emitDetails) {
      loom::mapping_debug::emit(
          loom::mapping_debug::Level::Detail,
          loom::mapping_debug::Stage::SpatialPnr,
          loom::mapping_debug::Event::CutAnalysis,
          [&](llvm::json::Object &fields) {
            fields["analysis_scope"] = "fixed_terminal_capacity_certificate";
            fields["iteration"] = iteration;
            fields["session_iteration"] = sessionIteration;
            fields["capacity_ref"] = capacity;
            fields["blocked_traversal_ranges"] =
                encodeSelectedOrdinalRanges(cutBlockedTraversals_);
            fields["contributing_logical_net_count"] =
                cutContributingNets_.size();
            fields["forced_logical_net_count"] = forcedNetCount;
            fields["mandatory_usage"] = mandatoryUsage;
            fields["capacity"] = physicalCapacity;
            fields["certificate"] = certificate;
          });
    }
    if (certificate)
      break;
  }
  return analysis;
}

llvm::Expected<SpatialPathFinderRouterScratch::RoutingRegionProjection>
SpatialPathFinderRouterScratch::projectRoutingRegion(
    const SpatialCandidateState &candidate, const SpatialRouteCostState &costs,
    llvm::ArrayRef<PnrIndex> logicalNets) {
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  const auto &routing = problem.routing();
  const auto &resources = problem.resources();
  std::fill(regionalCapacityMarks_.begin(), regionalCapacityMarks_.end(), 0);

  RoutingRegionProjection projection;
  const auto projectNet = [&](PnrIndex logicalNet) -> llvm::Error {
    if (logicalNet >= problem.transfers().logicalNets().size())
      return pathFinderError("routing region contains a foreign logical net");
    const RouteTreeState &tree = candidate.routeTree(logicalNet);
    if (tree.isUnrouted()) {
      const std::uint64_t sinkCount =
          problem.transfers().logicalNets()[logicalNet].sinkCount;
      if (sinkCount > std::numeric_limits<std::uint64_t>::max() -
                          projection.unroutedObligationCount)
        return pathFinderError(
            "regional unrouted-obligation count overflows u64");
      projection.unroutedObligationCount += sinkCount;
      return llvm::Error::success();
    }
    for (const RouteTreeNode &node : tree.nodeStorage()) {
      if (!node.isActive() || node.parentArc == getInvalidPnrIndex())
        continue;
      if (node.parentArc >= routing.routingArcs().size())
        return pathFinderError("regional RouteTree arc is out of range");
      const PnrIndex traversal =
          routing.routingArcs()[node.parentArc].traversal;
      if (traversal >= routing.traversals().size())
        return pathFinderError("regional RouteTree traversal is out of range");
      const FrozenSpatialTraversal &record = routing.traversals()[traversal];
      for (PnrIndex claim : routing.traversalClaimKeys().slice(
               record.routeClaimOffset, record.routeClaimCount)) {
        if (claim >= routing.routeClaims().size())
          return pathFinderError("regional route claim is out of range");
        const PnrIndex capacity =
            routing.routeClaims()[claim].capacityDimension;
        if (capacity >= regionalCapacityMarks_.size())
          return pathFinderError("regional route capacity is out of range");
        regionalCapacityMarks_[capacity] = 1;
      }
    }
    return llvm::Error::success();
  };

  if (logicalNets.empty()) {
    std::fill(regionalCapacityMarks_.begin(), regionalCapacityMarks_.end(), 1);
    for (PnrIndex logicalNet = 0;
         logicalNet < problem.transfers().logicalNets().size(); ++logicalNet)
      if (llvm::Error error = projectNet(logicalNet))
        return std::move(error);
  } else {
    for (PnrIndex logicalNet : logicalNets)
      if (llvm::Error error = projectNet(logicalNet))
        return std::move(error);
  }

  for (PnrIndex capacity = 0; capacity < regionalCapacityMarks_.size();
       ++capacity) {
    if (!regionalCapacityMarks_[capacity])
      continue;
    const std::uint64_t usage = costs.workingCapacityUsageRaw(capacity);
    const std::uint64_t available =
        resources.capacityDimensions()[capacity].capacity;
    if (usage <= available)
      continue;
    const std::uint64_t overuse = usage - available;
    if (overuse > std::numeric_limits<std::uint64_t>::max() -
                      projection.routeCapacityOveruse)
      return pathFinderError("regional route-capacity overuse overflows u64");
    projection.routeCapacityOveruse += overuse;
  }
  return projection;
}

void SpatialPathFinderRouterScratch::beginProjection() {
  ++projectionEpoch_;
  if (projectionEpoch_ == 0) {
    std::fill(claimEpochs_.begin(), claimEpochs_.end(), 0);
    std::fill(capacityEpochs_.begin(), capacityEpochs_.end(), 0);
    projectionEpoch_ = 1;
  }
  std::fill(activeClaimBits_.begin(), activeClaimBits_.end(), 0);
  touchedCapacities_.clear();
}

llvm::Expected<SpatialPathFinderRouterScratch::NetProjection>
SpatialPathFinderRouterScratch::projectLogicalNet(
    const SpatialCandidateState &candidate, const SpatialRouteCostState &costs,
    PnrIndex logicalNet) {
  if (logicalNet >= candidate.problem().transfers().logicalNets().size())
    return pathFinderError("logical net is out of range");
  beginProjection();

  const RouteTreeState &tree = candidate.routeTree(logicalNet);
  if (tree.isUnrouted())
    return NetProjection{0, 0};

  const FrozenSpatialRoutingGraph &routing = candidate.problem().routing();
  for (const RouteTreeNode &node : tree.nodeStorage()) {
    if (!node.isActive() || node.parentArc == getInvalidPnrIndex())
      continue;
    if (node.parentArc >= routing.routingArcs().size())
      return pathFinderError("RouteTree parent arc is out of range");
    const PnrIndex traversal = routing.routingArcs()[node.parentArc].traversal;
    if (traversal >= routing.traversals().size())
      return pathFinderError("RouteTree traversal is out of range");
    const FrozenSpatialTraversal &record = routing.traversals()[traversal];
    for (PnrIndex claim : routing.traversalClaimKeys().slice(
             record.routeClaimOffset, record.routeClaimCount)) {
      if (claim >= routing.routeClaims().size())
        return pathFinderError("RouteTree claim is out of range");
      if (claimEpochs_[claim] == projectionEpoch_)
        continue;
      claimEpochs_[claim] = projectionEpoch_;
      activeClaimBits_[claim / 64] |= std::uint64_t{1} << (claim % 64);

      const FrozenSpatialRouteClaim &claimRecord = routing.routeClaims()[claim];
      const PnrIndex capacity = claimRecord.capacityDimension;
      if (capacity >= capacityNetQCosts_.size())
        return pathFinderError("RouteTree claim capacity is out of range");
      if (capacityEpochs_[capacity] != projectionEpoch_) {
        capacityEpochs_[capacity] = projectionEpoch_;
        capacityNetQCosts_[capacity] = 0;
        touchedCapacities_.push_back(capacity);
      }
      auto qCost =
          accumulateRouteCost(capacityNetQCosts_[capacity], claimRecord.qCost);
      if (!qCost)
        return qCost.takeError();
      capacityNetQCosts_[capacity] = *qCost;
    }
  }

  RouteCost conflictPressure = 0;
  bool contributesToViolation = false;
  for (PnrIndex capacity : touchedCapacities_) {
    const RouteCost overuse = costs.capacityOveruseCost(capacity);
    contributesToViolation |= overuse != 0;
    auto term = scaledRouteProduct(capacityNetQCosts_[capacity], overuse);
    if (!term)
      return term.takeError();
    auto next = accumulateRouteCost(conflictPressure, *term);
    if (!next)
      return next.takeError();
    conflictPressure = *next;
  }
  return NetProjection{
      static_cast<std::uint8_t>(contributesToViolation ? 1 : 2),
      conflictPressure};
}

llvm::Error SpatialPathFinderRouterScratch::buildCanonicalNetOrder(
    const SpatialCandidateState &candidate, const SpatialRouteCostState &costs,
    llvm::ArrayRef<PnrIndex> logicalNets,
    llvm::ArrayRef<RouteCost> evaluationPriorities) {
  const std::size_t logicalNetCount =
      candidate.problem().transfers().logicalNets().size();
  if (!evaluationPriorities.empty() &&
      evaluationPriorities.size() != logicalNetCount)
    return pathFinderError("evaluation-priority vector has the wrong width");

  netOrder_.clear();
  if (!logicalNets.empty()) {
    for (auto [ordinal, logicalNet] : llvm::enumerate(logicalNets)) {
      if (logicalNet >= logicalNetCount)
        return pathFinderError("routing region contains a foreign logical net");
      if (ordinal != 0 && logicalNets[ordinal - 1] >= logicalNet)
        return pathFinderError(
            "routing region is not in canonical unique order");
    }
  }
  const auto append = [&](PnrIndex logicalNet) -> llvm::Error {
    auto projection = projectLogicalNet(candidate, costs, logicalNet);
    if (!projection)
      return projection.takeError();
    netOrder_.push_back(
        {projection->routeStateRank, projection->conflictPressure,
         evaluationPriorities.empty() ? 0 : evaluationPriorities[logicalNet],
         logicalNet});
    return llvm::Error::success();
  };
  if (logicalNets.empty()) {
    for (PnrIndex logicalNet = 0; logicalNet < logicalNetCount; ++logicalNet)
      if (llvm::Error error = append(logicalNet))
        return error;
  } else {
    for (PnrIndex logicalNet : logicalNets)
      if (llvm::Error error = append(logicalNet))
        return error;
  }
  llvm::sort(netOrder_, [](const NetOrderEntry &lhs, const NetOrderEntry &rhs) {
    if (lhs.routeStateRank != rhs.routeStateRank)
      return lhs.routeStateRank < rhs.routeStateRank;
    if (lhs.conflictPressure != rhs.conflictPressure)
      return lhs.conflictPressure > rhs.conflictPressure;
    if (lhs.evaluationPriority != rhs.evaluationPriority)
      return lhs.evaluationPriority > rhs.evaluationPriority;
    return lhs.logicalNet < rhs.logicalNet;
  });
  return llvm::Error::success();
}

llvm::Expected<SpatialPathFinderClosureResult>
SpatialPathFinderRouterScratch::routeToClosure(
    SpatialCandidateState &candidate, SpatialCandidateScratch &candidateScratch,
    SpatialRouteCostState &costs, SpatialPathFinderRoutingLimits limits,
    llvm::ArrayRef<RouteCost> evaluationPriorities,
    SpatialRoutingClosureRequirement closureRequirement) {
  if (llvm::Error error = costs.resetFromCandidate())
    return std::move(error);
  auto moveOrError = candidate.beginMove(candidateScratch);
  if (!moveOrError)
    return moveOrError.takeError();
  SpatialMoveTransaction move = std::move(*moveOrError);

  auto result = routeToClosureInMove(move, candidate, costs, limits, {},
                                     evaluationPriorities, closureRequirement);
  if (!result)
    return rollbackIteration(move, costs, result.takeError());
  auto closed = move.close();
  if (!closed)
    return rollbackIteration(move, costs, closed.takeError());
  if (!*closed)
    return rollbackIteration(
        move, costs,
        llvm::make_error<SpatialPathFinderClosureFailure>(
            SpatialPathFinderClosureFailure::Kind::
                SelectedCombinationalHandshakeCycle,
            "Spatial PathFinder selected a combinational handshake cycle"));
  if (llvm::Error error = move.commit())
    return error;
  return result;
}

llvm::Expected<SpatialPathFinderClosureResult>
SpatialPathFinderRouterScratch::routeToClosureInMove(
    SpatialMoveTransaction &move, SpatialCandidateState &candidate,
    SpatialRouteCostState &costs, SpatialPathFinderRoutingLimits limits,
    llvm::ArrayRef<PnrIndex> logicalNets,
    llvm::ArrayRef<RouteCost> evaluationPriorities,
    SpatialRoutingClosureRequirement closureRequirement) {
  if (!preparedProblem_ || preparedProblem_ != &candidate.problem())
    return pathFinderError("scratch is not prepared for the candidate freeze");
  if (!costs.isBoundTo(candidate))
    return pathFinderError("route costs are bound to another candidate");
  if (limits.endpointExpansionLimit == 0 || limits.iterationLimit == 0 ||
      limits.noProgressIterationLimit == 0 || limits.noProgressTrendWindow == 0)
    return pathFinderError("routing work limits must be positive");
  if (limits.noProgressTrendWindow > limits.noProgressIterationLimit ||
      limits.noProgressIterationLimit > limits.iterationLimit)
    return pathFinderError("routing no-progress limits are not canonical");
  if (limits.noProgressTrendWindow > std::numeric_limits<std::size_t>::max())
    return pathFinderError("routing trend window exceeds host size_t");
  const std::size_t trendWindow =
      static_cast<std::size_t>(limits.noProgressTrendWindow);
  if (rankTrendTransitions_.size() < trendWindow)
    rankTrendTransitions_.resize(trendWindow);
  if (!evaluationPriorities.empty() &&
      evaluationPriorities.size() !=
          candidate.problem().transfers().logicalNets().size())
    return pathFinderError("evaluation-priority vector has the wrong width");
  if (costs.selectedLogicalNet())
    return pathFinderError("route costs already have a selected logical net");

  auto initialProjection = move.projectCurrentRoutes();
  if (!initialProjection)
    return initialProjection.takeError();
  auto initialRegion = projectRoutingRegion(candidate, costs, logicalNets);
  if (!initialRegion)
    return initialRegion.takeError();
  if (initialProjection->routeTerminalsCompatible &&
      initialProjection->selectedHandshakeAcyclic &&
      initialRegion->unroutedObligationCount == 0 &&
      initialRegion->routeCapacityOveruse == 0)
    return SpatialPathFinderClosureResult{0, true};

  std::optional<dse::ObjectiveVector> bestRankObjective;
  std::optional<dse::ObjectiveVector> previousRankObjective;
  std::optional<dse::ObjectiveVector> bestTemporaryObjective;
  std::uint64_t consecutiveNoProgressIterations = 0;
  std::size_t trendHead = 0;
  std::size_t trendCount = 0;
  std::uint64_t trendImprovedCount = 0;
  std::uint64_t trendIneligibleCount = 0;
  std::uint64_t trendRegressedCount = 0;
  const std::uint64_t initialEndpointExpansions =
      netRouter_.endpointExpansionCount();
  loom::mapping_debug::MappingRunStatistics debugStatistics;
  const auto emitStatistics = [&](llvm::StringRef status) {
    debugStatistics.aStarExpansions =
        netRouter_.endpointExpansionCount() - initialEndpointExpansions;
    debugStatistics.emit(loom::mapping_debug::Stage::SpatialPnr, status);
  };

  for (std::uint64_t iteration = 0; iteration < limits.iterationLimit;
       ++iteration) {
    if (negotiationIterationCount_ ==
        std::numeric_limits<std::uint64_t>::max()) {
      ++debugStatistics.arithmeticFailures;
      loom::mapping_debug::emit(loom::mapping_debug::Level::Decision,
                                loom::mapping_debug::Stage::SpatialPnr,
                                loom::mapping_debug::Event::ArithmeticFailure,
                                [&](llvm::json::Object &fields) {
                                  fields["iteration"] = iteration;
                                  fields["operation"] =
                                      "negotiation_iteration_counter";
                                });
      emitStatistics("arithmetic_failure");
      return pathFinderError("negotiation iteration count overflows u64");
    }
    const std::uint64_t sessionIteration = negotiationIterationCount_;
    ++negotiationIterationCount_;
    ++debugStatistics.negotiatedIterations;
    if (llvm::Error error = buildCanonicalNetOrder(
            candidate, costs, logicalNets, evaluationPriorities))
      return std::move(error);

    constraintSweepNets_.clear();
    for (const NetOrderEntry &entry : netOrder_)
      constraintSweepNets_.push_back(entry.logicalNet);
    if (llvm::Error error =
            netRouter_.beginConstraintSweep(constraintSweepNets_))
      return std::move(error);

    for (const NetOrderEntry &entry : netOrder_) {
      auto projection = projectLogicalNet(candidate, costs, entry.logicalNet);
      if (!projection)
        return projection.takeError();
      if (llvm::Error error =
              costs.selectLogicalNet(entry.logicalNet, activeClaimBits_))
        return std::move(error);
      auto route =
          netRouter_.routeWholeNet(move, candidate, costs, entry.logicalNet,
                                   limits.endpointExpansionLimit);
      if (!route) {
        llvm::Error routeFailure = route.takeError();
        bool emittedTypedFailure = false;
        if (loom::mapping_debug::enabled(
                loom::mapping_debug::Level::Decision)) {
          routeFailure = llvm::handleErrors(
              std::move(routeFailure),
              [&](const EndpointRouteSearchFailure &failure) -> llvm::Error {
                emittedTypedFailure = true;
                std::string diagnostic = errorMessage(failure);
                loom::mapping_debug::emit(
                    loom::mapping_debug::Level::Decision,
                    loom::mapping_debug::Stage::SpatialPnr,
                    loom::mapping_debug::Event::MappingFailure,
                    [&](llvm::json::Object &fields) {
                      fields["iteration"] = iteration;
                      fields["session_iteration"] = sessionIteration;
                      fields["logical_net"] = entry.logicalNet;
                      fields["operation"] = "route_whole_net";
                      fields["failure_kind"] =
                          stringifyEndpointRouteSearchFailureKind(
                              failure.kind());
                      fields["diagnostic"] = diagnostic;
                    });
                return llvm::make_error<EndpointRouteSearchFailure>(
                    failure.kind(), std::move(diagnostic));
              });
        }
        if (!emittedTypedFailure)
          loom::mapping_debug::emit(loom::mapping_debug::Level::Decision,
                                    loom::mapping_debug::Stage::SpatialPnr,
                                    loom::mapping_debug::Event::MappingFailure,
                                    [&](llvm::json::Object &fields) {
                                      fields["iteration"] = iteration;
                                      fields["session_iteration"] =
                                          sessionIteration;
                                      fields["logical_net"] = entry.logicalNet;
                                      fields["operation"] = "route_whole_net";
                                    });
        emitStatistics("route_failure");
        return routeFailure;
      }
      loom::mapping_debug::emit(
          loom::mapping_debug::Level::Detail,
          loom::mapping_debug::Stage::SpatialPnr,
          loom::mapping_debug::Event::NetRoute,
          [&](llvm::json::Object &fields) {
            fields["iteration"] = iteration;
            fields["session_iteration"] = sessionIteration;
            fields["logical_net"] = entry.logicalNet;
            fields["route_cost"] = *route;
            fields["source_endpoint"] =
                candidate.logicalNetSourceEndpoint(entry.logicalNet);
            fields["sink_count"] = candidate.problem()
                                       .transfers()
                                       .logicalNets()[entry.logicalNet]
                                       .sinkCount;
          });
      if (llvm::Error error = costs.acceptSelectedLogicalNet())
        return std::move(error);
      if (llvm::Error error = netRouter_.finishConstraintNet(entry.logicalNet))
        return std::move(error);
    }

    const std::uint64_t completedIterations = iteration + 1;
    auto projection = move.projectCurrentRoutes();
    if (!projection)
      return projection.takeError();
    auto region = projectRoutingRegion(candidate, costs, logicalNets);
    if (!region)
      return region.takeError();
    const bool hasCapacityOveruse = region->routeCapacityOveruse != 0;
    if (logicalNets.empty() &&
        hasCapacityOveruse != (projection->routeCapacityOveruse != 0))
      return pathFinderError(
          "working route capacity disagrees with the provisional RouteTrees");
    CapacityConflictAnalysis conflictAnalysis;
    if (hasCapacityOveruse) {
      auto analyzed = analyzeCapacityConflicts(candidate, costs, iteration,
                                               sessionIteration);
      if (!analyzed)
        return analyzed.takeError();
      conflictAnalysis = *analyzed;
    }
    const std::uint64_t capacityConflicts = conflictAnalysis.conflictCount;
    debugStatistics.capacityConflicts += capacityConflicts;
    bool selectedRankImproved = false;
    bool temporaryAdmitted = false;
    bool mappingViolationsZero = false;
    const bool selectedRankEligible = projection->routeTerminalsCompatible &&
                                      projection->selectedHandshakeAcyclic;
    std::optional<RankTrendTransition> rankTrendTransition;
    if (selectedRankEligible) {
      auto disposition =
          classifyProjection(candidate, *projection, closureRequirement);
      if (!disposition)
        return disposition.takeError();
      temporaryAdmitted = disposition->temporaryAdmitted;
      mappingViolationsZero = disposition->violationsZero;
      auto objective =
          candidate.problem().objectiveProgram().evaluateSpatialProjection(
              candidate, *projection);
      if (!objective)
        return objective.takeError();
      selectedRankImproved = !bestRankObjective;
      if (bestRankObjective) {
        auto comparison =
            candidate.problem().objectiveProgram().compareSelectedRank(
                *objective, {}, *bestRankObjective, {});
        if (!comparison)
          return comparison.takeError();
        selectedRankImproved = *comparison < 0;
      }
      if (selectedRankImproved) {
        bestRankObjective = *objective;
        consecutiveNoProgressIterations = 0;
      } else {
        ++consecutiveNoProgressIterations;
      }

      if (previousRankObjective) {
        auto comparison =
            candidate.problem().objectiveProgram().compareSelectedRank(
                *objective, {}, *previousRankObjective, {});
        if (!comparison)
          return comparison.takeError();
        rankTrendTransition = *comparison < 0   ? RankTrendTransition::Improved
                              : *comparison > 0 ? RankTrendTransition::Regressed
                                                : RankTrendTransition::Equal;
      }
      previousRankObjective = *objective;

      if (temporaryAdmitted) {
        bool replace = !bestTemporaryObjective;
        if (bestTemporaryObjective) {
          auto comparison =
              candidate.problem().objectiveProgram().compareSelectedRank(
                  *objective, {}, *bestTemporaryObjective, {});
          if (!comparison)
            return comparison.takeError();
          replace = *comparison < 0;
        }
        if (replace) {
          if (llvm::Error error = captureCurrentRoutes(candidate, logicalNets))
            return std::move(error);
          bestTemporaryObjective = std::move(*objective);
        }
      }
    } else {
      ++consecutiveNoProgressIterations;
      rankTrendTransition = RankTrendTransition::Ineligible;
    }
    if (rankTrendTransition) {
      if (trendCount == trendWindow) {
        const auto evicted =
            static_cast<RankTrendTransition>(rankTrendTransitions_[trendHead]);
        trendImprovedCount -= evicted == RankTrendTransition::Improved ? 1 : 0;
        trendIneligibleCount -=
            evicted == RankTrendTransition::Ineligible ? 1 : 0;
        trendRegressedCount -=
            evicted == RankTrendTransition::Regressed ? 1 : 0;
      } else {
        ++trendCount;
      }
      rankTrendTransitions_[trendHead] =
          static_cast<std::uint8_t>(*rankTrendTransition);
      trendHead = (trendHead + 1) % trendWindow;
      trendImprovedCount +=
          *rankTrendTransition == RankTrendTransition::Improved ? 1 : 0;
      trendIneligibleCount +=
          *rankTrendTransition == RankTrendTransition::Ineligible ? 1 : 0;
      trendRegressedCount +=
          *rankTrendTransition == RankTrendTransition::Regressed ? 1 : 0;
    }
    loom::mapping_debug::emit(
        loom::mapping_debug::Level::Decision,
        loom::mapping_debug::Stage::SpatialPnr,
        loom::mapping_debug::Event::NegotiationIteration,
        [&](llvm::json::Object &fields) {
          fields["iteration"] = iteration;
          fields["session_iteration"] = sessionIteration;
          fields["logical_net_count"] = netOrder_.size();
          fields["capacity_conflicts"] = capacityConflicts;
          fields["capacity_conflict_events"] =
              conflictAnalysis.diagnosticConflictCount;
          fields["capacity_conflict_events_omitted"] =
              capacityConflicts - conflictAnalysis.diagnosticConflictCount;
          fields["capacity_closed"] = !hasCapacityOveruse;
          fields["mapping_closed"] = mappingViolationsZero;
          fields["route_terminals_compatible"] =
              projection->routeTerminalsCompatible;
          fields["selected_handshake_acyclic"] =
              projection->selectedHandshakeAcyclic;
          fields["unrouted_obligations"] = region->unroutedObligationCount;
          fields["route_capacity_overuse"] = region->routeCapacityOveruse;
          fields["tag_resident_capacity_overuse"] =
              projection->tagResidentCapacityOveruse;
          fields["tag_unassigned"] = projection->tagUnassignedCount;
          fields["tag_conflicts"] = projection->tagConflictCount;
          fields["hard_progress_violations"] =
              projection->hardProgressViolation;
          fields["selected_traversal_claim"] =
              projection->totalSelectedTraversalClaim;
          fields["temporary_admitted"] = temporaryAdmitted;
          fields["selected_rank_improved"] = selectedRankImproved;
          fields["consecutive_no_progress_iterations"] =
              consecutiveNoProgressIterations;
          fields["no_progress_iteration_limit"] =
              limits.noProgressIterationLimit;
          fields["no_progress_trend_window"] = limits.noProgressTrendWindow;
          fields["rank_trend_transition_count"] = trendCount;
          fields["rank_trend_improved_count"] = trendImprovedCount;
          fields["rank_trend_ineligible_count"] = trendIneligibleCount;
          fields["rank_trend_regressed_count"] = trendRegressedCount;
          fields["a_star_expansions"] =
              netRouter_.endpointExpansionCount() - initialEndpointExpansions;
        });
    const bool routingClosed = projection->routeTerminalsCompatible &&
                               projection->selectedHandshakeAcyclic &&
                               region->unroutedObligationCount == 0 &&
                               !hasCapacityOveruse;
    if (routingClosed) {
      emitStatistics("closed");
      return SpatialPathFinderClosureResult{completedIterations, true};
    }
    if (!hasCapacityOveruse) {
      if (bestTemporaryObjective) {
        if (llvm::Error error =
                restoreCapturedRoutes(move, candidate, costs, logicalNets))
          return std::move(error);
        emitStatistics("temporary_mapping");
        return SpatialPathFinderClosureResult{completedIterations, false};
      }
      emitStatistics(
          !projection->routeTerminalsCompatible  ? "route_terminal_mismatch"
          : projection->selectedHandshakeAcyclic ? "mapping_nonclosure"
                                                 : "selected_handshake_cycle");
      if (!projection->routeTerminalsCompatible)
        return pathFinderError(
            "provisional RouteTree terminals disagree with the candidate");
      if (!projection->selectedHandshakeAcyclic)
        return llvm::make_error<SpatialPathFinderClosureFailure>(
            SpatialPathFinderClosureFailure::Kind::
                SelectedCombinationalHandshakeCycle,
            "Spatial PathFinder selected a combinational handshake cycle");
      return llvm::make_error<SpatialPathFinderClosureFailure>(
          SpatialPathFinderClosureFailure::Kind::NonClosure,
          "Spatial PathFinder closed route capacity without Mapping closure");
    }
    if (conflictAnalysis.hasCertificate()) {
      loom::mapping_debug::emit(
          loom::mapping_debug::Level::Decision,
          loom::mapping_debug::Stage::SpatialPnr,
          loom::mapping_debug::Event::MappingFailure,
          [&](llvm::json::Object &fields) {
            fields["iteration"] = iteration;
            fields["session_iteration"] = sessionIteration;
            fields["operation"] = "fixed_terminal_capacity_cut";
            fields["capacity_ref"] = conflictAnalysis.certificateCapacity;
            fields["mandatory_usage"] = conflictAnalysis.mandatoryUsage;
            fields["capacity"] = conflictAnalysis.physicalCapacity;
            fields["temporary_return"] = bestTemporaryObjective.has_value();
          });
      if (bestTemporaryObjective) {
        if (llvm::Error error =
                restoreCapturedRoutes(move, candidate, costs, logicalNets))
          return std::move(error);
        emitStatistics("fixed_terminal_cut_temporary");
        return SpatialPathFinderClosureResult{completedIterations, false};
      }
      emitStatistics("fixed_terminal_cut");
      return llvm::make_error<SpatialPathFinderClosureFailure>(
          SpatialPathFinderClosureFailure::Kind::FixedTerminalCapacityCut,
          "Spatial PathFinder proved fixed-terminal capacity cut at capacity " +
              std::to_string(conflictAnalysis.certificateCapacity) +
              " with mandatory usage " +
              std::to_string(conflictAnalysis.mandatoryUsage) +
              " greater than capacity " +
              std::to_string(conflictAnalysis.physicalCapacity),
          conflictAnalysis.certificateCapacity, conflictAnalysis.mandatoryUsage,
          conflictAnalysis.physicalCapacity, cutCertificateForcedNets_);
    }
    if (consecutiveNoProgressIterations >= limits.noProgressIterationLimit &&
        trendCount == trendWindow &&
        trendImprovedCount <= trendRegressedCount + trendIneligibleCount) {
      loom::mapping_debug::emit(
          loom::mapping_debug::Level::Decision,
          loom::mapping_debug::Stage::SpatialPnr,
          loom::mapping_debug::Event::MappingFailure,
          [&](llvm::json::Object &fields) {
            fields["iteration"] = iteration;
            fields["session_iteration"] = sessionIteration;
            fields["operation"] = "selected_rank_no_progress";
            fields["consecutive_no_progress_iterations"] =
                consecutiveNoProgressIterations;
            fields["no_progress_iteration_limit"] =
                limits.noProgressIterationLimit;
            fields["no_progress_trend_window"] = limits.noProgressTrendWindow;
            fields["rank_trend_improved_count"] = trendImprovedCount;
            fields["rank_trend_ineligible_count"] = trendIneligibleCount;
            fields["rank_trend_regressed_count"] = trendRegressedCount;
            fields["temporary_return"] = bestTemporaryObjective.has_value();
          });
      if (bestTemporaryObjective) {
        if (llvm::Error error =
                restoreCapturedRoutes(move, candidate, costs, logicalNets))
          return std::move(error);
        emitStatistics("no_progress_temporary");
        return SpatialPathFinderClosureResult{completedIterations, false};
      }
      emitStatistics("no_progress");
      return llvm::make_error<SpatialPathFinderClosureFailure>(
          SpatialPathFinderClosureFailure::Kind::NoProgress,
          "Spatial PathFinder exhausted its selected-rank no-progress limit "
          "before capacity closure");
    }
    if (completedIterations == limits.iterationLimit) {
      if (bestTemporaryObjective) {
        if (llvm::Error error =
                restoreCapturedRoutes(move, candidate, costs, logicalNets))
          return std::move(error);
        emitStatistics("temporary_capacity");
        return SpatialPathFinderClosureResult{completedIterations, false};
      }
      emitStatistics("iteration_limit");
      return llvm::make_error<SpatialPathFinderClosureFailure>(
          SpatialPathFinderClosureFailure::Kind::NonClosure,
          "Spatial PathFinder exhausted its iteration limit before capacity "
          "closure");
    }
    if (llvm::Error error = costs.advancePathFinderIteration()) {
      ++debugStatistics.arithmeticFailures;
      loom::mapping_debug::emit(
          loom::mapping_debug::Level::Decision,
          loom::mapping_debug::Stage::SpatialPnr,
          loom::mapping_debug::Event::ArithmeticFailure,
          [&](llvm::json::Object &fields) {
            fields["iteration"] = iteration;
            fields["session_iteration"] = sessionIteration;
            fields["operation"] = "advance_pathfinder_iteration";
            fields["capacity_conflicts"] = capacityConflicts;
          });
      emitStatistics("arithmetic_failure");
      return std::move(error);
    }
  }
  llvm_unreachable("positive iteration limit executes or returns");
}

llvm::Expected<RouteCost> SpatialPathFinderRouterScratch::routeWholeNetInMove(
    SpatialMoveTransaction &move, const SpatialCandidateState &candidate,
    SpatialRouteCostState &costs, PnrIndex logicalNet,
    std::uint64_t endpointExpansionLimit) {
  if (!preparedProblem_ || preparedProblem_ != &candidate.problem())
    return pathFinderError("scratch is not prepared for the candidate freeze");
  if (!costs.isBoundTo(candidate))
    return pathFinderError("route costs are bound to another candidate");
  return netRouter_.routeWholeNet(move, candidate, costs, logicalNet,
                                  endpointExpansionLimit);
}

llvm::Expected<RouteCost> SpatialPathFinderRouterScratch::routeSingleSinkInMove(
    SpatialMoveTransaction &move, const SpatialCandidateState &candidate,
    SpatialRouteCostState &costs, PnrIndex logicalNet, PnrIndex sinkObligation,
    std::uint64_t endpointExpansionLimit) {
  if (!preparedProblem_ || preparedProblem_ != &candidate.problem())
    return pathFinderError("scratch is not prepared for the candidate freeze");
  if (!costs.isBoundTo(candidate))
    return pathFinderError("route costs are bound to another candidate");
  return netRouter_.routeSingleSink(move, candidate, costs, logicalNet,
                                    sinkObligation, endpointExpansionLimit);
}

llvm::Expected<RouteCost>
SpatialPathFinderRouterScratch::routeRootedSubtreeInMove(
    SpatialMoveTransaction &move, const SpatialCandidateState &candidate,
    SpatialRouteCostState &costs, PnrIndex logicalNet, PnrIndex rootEndpoint,
    std::uint64_t endpointExpansionLimit) {
  if (!preparedProblem_ || preparedProblem_ != &candidate.problem())
    return pathFinderError("scratch is not prepared for the candidate freeze");
  if (!costs.isBoundTo(candidate))
    return pathFinderError("route costs are bound to another candidate");
  return netRouter_.routeRootedSubtree(move, candidate, costs, logicalNet,
                                       rootEndpoint, endpointExpansionLimit);
}

llvm::Error SpatialPathFinderRouterScratch::beginConstraintSweep(
    llvm::ArrayRef<PnrIndex> logicalNets) {
  return netRouter_.beginConstraintSweep(logicalNets);
}

llvm::Error
SpatialPathFinderRouterScratch::finishConstraintNet(PnrIndex logicalNet) {
  return netRouter_.finishConstraintNet(logicalNet);
}

std::size_t SpatialPathFinderRouterScratch::retainedStorageBytes() const {
  return netRouter_.retainedStorageBytes() + retainedBytes(netOrder_) +
         retainedBytes(activeClaimBits_) + retainedBytes(claimEpochs_) +
         retainedBytes(capacityEpochs_) + retainedBytes(capacityNetQCosts_) +
         retainedBytes(touchedCapacities_) +
         retainedBytes(regionalCapacityMarks_) +
         retainedBytes(constraintSweepNets_) +
         retainedBytes(capturedSinkPathOffsets_) +
         retainedBytes(capturedForwardArcs_) + retainedBytes(reversePath_) +
         retainedBytes(cutBlockedTraversals_) +
         retainedBytes(cutReachableEndpoints_) +
         retainedBytes(cutSeenTraversals_) + retainedBytes(cutSeenEndpoints_) +
         retainedBytes(cutWorklist_) + retainedBytes(cutContributingNets_) +
         retainedBytes(cutForcedNets_) +
         retainedBytes(cutCertificateForcedNets_) +
         retainedBytes(cutPayloadWidths_) + retainedBytes(cutMinimumClaims_) +
         retainedBytes(rankTrendTransitions_) +
         retainedBytes(cutTouchedClaims_) +
         retainedBytes(cutNetClaimRefcounts_) +
         retainedBytes(cutClaimSelectionCounts_) +
         retainedBytes(cutClaimTraversalRefcounts_);
}
