#include "PnR/SpatialPathFinderRouter.h"

#include "Common/MappingDebugLog.h"
#include "Fabric/Identity/FabricRefText.h"
#include "SpatialPathFinderRouterInternal.h"
#include "SpatialPhysicalTiming.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

using namespace loom::pnr;
using loom::pnr::detail::encodeLogicalNetDetail;
using loom::pnr::detail::encodeSelectedOrdinalRanges;
using loom::pnr::detail::pathFinderError;
using loom::pnr::detail::resourceOwnerForState;
using loom::pnr::detail::resourceStateForCapacity;

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
  cutCertificateForcedNetCuts_.clear();

  conflictCapacities_.clear();
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
    conflictCapacities_.push_back(capacity);
  }

  for (PnrIndex capacity : conflictCapacities_) {
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
    if (emitDecisions) {
      std::fill(cutSeenTraversals_.begin(), cutSeenTraversals_.end(), 0);
      std::fill(cutSeenEndpoints_.begin(), cutSeenEndpoints_.end(), 0);
    }
    cutContributingNets_.clear();
    cutForcedNetCuts_.clear();
    std::uint64_t derivedUsage = capacityRecord.initialOccupancy;
    for (PnrIndex logicalNet = 0;
         logicalNet < problem.transfers().logicalNets().size(); ++logicalNet) {
      cutTouchedClaims_.clear();
      const auto accountTraversal =
          [&](PnrIndex traversal,
              std::optional<std::pair<PnrIndex, PnrIndex>> endpoints)
          -> llvm::Error {
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
          if (endpoints) {
            cutSeenEndpoints_[endpoints->first] = 1;
            cutSeenEndpoints_[endpoints->second] = 1;
          }
        }
        return llvm::Error::success();
      };
      if (candidate.usesRegisterFifo(logicalNet)) {
        const PnrIndex selected = candidate.registerFifoTransfer(logicalNet);
        if (selected >= problem.localTransfers().options().size())
          return pathFinderError(
              "capacity analysis register-FIFO option is out of range");
        const auto &option = problem.localTransfers().options()[selected];
        if (llvm::Error error =
                accountTraversal(option.writeTraversal, std::nullopt))
          return std::move(error);
        if (llvm::Error error =
                accountTraversal(option.readTraversal, std::nullopt))
          return std::move(error);
      } else {
        const RouteTreeState &tree = candidate.routeTree(logicalNet);
        for (const RouteTreeNode &node : tree.nodeStorage()) {
          if (!node.isActive() || node.parentArc == getInvalidPnrIndex())
            continue;
          if (node.parentArc >= arcs.size() ||
              node.parentArc >= arcSources.size())
            return pathFinderError(
                "capacity analysis RouteTree arc is out of range");
          const PnrIndex traversal = arcs[node.parentArc].traversal;
          if (llvm::Error error = accountTraversal(
                  traversal, std::pair(arcSources[node.parentArc],
                                       arcs[node.parentArc].target)))
            return std::move(error);
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
      if (candidate.usesRegisterFifo(logicalNet))
        continue;
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
          if (!problem.activeRouting().arcIsActive(arc) ||
              cutBlockedTraversals_[record.traversal] ||
              record.payloadCapacityBits < payloadWidth ||
              cutReachableEndpoints_[record.target])
            continue;
          cutReachableEndpoints_[record.target] = 1;
          cutWorklist_.push_back(record.target);
        }
      }

      std::optional<PnrIndex> unreachableSink;
      const PnrIndex sinkCount =
          problem.transfers().logicalNets()[logicalNet].sinkCount;
      for (PnrIndex sink = 0; sink < sinkCount; ++sink) {
        const PnrIndex endpoint =
            candidate.logicalNetSinkEndpoint(logicalNet, sink);
        if (endpoint >= cutReachableEndpoints_.size())
          return pathFinderError("cut sink endpoint is out of range");
        if (!cutReachableEndpoints_[endpoint] && !unreachableSink)
          unreachableSink = sink;
      }
      if (unreachableSink) {
        if (minimumClaim == std::numeric_limits<std::uint64_t>::max())
          return pathFinderError(
              "separating capacity cut has no positive compatible claim");
        if (minimumClaim >
            std::numeric_limits<std::uint64_t>::max() - mandatoryUsage)
          return pathFinderError("mandatory capacity usage exceeds u64");
        mandatoryUsage += minimumClaim;
        ++forcedNetCount;
        cutForcedNetCuts_.push_back({logicalNet, *unreachableSink});
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
              fields["separating_cut"] = unreachableSink.has_value();
              if (unreachableSink)
                fields["certificate_sink"] = *unreachableSink;
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
      cutCertificateForcedNetCuts_ = cutForcedNetCuts_;
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
      llvm::json::Array logicalNetDetails;
      llvm::json::Array traversalDetails;
      llvm::json::Array endpointDetails;
      if (emitDetails || certificate)
        for (PnrIndex logicalNet : cutContributingNets_)
          logicalNetDetails.push_back(
              encodeLogicalNetDetail(candidate, logicalNet));
      std::uint64_t traversalCount = 0;
      std::uint64_t endpointCount = 0;
      for (PnrIndex traversal = 0; traversal < cutSeenTraversals_.size();
           ++traversal)
        if (cutSeenTraversals_[traversal]) {
          ++traversalCount;
          if (traversals.size() < sampleLimit) {
            traversals.push_back(traversal);
            if (emitDetails || certificate) {
              llvm::json::Object row;
              row["ref"] = loom::fabric::printFabricRef(
                  routing.traversals()[traversal].reference);
              row["traversal"] = traversal;
              traversalDetails.push_back(std::move(row));
            }
          }
        }
      for (PnrIndex endpoint = 0; endpoint < cutSeenEndpoints_.size();
           ++endpoint)
        if (cutSeenEndpoints_[endpoint]) {
          ++endpointCount;
          if (endpoints.size() < sampleLimit) {
            endpoints.push_back(endpoint);
            if (emitDetails || certificate) {
              llvm::json::Object row;
              row["endpoint"] = endpoint;
              row["ref"] = loom::fabric::printFabricRef(
                  routing.routingEndpoints()[endpoint].reference);
              endpointDetails.push_back(std::move(row));
            }
          }
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
              fields["logical_net_details"] = std::move(logicalNetDetails);
              fields["claims"] = std::move(claims);
              fields["traversals"] = std::move(traversals);
              fields["traversal_details"] = std::move(traversalDetails);
              fields["endpoints"] = std::move(endpoints);
              fields["endpoint_details"] = std::move(endpointDetails);
            } else {
              fields["logical_net_sample"] = std::move(logicalNets);
              fields["claim_sample"] = std::move(claims);
              fields["traversal_sample"] = std::move(traversals);
              fields["endpoint_sample"] = std::move(endpoints);
              if (certificate) {
                fields["certificate_logical_net_details"] =
                    std::move(logicalNetDetails);
                fields["certificate_traversal_details"] =
                    std::move(traversalDetails);
                fields["certificate_endpoint_details"] =
                    std::move(endpointDetails);
                llvm::json::Array forcedCuts;
                for (const SpatialFixedTerminalCutNet &cut :
                     cutCertificateForcedNetCuts_) {
                  llvm::json::Object row;
                  row["logical_net"] = cut.logicalNet;
                  row["unreachable_sink"] = cut.unreachableSink;
                  forcedCuts.push_back(std::move(row));
                }
                fields["forced_net_cuts"] = std::move(forcedCuts);
              }
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

llvm::Expected<SpatialPathFinderRouterScratch::RoutingRegionState>
SpatialPathFinderRouterScratch::projectRoutingRegion(
    const SpatialCandidateState &candidate, const SpatialRouteCostState &costs,
    llvm::ArrayRef<PnrIndex> logicalNets) {
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  const auto &routing = problem.routing();
  const auto &resources = problem.resources();
  std::fill(regionalCapacityMarks_.begin(), regionalCapacityMarks_.end(), 0);
  std::fill(regionalTagDomainMarks_.begin(), regionalTagDomainMarks_.end(), 0);

  RoutingRegionState projection;
  const auto projectNet = [&](PnrIndex logicalNet) -> llvm::Error {
    if (logicalNet >= problem.transfers().logicalNets().size())
      return pathFinderError("routing region contains a foreign logical net");
    if (candidate.usesRegisterFifo(logicalNet)) {
      const PnrIndex selected = candidate.registerFifoTransfer(logicalNet);
      if (selected >= problem.localTransfers().options().size())
        return pathFinderError(
            "routing region has a foreign register-FIFO transfer");
      const auto &option = problem.localTransfers().options()[selected];
      const std::array<PnrIndex, 2> traversals = {option.writeTraversal,
                                                  option.readTraversal};
      for (PnrIndex traversal : traversals) {
        if (traversal >= routing.traversals().size())
          return pathFinderError("register-FIFO traversal is out of range");
        const FrozenSpatialTraversal &record = routing.traversals()[traversal];
        for (PnrIndex claim : routing.traversalClaimKeys().slice(
                 record.routeClaimOffset, record.routeClaimCount)) {
          if (claim >= routing.routeClaims().size())
            return pathFinderError("register-FIFO route claim is out of range");
          const PnrIndex capacity =
              routing.routeClaims()[claim].capacityDimension;
          if (capacity >= regionalCapacityMarks_.size())
            return pathFinderError(
                "register-FIFO route capacity is out of range");
          regionalCapacityMarks_[capacity] = 1;
        }
      }
      return llvm::Error::success();
    }
    const std::uint64_t unassigned =
        costs.logicalNetTagUnassignedCount(logicalNet);
    if (unassigned > std::numeric_limits<std::uint64_t>::max() -
                         projection.tagUnassignedCount)
      return pathFinderError("regional unassigned tag count overflows u64");
    projection.tagUnassignedCount += unassigned;
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
    for (const SpatialTagDomainUse &use :
         costs.logicalNetTagDomainUses(logicalNet)) {
      if (use.domain >= regionalTagDomainMarks_.size())
        return pathFinderError("regional tag domain is out of range");
      regionalTagDomainMarks_[use.domain] = 1;
    }
    return llvm::Error::success();
  };

  if (logicalNets.empty()) {
    std::fill(regionalCapacityMarks_.begin(), regionalCapacityMarks_.end(), 1);
    std::fill(regionalTagDomainMarks_.begin(), regionalTagDomainMarks_.end(),
              1);
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
  for (PnrIndex domain = 0; domain < regionalTagDomainMarks_.size(); ++domain) {
    if (!regionalTagDomainMarks_[domain])
      continue;
    const std::uint64_t resident = costs.tagDomainResidentOveruse(domain);
    const std::uint64_t conflicts = costs.tagDomainConflictCount(domain);
    if (resident > std::numeric_limits<std::uint64_t>::max() -
                       projection.tagResidentCapacityOveruse ||
        conflicts > std::numeric_limits<std::uint64_t>::max() -
                        projection.tagConflictCount)
      return pathFinderError("regional tag pressure overflows u64");
    projection.tagResidentCapacityOveruse += resident;
    projection.tagConflictCount += conflicts;
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
  if (tree.isUnrouted() && !candidate.usesRegisterFifo(logicalNet))
    return NetProjection{0, 0};

  const FrozenSpatialRoutingGraph &routing = candidate.problem().routing();
  const auto projectTraversal = [&](PnrIndex traversal) -> llvm::Error {
    if (traversal >= routing.traversals().size())
      return pathFinderError("selected traversal is out of range");
    const FrozenSpatialTraversal &record = routing.traversals()[traversal];
    for (PnrIndex claim : routing.traversalClaimKeys().slice(
             record.routeClaimOffset, record.routeClaimCount)) {
      if (claim >= routing.routeClaims().size())
        return pathFinderError("selected traversal claim is out of range");
      if (claimEpochs_[claim] == projectionEpoch_)
        continue;
      claimEpochs_[claim] = projectionEpoch_;
      activeClaimBits_[claim / 64] |= std::uint64_t{1} << (claim % 64);

      const FrozenSpatialRouteClaim &claimRecord = routing.routeClaims()[claim];
      const PnrIndex capacity = claimRecord.capacityDimension;
      if (capacity >= capacityNetQCosts_.size())
        return pathFinderError(
            "selected traversal claim capacity is out of range");
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
    return llvm::Error::success();
  };
  if (candidate.usesRegisterFifo(logicalNet)) {
    const PnrIndex selected = candidate.registerFifoTransfer(logicalNet);
    if (selected >= candidate.problem().localTransfers().options().size())
      return pathFinderError("register-FIFO projection option is out of range");
    const auto &option =
        candidate.problem().localTransfers().options()[selected];
    if (llvm::Error error = projectTraversal(option.writeTraversal))
      return std::move(error);
    if (llvm::Error error = projectTraversal(option.readTraversal))
      return std::move(error);
  } else {
    for (const RouteTreeNode &node : tree.nodeStorage()) {
      if (!node.isActive() || node.parentArc == getInvalidPnrIndex())
        continue;
      if (node.parentArc >= routing.routingArcs().size())
        return pathFinderError("RouteTree parent arc is out of range");
      if (llvm::Error error =
              projectTraversal(routing.routingArcs()[node.parentArc].traversal))
        return std::move(error);
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
  if (costs.logicalNetHasTagPressure(logicalNet)) {
    contributesToViolation = true;
    auto next = accumulateRouteCost(conflictPressure,
                                    costs.logicalNetTagPressure(logicalNet));
    if (!next)
      return next.takeError();
    conflictPressure = *next;
  }
  auto timing = detail::projectSpatialLogicalNetPhysicalTiming(
      candidate.problem(), logicalNet, tree,
      candidate.registerFifoTransfer(logicalNet),
      candidate.portAttachmentSelections(),
      candidate.graphBoundaryAttachmentSelections(), &timingRouteNodeArrivals_,
      &timingRouteNodeWorklist_);
  if (!timing)
    return timing.takeError();
  const unsigned __int128 criticalDelay =
      (static_cast<unsigned __int128>(timing->worstArrivalDelayQuanta) + 1) *
      (static_cast<unsigned __int128>(timing->structuralCriticality) + 1);
  if (criticalDelay > std::numeric_limits<std::uint64_t>::max())
    return pathFinderError("physical net criticality exceeds u64");
  return NetProjection{
      static_cast<std::uint8_t>(contributesToViolation ? 1 : 2),
      conflictPressure, timing->totalNegativeSlackQuanta,
      static_cast<std::uint64_t>(criticalDelay)};
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
    if (candidate.usesRegisterFifo(logicalNet))
      return llvm::Error::success();
    auto projection = projectLogicalNet(candidate, costs, logicalNet);
    if (!projection)
      return projection.takeError();
    netOrder_.push_back(
        {projection->routeStateRank, projection->conflictPressure,
         projection->physicalNegativeSlack, projection->physicalCriticalDelay,
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
    if (lhs.physicalNegativeSlack != rhs.physicalNegativeSlack)
      return lhs.physicalNegativeSlack > rhs.physicalNegativeSlack;
    if (lhs.physicalCriticalDelay != rhs.physicalCriticalDelay)
      return lhs.physicalCriticalDelay > rhs.physicalCriticalDelay;
    if (lhs.evaluationPriority != rhs.evaluationPriority)
      return lhs.evaluationPriority > rhs.evaluationPriority;
    return lhs.logicalNet < rhs.logicalNet;
  });
  return llvm::Error::success();
}
