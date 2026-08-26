#include "SpatialActiveProblemStatistics.h"

#include "llvm/ADT/StringExtras.h"

#include <cstdint>
#include <limits>

namespace loom::pnr {
namespace {

template <typename T>
void addTrackedArray(std::uint64_t &bytes, std::uint64_t &work,
                     llvm::ArrayRef<T> values) {
  const std::uint64_t count = values.size();
  const std::uint64_t elementBytes = sizeof(T);
  if (count >
      (std::numeric_limits<std::uint64_t>::max() - bytes) / elementBytes)
    bytes = std::numeric_limits<std::uint64_t>::max();
  else
    bytes += count * elementBytes;
  if (count > std::numeric_limits<std::uint64_t>::max() - work)
    work = std::numeric_limits<std::uint64_t>::max();
  else
    work += count;
}

void saturatingAdd(std::uint64_t &value, std::uint64_t added) {
  value = added > std::numeric_limits<std::uint64_t>::max() - value
              ? std::numeric_limits<std::uint64_t>::max()
              : value + added;
}

} // namespace

SpatialActiveProblemStatistics buildSpatialActiveProblemStatistics(
    const FrozenSpatialRealizationIndex &realizations,
    const FrozenSpatialMemoryIndex &memory,
    const FrozenSpatialTransferIndex &transfers,
    const FrozenSpatialLocalTransferIndex &localTransfers,
    const FrozenSpatialPortIndex &ports,
    const FrozenSpatialCapacityIndex &capacity,
    const FrozenSpatialActiveRoutingDomain &activeRouting,
    const FrozenSpatialHandshakeIndex &handshake,
    std::uint64_t constructionNanoseconds) {
  SpatialActiveProblemStatistics result;
  result.context.constructionCount = 1;
  result.context.constructionNanoseconds = constructionNanoseconds;
  result.computeRealizationCount = realizations.computeRealizations().size();
  result.computePlacementCount = realizations.computePlacements().size();
  result.memoryRealizationCount = realizations.memoryRealizations().size();
  result.memoryPlacementCount = realizations.memoryPlacements().size();
  result.logicalNetCount = transfers.logicalNets().size();
  result.logicalSinkCount = transfers.logicalNetSinks().size();
  result.localTransferOptionCount = localTransfers.options().size();
  result.portDemandCount = ports.portDemands().size();
  result.attachmentOptionCount = ports.attachmentOptions().size();
  result.operandPairingGroupCount = ports.operandPairingGroups().size();
  result.operandPairingMemberCount = ports.operandPairingGroupMembers().size();
  result.computeAttachmentClassLookupCount =
      ports.computeAttachmentClassLookupCount();
  result.computeAttachmentClassHitCount =
      ports.computeAttachmentClassHitCount();
  result.computeAttachmentClassMissCount =
      ports.computeAttachmentClassMissCount();
  result.activeEndpointCount = activeRouting.activeEndpointCount();
  result.activeTraversalCount = activeRouting.activeTraversalCount();
  result.activeRoutingArcCount = activeRouting.activeArcCount();
  result.handshakeOwnerCount = handshake.ownerModels().size();
  result.handshakePotentialFragmentCount = handshake.fragments().size();
  for (const FrozenSpatialHandshakeFragment &fragment : handshake.fragments())
    saturatingAdd(result.handshakePotentialContributionCount,
                  fragment.contributionCount);

  std::uint64_t &bytes = result.context.retainedBytes;
  std::uint64_t &work = result.context.deterministicWork;
  addTrackedArray(bytes, work, realizations.computeRealizations());
  addTrackedArray(bytes, work, realizations.computeActors());
  addTrackedArray(bytes, work, realizations.computeActorRealizations());
  addTrackedArray(bytes, work, realizations.computePlacements());
  addTrackedArray(bytes, work, realizations.computeInstructionContexts());
  addTrackedArray(bytes, work, realizations.memoryRealizations());
  addTrackedArray(bytes, work, realizations.memoryActors());
  addTrackedArray(bytes, work, realizations.memoryActorRealizations());
  addTrackedArray(bytes, work, realizations.memoryPlacements());
  addTrackedArray(bytes, work, memory.logicalBindings());
  addTrackedArray(bytes, work, memory.bindingTargets());
  addTrackedArray(bytes, work, memory.rootedUses());
  addTrackedArray(bytes, work, memory.serviceUseGroups());
  addTrackedArray(bytes, work, memory.exposures());
  addTrackedArray(bytes, work, memory.exposureProviders());
  addTrackedArray(bytes, work, memory.exposureOptions());
  addTrackedArray(bytes, work, memory.dispatchDomains());
  addTrackedArray(bytes, work, memory.dispatchOptions());
  addTrackedArray(bytes, work, transfers.logicalNets());
  addTrackedArray(bytes, work, transfers.logicalNetSinks());
  addTrackedArray(bytes, work, transfers.logicalNetSourceBindings());
  addTrackedArray(bytes, work, transfers.logicalNetSinkBindings());
  addTrackedArray(bytes, work, localTransfers.domains());
  addTrackedArray(bytes, work, localTransfers.options());
  addTrackedArray(bytes, work, ports.portDemands());
  addTrackedArray(bytes, work, ports.placementDomains());
  addTrackedArray(bytes, work, ports.attachmentOptions());
  addTrackedArray(bytes, work, ports.graphBoundaries());
  addTrackedArray(bytes, work, ports.operandPairingGroups());
  addTrackedArray(bytes, work, ports.operandPairingGroupMembers());
  addTrackedArray(bytes, work, ports.demandOperandPairingOffsets());
  addTrackedArray(bytes, work, ports.demandOperandPairingGroups());
  saturatingAdd(work, result.computeAttachmentClassLookupCount);
  addTrackedArray(bytes, work, capacity.resourceEvents());
  addTrackedArray(bytes, work, capacity.resourceUses());
  addTrackedArray(bytes, work, capacity.resourceTimeEnvelopes());
  addTrackedArray(bytes, work, capacity.resourceTimeSegments());
  addTrackedArray(bytes, work, handshake.ownerModels());
  addTrackedArray(bytes, work, handshake.fragments());
  addTrackedArray(bytes, work, handshake.traversalFragments());
  addTrackedArray(bytes, work, handshake.computePlacementFragments());
  addTrackedArray(bytes, work, handshake.memoryOperationDomains());
  addTrackedArray(bytes, work, handshake.memoryOperationPlans());
  addTrackedArray(bytes, work, handshake.memoryPlanFragments());
  addTrackedArray(bytes, work, handshake.projectionArcs());
  addTrackedArray(bytes, work, handshake.projectionFixedArcs());
  addTrackedArray(bytes, work, handshake.projectionFragmentArcOffsets());
  addTrackedArray(bytes, work, handshake.projectionFragmentArcs());
  addTrackedArray(bytes, work, handshake.projectionOutgoingArcOffsets());
  saturatingAdd(work, result.handshakePotentialContributionCount);
  saturatingAdd(bytes, activeRouting.retainedBytes());
  saturatingAdd(work, activeRouting.deterministicWork());
  return result;
}

void emitSpatialActiveProblemStatistics(const FrozenSpatialPnrProblem &problem,
                                        mapping_debug::Stage stage,
                                        std::uint64_t hits,
                                        std::uint64_t misses) {
  const SpatialActiveProblemStatistics &statistics = problem.statistics();
  mapping_debug::emit(
      mapping_debug::Level::Summary, stage,
      mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
        fields["context_kind"] = "spatial_active";
        fields["context_key"] = llvm::toHex(problem.cacheKey().bytes(),
                                             /*LowerCase=*/true);
        fields["cache_hits"] = hits;
        fields["cache_misses"] = misses;
        fields["construction_count"] = statistics.context.constructionCount;
        fields["construction_time_ns"] =
            statistics.context.constructionNanoseconds;
        fields["retained_bytes"] = statistics.context.retainedBytes;
        fields["deterministic_work"] = statistics.context.deterministicWork;
        fields["compute_realization_count"] =
            statistics.computeRealizationCount;
        fields["compute_placement_count"] = statistics.computePlacementCount;
        fields["memory_realization_count"] = statistics.memoryRealizationCount;
        fields["memory_placement_count"] = statistics.memoryPlacementCount;
        fields["logical_net_count"] = statistics.logicalNetCount;
        fields["logical_sink_count"] = statistics.logicalSinkCount;
        fields["local_transfer_option_count"] =
            statistics.localTransferOptionCount;
        fields["port_demand_count"] = statistics.portDemandCount;
        fields["attachment_option_count"] = statistics.attachmentOptionCount;
        fields["operand_pairing_group_count"] =
            statistics.operandPairingGroupCount;
        fields["operand_pairing_member_count"] =
            statistics.operandPairingMemberCount;
        fields["compute_attachment_class_lookups"] =
            statistics.computeAttachmentClassLookupCount;
        fields["compute_attachment_class_hits"] =
            statistics.computeAttachmentClassHitCount;
        fields["compute_attachment_class_misses"] =
            statistics.computeAttachmentClassMissCount;
        fields["active_endpoint_count"] = statistics.activeEndpointCount;
        fields["active_traversal_count"] = statistics.activeTraversalCount;
        fields["active_routing_arc_count"] = statistics.activeRoutingArcCount;
        fields["handshake_owner_count"] = statistics.handshakeOwnerCount;
        fields["handshake_potential_fragment_count"] =
            statistics.handshakePotentialFragmentCount;
        fields["handshake_potential_contribution_count"] =
            statistics.handshakePotentialContributionCount;
      });
}

} // namespace loom::pnr
