#include "FabricTopologyQualityDiagnostic.h"

#include "Common/ArtifactText.h"
#include "Fabric/IR/FabricEnums.h"
#include "Fabric/Identity/FabricRefText.h"

#include "llvm/Support/JSON.h"

#include <cstdint>

namespace loom::pnr {
namespace {

llvm::json::Array
ownerArray(llvm::ArrayRef<fabric::FabricTransportEndpointOwnerRef> owners) {
  llvm::json::Array result;
  for (const auto &owner : owners)
    result.emplace_back(fabric::printFabricRef(owner));
  return result;
}

llvm::json::Array
peArray(llvm::ArrayRef<fabric::FabricPeOccurrenceRef> pes) {
  llvm::json::Array result;
  for (fabric::FabricPeOccurrenceRef pe : pes)
    result.emplace_back(fabric::printFabricRef(pe));
  return result;
}

void emitHopDistribution(
    llvm::json::Object &fields, llvm::StringRef prefix,
    const fabric::FabricTopologyHopDistribution &distribution) {
  fields[(prefix + "_subject_count").str()] = distribution.subjectCount;
  fields[(prefix + "_reachable_subject_count").str()] =
      distribution.reachableSubjectCount;
  fields[(prefix + "_total_reachable_hops").str()] =
      distribution.totalReachableHops;
  fields[(prefix + "_unreachable_subjects").str()] =
      peArray(distribution.unreachableSubjects);
  if (distribution.minimum) {
    fields[(prefix + "_minimum_hops").str()] = distribution.minimum->hops;
    fields[(prefix + "_minimum_subjects").str()] =
        peArray(distribution.minimum->subjects);
  }
  if (distribution.maximum) {
    fields[(prefix + "_maximum_hops").str()] = distribution.maximum->hops;
    fields[(prefix + "_maximum_subjects").str()] =
        peArray(distribution.maximum->subjects);
  }
}

} // namespace

llvm::Expected<fabric::FabricTopologyQualityReport>
analyzeAndEmitFabricTopologyQuality(const fabric::FabricArtifactView &fabric,
                                    mapping_debug::Stage stage) {
  auto report = fabric::analyzeFabricTopologyQuality(fabric);
  if (!report)
    return report.takeError();

  std::uint64_t portCount = 0;
  std::uint64_t routingResourceIncidences = 0;
  std::uint64_t directResourceIncidences = 0;
  std::uint64_t boundaryPortCount = 0;
  std::uint64_t unreachablePortCount = 0;
  std::uint64_t directBindingOwners = 0;
  for (const fabric::FabricTopologyOwnerQuality &owner : report->owners) {
    portCount += owner.portCount();
    routingResourceIncidences += owner.routingResourceCount();
    directResourceIncidences += owner.directResourceCount();
    boundaryPortCount += owner.boundaryPortCount;
    unreachablePortCount += owner.unreachablePortCount;
    directBindingOwners += owner.directResourceCount() != 0;
  }

  const std::string artifact = formatArtifactIdentityHex(report->artifact);
  const fabric::FabricTopologyDseQuality dseQuality =
      fabric::projectFabricTopologyDseQuality(*report);
  mapping_debug::emit(
      mapping_debug::Level::Summary, stage,
      mapping_debug::Event::TopologyQuality, [&](llvm::json::Object &fields) {
        fields["scope"] = "root";
        fields["artifact"] = artifact;
        fields["root_kind"] = fabric::fabricRefKeyword(report->rootKind);
        fields["owner_count"] = report->owners.size();
        fields["port_count"] = portCount;
        fields["routing_resource_incidences"] = routingResourceIncidences;
        fields["direct_resource_incidences"] = directResourceIncidences;
        fields["boundary_port_count"] = boundaryPortCount;
        fields["unreachable_port_count"] = unreachablePortCount;
        fields["direct_binding_owner_count"] = directBindingOwners;
        fields["unscheduled_memory_count"] = report->unscheduledMemoryCount;
        fields["schedule_distribution_count"] = report->schedules.size();
        fields["capability_distribution_count"] =
            report->capabilities.size();
        fields["schedule_supply_gap"] = dseQuality.scheduleSupplyGap;
        fields["matching_memory_unreachable_pe_count"] =
            dseQuality.matchingMemoryUnreachablePeCount;
        fields["matching_memory_total_reachable_hops"] =
            dseQuality.matchingMemoryTotalReachableHops;
        fields["capability_coverage_unreachable_pe_count"] =
            dseQuality.capabilityCoverageUnreachablePeCount;
        fields["capability_coverage_total_reachable_hops"] =
            dseQuality.capabilityCoverageTotalReachableHops;
        fields["isolated_capability_supporting_pe_count"] =
            dseQuality.isolatedCapabilitySupportingPeCount;
      });

  const auto emitRatio = [](llvm::json::Object &fields, llvm::StringRef prefix,
                            const auto &extreme) {
    if (!extreme)
      return;
    fields[(prefix + "_numerator").str()] = extreme->numerator;
    fields[(prefix + "_denominator").str()] = extreme->denominator;
    fields[(prefix + "_owners").str()] = ownerArray(extreme->owners);
  };
  for (const fabric::FabricTopologyKindDistribution &distribution :
       fabric::summarizeFabricTopologyQuality(*report)) {
    mapping_debug::emit(
        mapping_debug::Level::Summary, stage,
        mapping_debug::Event::TopologyQuality, [&](llvm::json::Object &fields) {
          fields["scope"] = "terminal_distribution";
          fields["artifact"] = artifact;
          fields["terminal_kind"] =
              fabric::fabricTopologyTerminalKindSpelling(distribution.kind);
          fields["owner_count"] = distribution.ownerCount;
          fields["zero_port_owner_count"] = distribution.zeroPortOwnerCount;
          fields["minimum_port_count"] = distribution.minimumPortCount.value;
          fields["minimum_port_count_owners"] =
              ownerArray(distribution.minimumPortCount.owners);
          fields["maximum_port_count"] = distribution.maximumPortCount.value;
          fields["maximum_port_count_owners"] =
              ownerArray(distribution.maximumPortCount.owners);
          emitRatio(fields, "minimum_routing_ratio",
                    distribution.minimumRoutingRatio);
          emitRatio(fields, "maximum_routing_ratio",
                    distribution.maximumRoutingRatio);
          emitRatio(fields, "minimum_direct_ratio",
                    distribution.minimumDirectRatio);
          emitRatio(fields, "maximum_direct_ratio",
                    distribution.maximumDirectRatio);
        });
  }

  for (const fabric::FabricTopologyScheduleQuality &schedule :
       report->schedules) {
    mapping_debug::emit(
        mapping_debug::Level::Summary, stage,
        mapping_debug::Event::TopologyQuality, [&](llvm::json::Object &fields) {
          fields["scope"] = "schedule_distribution";
          fields["artifact"] = artifact;
          fields["schedule"] = ::fabric::stringifySchedule(schedule.schedule);
          fields["pe_count"] = schedule.peCount;
          fields["memory_count"] = schedule.memoryCount;
          fields["switch_count"] = schedule.switchCount;
          emitHopDistribution(fields, "nearest_same_schedule_pe",
                              schedule.nearestSameSchedulePe);
          emitHopDistribution(fields, "nearest_other_schedule_pe",
                              schedule.nearestOtherSchedulePe);
          emitHopDistribution(fields, "nearest_matching_memory",
                              schedule.nearestMatchingMemory);
          emitHopDistribution(fields, "nearest_other_schedule_memory",
                              schedule.nearestOtherScheduleMemory);
        });
  }

  for (const fabric::FabricTopologyCapabilityQuality &capability :
       report->capabilities) {
    mapping_debug::emit(
        mapping_debug::Level::Summary, stage,
        mapping_debug::Event::TopologyQuality, [&](llvm::json::Object &fields) {
          fields["scope"] = "capability_distribution";
          fields["artifact"] = artifact;
          fields["operation_schema"] =
              ::dataflow::operationSchemaSpelling(capability.schema);
          fields["supporting_pes"] = peArray(capability.supportingPes);
          fields["spatial_pe_count"] = capability.spatialPeCount;
          fields["temporal_pe_count"] = capability.temporalPeCount;
          emitHopDistribution(fields, "coverage", capability.coverage);
          emitHopDistribution(fields, "supporting_peer",
                              capability.supportingPeer);
        });
  }

  for (const fabric::FabricTopologyOwnerQuality &owner : report->owners) {
    mapping_debug::emit(
        mapping_debug::Level::Decision, stage,
        mapping_debug::Event::TopologyQuality, [&](llvm::json::Object &fields) {
          fields["scope"] = "owner";
          fields["artifact"] = artifact;
          fields["terminal_kind"] =
              fabric::fabricTopologyTerminalKindSpelling(owner.kind);
          fields["owner"] = fabric::printFabricRef(owner.owner);
          fields["routing_ratio_numerator"] = owner.routingResourceCount();
          fields["direct_ratio_numerator"] = owner.directResourceCount();
          fields["ratio_denominator"] = owner.portCount();
          fields["boundary_port_count"] = owner.boundaryPortCount;
          fields["unreachable_port_count"] = owner.unreachablePortCount;
        });
    for (const fabric::FabricTopologyPortQuality &port : owner.ports) {
      mapping_debug::emit(
          mapping_debug::Level::Detail, stage,
          mapping_debug::Event::TopologyQuality,
          [&](llvm::json::Object &fields) {
            fields["scope"] = "port";
            fields["artifact"] = artifact;
            fields["owner"] = fabric::printFabricRef(owner.owner);
            fields["endpoint"] = fabric::printFabricRef(port.endpoint);
            fields["routing_resources"] = ownerArray(port.routingResources);
            fields["direct_resources"] = ownerArray(port.directResources);
            fields["reaches_module_boundary"] = port.reachesModuleBoundary;
            fields["unreachable"] = port.unreachable();
          });
    }
  }
  return report;
}

} // namespace loom::pnr
