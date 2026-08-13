#include "FabricTopologyQualityDiagnostic.h"

#include "Common/ArtifactText.h"
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
