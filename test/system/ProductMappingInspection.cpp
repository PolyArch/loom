#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowStructuralRefs.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"
#include "Mapping/Inspection/SpatialMappingInspection.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <variant>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::Error error) {
  llvm::errs() << "product mapping inspection: "
               << llvm::toString(std::move(error)) << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(value.takeError());
  return std::move(*value);
}

loom::ArtifactRootReference
reference(const loom::ArtifactSchemaDescriptor &schema,
          const loom::ArtifactIdentity &identity) {
  return {schema.identity.str(), schema.version, identity};
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    llvm::errs() << "usage: " << argv[0]
                 << " ARTIFACT_STORE SPATIAL_MAPPING_IDENTITY\n";
    return 2;
  }

  const loom::ArtifactStore artifacts(argv[1]);
  const loom::ArtifactIdentity spatialIdentity =
      take(loom::parseArtifactIdentityHex(argv[2]));
  const auto spatial = take(loom::mapping::importSpatialMapping(
      reference(loom::mapping::mappingArtifactSchema, spatialIdentity),
      artifacts));
  const auto dataflow = take(dataflow::importCanonicalDataflow(
      reference(dataflow::canonicalDataflowSchema,
                spatial.view().dataflowIdentity()),
      artifacts));
  const auto dataflowView = take(dataflow.view());
  const auto tech = take(loom::mapping::importTechMapping(
      reference(loom::mapping::mappingArtifactSchema,
                spatial.view().techMappingIdentity()),
      artifacts));
  const auto fabric = take(loom::fabric::importEntireFabricRoot(
      reference(loom::fabric::fabricArtifactSchema,
                spatial.view().fabricIdentity()),
      artifacts));
  const auto inspection = take(loom::mapping::inspectSpatialMapping(
      dataflowView, tech.view(), fabric.view(), spatial.view()));
  const auto operandQueueGroups =
      take(loom::mapping::deriveSpatialPeOperandQueueMatchGroups(
          tech.view(), fabric.view(), spatial.view().computeBindings(),
          spatial.view().routeTrees(), spatial.view().resourceUses(),
          spatial.view().physicalTagSegments()));
  const auto temporalDispatchDomains =
      take(loom::mapping::deriveSpatialTemporalPeDispatchDomains(
          fabric.view(), spatial.view().computeBindings()));
  const auto packedSwitchRows =
      take(loom::mapping::deriveSpatialTemporalSwitchPackedRows(
          fabric.view(), spatial.view().routeTrees(),
          spatial.view().resourceUses(), spatial.view().physicalTagSegments()));

  std::uint64_t fabricMemoryTemplateInternalConnectionCount = 0;
  std::uint64_t fabricMemoryTemplateWithInternalConnectionCount = 0;
  for (const auto definition : fabric.view().memoryEngineTemplates()) {
    const auto *record = fabric.view().memoryEngineTemplate(definition);
    if (!record) {
      llvm::errs() << "product mapping inspection: memory template does not "
                      "resolve\n";
      return 1;
    }
    fabricMemoryTemplateInternalConnectionCount +=
        record->internalConnections.size();
    fabricMemoryTemplateWithInternalConnectionCount +=
        !record->internalConnections.empty();
  }

  std::uint64_t actorMulticastRouteCount = 0;
  std::uint64_t maximumActorMulticastSinks = 0;
  for (const loom::mapping::SpatialRouteInspection &route : inspection.routes) {
    if (!std::holds_alternative<dataflow::ActorTokenResultRef>(
            route.logicalNet))
      continue;
    const auto residual = llvm::find_if(
        tech.view().residualLogicalNets(), [&](const auto &logicalNet) {
          return logicalNet.producer == route.logicalNet;
        });
    if (residual == tech.view().residualLogicalNets().end()) {
      llvm::errs() << "product mapping inspection: route has no TechMapping "
                      "residual net\n";
      return 1;
    }
    if (residual->sinks.size() < 2)
      continue;
    if (route.sinkCount != residual->sinks.size()) {
      llvm::errs() << "product mapping inspection: actor multicast route "
                      "does not cover its complete residual sink set\n";
      return 1;
    }
    ++actorMulticastRouteCount;
    maximumActorMulticastSinks =
        std::max(maximumActorMulticastSinks, route.sinkCount);
  }

  std::uint64_t memoryInternalEdgeCount = 0;
  std::uint64_t techComputeActorCount = 0;
  for (const loom::mapping::TechComputeRealizationView &realization :
       tech.view().computeRealizations())
    techComputeActorCount += realization.actors.size();
  std::uint64_t techMemoryActorCount = 0;
  for (const loom::mapping::TechMemoryRealizationView &realization :
       tech.view().memoryRealizations()) {
    techMemoryActorCount += realization.actors.size();
    memoryInternalEdgeCount += realization.internalEdges.size();
  }

  std::uint64_t spatialMemoryEngineBindingCount = 0;
  std::uint64_t temporalMemoryEngineBindingCount = 0;
  std::uint64_t spatialMemoryOperationCount = 0;
  std::uint64_t temporalMemoryOperationCount = 0;
  std::map<std::vector<std::uint8_t>, std::vector<std::uint64_t>>
      temporalContextOrdinals;
  std::map<std::uint64_t, loom::fabric::FabricMemoryOccurrenceRef>
      memoryOccurrenceByRealization;
  std::map<std::vector<std::uint8_t>, std::uint64_t>
      memoryOperationsByOccurrence;
  for (const loom::mapping::SpatialMemoryEngineBindingView &binding :
       spatial.view().memoryEngineBindings()) {
    if (!memoryOccurrenceByRealization
             .emplace(binding.realization, binding.occurrence)
             .second) {
      llvm::errs() << "product mapping inspection: duplicate memory "
                      "realization binding\n";
      return 1;
    }
    const auto schedule = fabric.view().memorySchedule(binding.occurrence);
    if (!schedule) {
      llvm::errs() << "product mapping inspection: memory occurrence has no "
                      "schedule\n";
      return 1;
    }
    const auto occurrenceKey =
        loom::fabric::canonicalFabricBytes(binding.occurrence);
    memoryOperationsByOccurrence[occurrenceKey] += binding.operations.size();
    if (*schedule == ::fabric::Schedule::Spatial) {
      ++spatialMemoryEngineBindingCount;
      spatialMemoryOperationCount += binding.operations.size();
    } else {
      ++temporalMemoryEngineBindingCount;
      temporalMemoryOperationCount += binding.operations.size();
    }
    for (const loom::mapping::SpatialMemoryOperationView &operation :
         binding.operations) {
      const auto &placement = std::visit(
          [](const auto &selected)
              -> const loom::mapping::SpatialMemoryOperationPlacementView & {
            return selected.placement;
          },
          operation);
      if (*schedule == ::fabric::Schedule::Spatial) {
        if (!std::holds_alternative<loom::fabric::FabricMemoryOperationPortRef>(
                placement)) {
          llvm::errs() << "product mapping inspection: Spatial memory "
                          "operation has a Temporal placement\n";
          return 1;
        }
        continue;
      }
      const auto *context =
          std::get_if<loom::fabric::FabricMemoryOperationContextRef>(
              &placement);
      if (!context) {
        llvm::errs() << "product mapping inspection: Temporal memory "
                        "operation has a Spatial placement\n";
        return 1;
      }
      temporalContextOrdinals[occurrenceKey].push_back(context->ordinal);
    }
  }

  std::uint64_t denseTemporalMemoryOccurrenceCount = 0;
  for (auto &[occurrence, ordinals] : temporalContextOrdinals) {
    (void)occurrence;
    llvm::sort(ordinals);
    bool dense = true;
    for (const auto [expected, actual] : llvm::enumerate(ordinals))
      dense &= actual == expected;
    denseTemporalMemoryOccurrenceCount += dense;
  }

  using TemporalIngressClaim =
      std::pair<std::vector<std::uint8_t>, std::vector<std::uint8_t>>;
  std::set<TemporalIngressClaim> temporalIngressClaims;
  std::uint64_t temporalIngressClaimCount = 0;
  for (const loom::mapping::TechMemoryRealizationView &realization :
       tech.view().memoryRealizations()) {
    const auto found = memoryOccurrenceByRealization.find(realization.entityId);
    if (found == memoryOccurrenceByRealization.end()) {
      llvm::errs() << "product mapping inspection: memory realization has no "
                      "Spatial binding\n";
      return 1;
    }
    const auto schedule = fabric.view().memorySchedule(found->second);
    if (!schedule || *schedule != ::fabric::Schedule::Temporal)
      continue;
    const auto demand = take(loom::mapping::deriveSpatialMemoryOccurrenceDemand(
        realization, dataflowView, fabric.view()));
    for (const auto &resource : demand.exclusiveResources) {
      if (resource.kind != loom::mapping::SpatialMemoryExclusiveResourceKind::
                               TemporalExternalIngress)
        continue;
      ++temporalIngressClaimCount;
      temporalIngressClaims.emplace(
          loom::fabric::canonicalFabricBytes(found->second), resource.key);
    }
  }

  std::uint64_t memoryBindingCount = spatial.view().memoryBindings().size();
  std::uint64_t boundaryMemoryBindingCount = 0;
  for (const loom::mapping::SpatialMemoryBindingView &binding :
       spatial.view().memoryBindings())
    boundaryMemoryBindingCount +=
        std::holds_alternative<loom::mapping::SpatialMemoryBoundaryProxyView>(
            binding.target);

  std::uint64_t configuredMemoryOccurrenceCount = 0;
  std::uint64_t configuredMemoryActiveOperationRowCount = 0;
  std::uint64_t configuredMemoryActiveProviderRowCount = 0;
  for (const auto &entry : memoryOperationsByOccurrence) {
    const auto &occurrenceKey = entry.first;
    const std::uint64_t expectedOperationCount = entry.second;
    const auto engine = llvm::find_if(
        spatial.view().memoryEngineBindings(), [&](const auto &binding) {
          return loom::fabric::canonicalFabricBytes(binding.occurrence) ==
                 occurrenceKey;
        });
    if (engine == spatial.view().memoryEngineBindings().end()) {
      llvm::errs() << "product mapping inspection: missing memory occurrence "
                      "representative\n";
      return 1;
    }
    const auto schema =
        take(fabric.view().memoryConfigurationSchema(engine->occurrence));
    const loom::mapping::ConfiguredHardwareFieldValueView *selected = nullptr;
    for (const auto &field : spatial.view().configuredHardware().fields()) {
      if (field.slot.field != schema.field())
        continue;
      if (selected) {
        llvm::errs() << "product mapping inspection: duplicate configured "
                        "memory field\n";
        return 1;
      }
      selected = &field;
    }
    if (!selected) {
      llvm::errs() << "product mapping inspection: missing configured memory "
                      "field\n";
      return 1;
    }
    const auto value = take(schema.decode(selected->value.bytes()));
    const auto *active = std::get_if<loom::fabric::FabricMemoryActive>(&value);
    if (!active) {
      llvm::errs() << "product mapping inspection: selected memory is "
                      "disabled\n";
      return 1;
    }
    const std::uint64_t activeOperationRows = llvm::count_if(
        active->operationRows, [](const auto &row) { return row.has_value(); });
    if (activeOperationRows != expectedOperationCount) {
      llvm::errs() << "product mapping inspection: configured memory row "
                      "count disagrees with operation bindings\n";
      return 1;
    }
    ++configuredMemoryOccurrenceCount;
    configuredMemoryActiveOperationRowCount += activeOperationRows;
    for (const auto &rows : active->providerDecodeRows)
      configuredMemoryActiveProviderRowCount +=
          llvm::count_if(rows, [](const auto &row) { return row.has_value(); });
  }

  std::map<loom::fabric::FabricEntityId, std::uint64_t> temporalBindingsByPe;
  std::uint64_t temporalComputeBindingCount = 0;
  for (const loom::mapping::SpatialComputeBindingView &binding :
       spatial.view().computeBindings()) {
    const auto pe = fabric.view().parentPeOf(binding.occurrence);
    if (pe && fabric.view().peSchedule(*pe) == ::fabric::Schedule::Temporal) {
      ++temporalComputeBindingCount;
      ++temporalBindingsByPe[pe->id()];
    }
  }
  std::uint64_t maximumTemporalBindingsPerPe = 0;
  for (const auto &[pe, count] : temporalBindingsByPe) {
    (void)pe;
    maximumTemporalBindingsPerPe =
        std::max(maximumTemporalBindingsPerPe, count);
  }

  std::uint64_t operandQueueMatchCount = 0;
  std::uint64_t operandQueueAtomicFanoutGroupCount = 0;
  std::uint64_t maximumOperandQueueMatches = 0;
  for (const loom::mapping::SpatialPeOperandQueueMatchGroupView &group :
       operandQueueGroups) {
    operandQueueMatchCount += group.matches.size();
    maximumOperandQueueMatches =
        std::max(maximumOperandQueueMatches,
                 static_cast<std::uint64_t>(group.matches.size()));
    operandQueueAtomicFanoutGroupCount += group.matches.size() > 1;
  }

  std::uint64_t temporalDispatchCandidateCount = 0;
  for (const loom::mapping::SpatialTemporalPeDispatchDomainView &domain :
       temporalDispatchDomains)
    temporalDispatchCandidateCount += domain.candidates.size();

  std::uint64_t packedSwitchSignatureCount = 0;
  std::uint64_t sharedPackedSwitchRowCount = 0;
  std::uint64_t maximumPackedSwitchRowSignatures = 0;
  for (const loom::mapping::SpatialTemporalSwitchPackedRowView &row :
       packedSwitchRows) {
    packedSwitchSignatureCount += row.signatures.size();
    maximumPackedSwitchRowSignatures =
        std::max(maximumPackedSwitchRowSignatures,
                 static_cast<std::uint64_t>(row.signatures.size()));
    sharedPackedSwitchRowCount += row.signatures.size() > 1;
  }

  llvm::json::Object report{
      {"schema", "loom.test.product_mapping_inspection.1"},
      {"route_tree_count", inspection.summary.routeTreeCount},
      {"route_sink_count", inspection.summary.routeSinkCount},
      {"tech_compute_realization_count",
       tech.view().computeRealizations().size()},
      {"tech_compute_actor_count", techComputeActorCount},
      {"tech_memory_realization_count",
       tech.view().memoryRealizations().size()},
      {"tech_memory_actor_count", techMemoryActorCount},
      {"spatial_memory_engine_binding_count",
       spatial.view().memoryEngineBindings().size()},
      {"spatial_schedule_memory_engine_binding_count",
       spatialMemoryEngineBindingCount},
      {"temporal_memory_engine_binding_count",
       temporalMemoryEngineBindingCount},
      {"spatial_memory_operation_count", spatialMemoryOperationCount},
      {"temporal_memory_operation_count", temporalMemoryOperationCount},
      {"temporal_memory_occurrence_count", temporalContextOrdinals.size()},
      {"dense_temporal_memory_occurrence_count",
       denseTemporalMemoryOccurrenceCount},
      {"temporal_memory_external_ingress_claim_count",
       temporalIngressClaimCount},
      {"unique_temporal_memory_external_ingress_claim_count",
       temporalIngressClaims.size()},
      {"memory_binding_count", memoryBindingCount},
      {"local_memory_binding_count",
       inspection.summary.localMemoryBindingCount},
      {"boundary_memory_binding_count", boundaryMemoryBindingCount},
      {"memory_internal_edge_count", memoryInternalEdgeCount},
      {"fabric_memory_template_internal_connection_count",
       fabricMemoryTemplateInternalConnectionCount},
      {"fabric_memory_template_with_internal_connection_count",
       fabricMemoryTemplateWithInternalConnectionCount},
      {"configured_memory_occurrence_count", configuredMemoryOccurrenceCount},
      {"configured_memory_active_operation_row_count",
       configuredMemoryActiveOperationRowCount},
      {"configured_memory_active_provider_row_count",
       configuredMemoryActiveProviderRowCount},
      {"temporal_compute_binding_count", temporalComputeBindingCount},
      {"temporal_pe_count", temporalBindingsByPe.size()},
      {"maximum_temporal_compute_bindings_per_pe",
       maximumTemporalBindingsPerPe},
      {"register_fifo_transfer_count",
       spatial.view().registerFifoTransfers().size()},
      {"operand_queue_match_group_count", operandQueueGroups.size()},
      {"operand_queue_match_count", operandQueueMatchCount},
      {"operand_queue_atomic_fanout_group_count",
       operandQueueAtomicFanoutGroupCount},
      {"maximum_operand_queue_matches", maximumOperandQueueMatches},
      {"temporal_dispatch_domain_count", temporalDispatchDomains.size()},
      {"temporal_dispatch_candidate_count", temporalDispatchCandidateCount},
      {"packed_switch_row_count", packedSwitchRows.size()},
      {"packed_switch_signature_count", packedSwitchSignatureCount},
      {"shared_packed_switch_row_count", sharedPackedSwitchRowCount},
      {"maximum_packed_switch_row_signatures",
       maximumPackedSwitchRowSignatures},
      {"actor_multicast_route_count", actorMulticastRouteCount},
      {"maximum_actor_multicast_sinks", maximumActorMulticastSinks},
  };
  llvm::outs() << llvm::formatv("{0:2}\n",
                                llvm::json::Value(std::move(report)));
  return 0;
}
