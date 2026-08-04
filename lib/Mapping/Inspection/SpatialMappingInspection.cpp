#include "Mapping/Inspection/SpatialMappingInspection.h"

#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::mapping {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "spatial_mapping_inspection_invalid: " +
                                     message);
}

llvm::Error add(std::uint64_t amount, std::uint64_t &target,
                llvm::StringRef subject) {
  const auto updated = llvm::checkedAddUnsigned(target, amount);
  if (!updated)
    return invalid(subject + " count overflows u64");
  target = *updated;
  return llvm::Error::success();
}

template <typename Range>
llvm::Error addSize(const Range &range, std::uint64_t &target,
                    llvm::StringRef subject) {
  return add(static_cast<std::uint64_t>(range.size()), target, subject);
}

struct ComputeRow final {
  std::vector<std::uint8_t> occurrenceKey;
  std::vector<std::uint8_t> contextKey;
  SpatialComputeOccupancyInspection value;
};

bool computeRowLess(const ComputeRow &lhs, const ComputeRow &rhs) {
  if (lhs.occurrenceKey != rhs.occurrenceKey)
    return lhs.occurrenceKey < rhs.occurrenceKey;
  return lhs.contextKey < rhs.contextKey;
}

struct MemoryRow final {
  std::vector<std::uint8_t> occurrenceKey;
  SpatialMemoryOccupancyInspection value;
};

const TechComputeRealizationView *findComputeRealization(
    const llvm::DenseMap<std::uint64_t, const TechComputeRealizationView *>
        &realizations,
    std::uint64_t entity) {
  auto found = realizations.find(entity);
  return found == realizations.end() ? nullptr : found->second;
}

const TechMemoryRealizationView *findMemoryRealization(
    const llvm::DenseMap<std::uint64_t, const TechMemoryRealizationView *>
        &realizations,
    std::uint64_t entity) {
  auto found = realizations.find(entity);
  return found == realizations.end() ? nullptr : found->second;
}

llvm::Error inspectMemoryOperation(const SpatialMemoryOperationView &operation,
                                   SpatialMappingInspectionSummary &summary) {
  if (llvm::Error error =
          add(1, summary.memoryOperationCount, "memory operation"))
    return error;
  return std::visit(
      [&](const auto &typed) -> llvm::Error {
        if (llvm::Error error =
                addSize(typed.uses, summary.memoryUseCount, "memory use"))
          return error;
        using Operation = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<Operation,
                                     SpatialAddressedMemoryOperationView>)
          return addSize(typed.uses, summary.memoryDispatchCount,
                         "memory operation dispatch");
        return llvm::Error::success();
      },
      operation);
}

llvm::Error inspectResourceUse(const SpatialResourceUseView &use,
                               SpatialMappingInspectionSummary &summary) {
  if (llvm::Error error = add(1, summary.resourceUseCount, "resource use"))
    return error;
  if (llvm::Error error = addSize(use.parameters, summary.physicalTagValueCount,
                                  "resource-use parameter"))
    return error;
  return addSize(use.sharingAssignments, summary.physicalTagValueCount,
                 "resource-use sharing assignment");
}

} // namespace

llvm::Expected<SpatialMappingInspection>
inspectSpatialMapping(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                      const TechMappingView &techMapping,
                      const ::loom::fabric::FabricArtifactView &fabric,
                      const SpatialMappingView &spatialMapping) {
  if (techMapping.dataflowIdentity() != dataflow.identity() ||
      techMapping.fabricIdentity() != fabric.identity() ||
      spatialMapping.dataflowIdentity() != dataflow.identity() ||
      spatialMapping.techMappingIdentity() != techMapping.identity() ||
      spatialMapping.fabricIdentity() != fabric.identity())
    return invalid("D/T/F/SpatialMapping owner tuple is inconsistent");

  SpatialMappingInspection result;
  if (llvm::Error error =
          addSize(techMapping.covers(), result.summary.coveredGraphCount,
                  "covered graph"))
    return std::move(error);

  llvm::DenseMap<std::uint64_t, const TechComputeRealizationView *>
      computeRealizations;
  computeRealizations.reserve(techMapping.computeRealizations().size());
  for (const TechComputeRealizationView &realization :
       techMapping.computeRealizations())
    if (!computeRealizations.try_emplace(realization.entityId, &realization)
             .second)
      return invalid("TechMapping has duplicate compute realization IDs");

  llvm::DenseMap<std::uint64_t, const TechMemoryRealizationView *>
      memoryRealizations;
  memoryRealizations.reserve(techMapping.memoryRealizations().size());
  for (const TechMemoryRealizationView &realization :
       techMapping.memoryRealizations())
    if (!memoryRealizations.try_emplace(realization.entityId, &realization)
             .second)
      return invalid("TechMapping has duplicate memory realization IDs");

  std::vector<ComputeRow> computeRows;
  computeRows.reserve(spatialMapping.computeBindings().size());
  for (const SpatialComputeBindingView &binding :
       spatialMapping.computeBindings()) {
    const TechComputeRealizationView *realization =
        findComputeRealization(computeRealizations, binding.realization);
    if (!realization)
      return invalid("compute binding references an absent realization");
    if (llvm::Error error = add(1, result.summary.computeRealizationCount,
                                "compute realization"))
      return std::move(error);
    if (llvm::Error error =
            addSize(realization->actors, result.summary.selectedActorCount,
                    "selected actor"))
      return std::move(error);
    computeRows.push_back(
        {::loom::fabric::canonicalFabricBytes(binding.occurrence),
         ::loom::fabric::canonicalFabricBytes(binding.context),
         {binding.occurrence, binding.context, 1,
          static_cast<std::uint64_t>(realization->actors.size())}});
  }
  std::sort(computeRows.begin(), computeRows.end(), computeRowLess);
  result.computeOccupancy.reserve(computeRows.size());
  for (ComputeRow &row : computeRows) {
    if (result.computeOccupancy.empty() ||
        !(result.computeOccupancy.back().occurrence == row.value.occurrence &&
          result.computeOccupancy.back().context == row.value.context)) {
      result.computeOccupancy.push_back(std::move(row.value));
      continue;
    }
    SpatialComputeOccupancyInspection &group = result.computeOccupancy.back();
    if (llvm::Error error =
            add(row.value.realizationCount, group.realizationCount,
                "compute occurrence realization"))
      return std::move(error);
    if (llvm::Error error = add(row.value.actorCount, group.actorCount,
                                "compute occurrence actor"))
      return std::move(error);
  }
  result.summary.computeOccurrenceContextCount = result.computeOccupancy.size();

  std::vector<MemoryRow> memoryRows;
  memoryRows.reserve(spatialMapping.memoryEngineBindings().size());
  for (const SpatialMemoryEngineBindingView &binding :
       spatialMapping.memoryEngineBindings()) {
    const TechMemoryRealizationView *realization =
        findMemoryRealization(memoryRealizations, binding.realization);
    if (!realization)
      return invalid("memory binding references an absent realization");
    if (llvm::Error error =
            add(1, result.summary.memoryRealizationCount, "memory realization"))
      return std::move(error);
    if (llvm::Error error =
            addSize(realization->actors, result.summary.selectedActorCount,
                    "selected memory actor"))
      return std::move(error);
    for (const SpatialMemoryOperationView &operation : binding.operations)
      if (llvm::Error error = inspectMemoryOperation(operation, result.summary))
        return std::move(error);
    memoryRows.push_back(
        {::loom::fabric::canonicalFabricBytes(binding.occurrence),
         {binding.occurrence, 1,
          static_cast<std::uint64_t>(binding.operations.size())}});
  }
  std::sort(memoryRows.begin(), memoryRows.end(),
            [](const MemoryRow &lhs, const MemoryRow &rhs) {
              return lhs.occurrenceKey < rhs.occurrenceKey;
            });
  result.memoryOccupancy.reserve(memoryRows.size());
  for (MemoryRow &row : memoryRows) {
    if (result.memoryOccupancy.empty() ||
        !(result.memoryOccupancy.back().occurrence == row.value.occurrence)) {
      result.memoryOccupancy.push_back(std::move(row.value));
      continue;
    }
    SpatialMemoryOccupancyInspection &group = result.memoryOccupancy.back();
    if (llvm::Error error =
            add(row.value.realizationCount, group.realizationCount,
                "memory occurrence realization"))
      return std::move(error);
    if (llvm::Error error = add(row.value.operationCount, group.operationCount,
                                "memory occurrence operation"))
      return std::move(error);
  }
  result.summary.memoryOccurrenceCount = result.memoryOccupancy.size();

  result.routes.reserve(spatialMapping.routeTrees().size());
  for (const SpatialRouteTreeView &route : spatialMapping.routeTrees()) {
    SpatialRouteInspection inspection{route.logicalNet};
    if (llvm::Error error =
            addSize(route.nodes, inspection.nodeCount, "route node"))
      return std::move(error);
    if (llvm::Error error =
            addSize(route.sinks, inspection.sinkCount, "route sink"))
      return std::move(error);
    for (const SpatialRouteNodeView &node : route.nodes)
      if (node.incomingTraversal)
        if (llvm::Error error =
                add(1, inspection.traversalCount, "route traversal"))
          return std::move(error);
    if (llvm::Error error = add(1, result.summary.routeTreeCount, "route tree"))
      return std::move(error);
    if (llvm::Error error = add(inspection.nodeCount,
                                result.summary.routeNodeCount, "route node"))
      return std::move(error);
    if (llvm::Error error =
            add(inspection.traversalCount, result.summary.routeTraversalCount,
                "route traversal"))
      return std::move(error);
    if (llvm::Error error = add(inspection.sinkCount,
                                result.summary.routeSinkCount, "route sink"))
      return std::move(error);
    result.routes.push_back(std::move(inspection));
  }

  for (const SpatialMemoryBindingView &binding :
       spatialMapping.memoryBindings()) {
    if (std::holds_alternative<SpatialMemoryLocalRegionView>(binding.target)) {
      if (llvm::Error error = add(1, result.summary.localMemoryBindingCount,
                                  "local memory binding"))
        return std::move(error);
    } else if (llvm::Error error =
                   add(1, result.summary.boundaryMemoryBindingCount,
                       "boundary memory binding")) {
      return std::move(error);
    }
    if (llvm::Error error = addSize(
            binding.exposures, result.summary.exposureCount, "memory exposure"))
      return std::move(error);
    if (llvm::Error error =
            addSize(binding.exposures, result.summary.memoryDispatchCount,
                    "exposure dispatch"))
      return std::move(error);
  }
  for (const SpatialResourceUseView &use : spatialMapping.resourceUses())
    if (llvm::Error error = inspectResourceUse(use, result.summary))
      return std::move(error);

  return result;
}

} // namespace loom::mapping
