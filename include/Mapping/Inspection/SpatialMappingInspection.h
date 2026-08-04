#ifndef LOOM_MAPPING_INSPECTION_SPATIALMAPPINGINSPECTION_H
#define LOOM_MAPPING_INSPECTION_SPATIALMAPPINGINSPECTION_H

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Mapping/Artifact/MappingArtifact.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::mapping {

struct SpatialMappingInspectionSummary final {
  std::uint64_t coveredGraphCount = 0;
  std::uint64_t computeRealizationCount = 0;
  std::uint64_t memoryRealizationCount = 0;
  std::uint64_t selectedActorCount = 0;
  std::uint64_t computeOccurrenceContextCount = 0;
  std::uint64_t memoryOccurrenceCount = 0;
  std::uint64_t routeTreeCount = 0;
  std::uint64_t routeNodeCount = 0;
  std::uint64_t routeTraversalCount = 0;
  std::uint64_t routeSinkCount = 0;
  std::uint64_t resourceUseCount = 0;
  std::uint64_t physicalTagValueCount = 0;
  std::uint64_t memoryOperationCount = 0;
  std::uint64_t memoryUseCount = 0;
  std::uint64_t localMemoryBindingCount = 0;
  std::uint64_t boundaryMemoryBindingCount = 0;
  std::uint64_t memoryDispatchCount = 0;
  std::uint64_t exposureCount = 0;
};

struct SpatialComputeOccupancyInspection final {
  ::loom::fabric::FabricFuOccurrenceRef occurrence;
  ::loom::fabric::InstructionContextRef context;
  std::uint64_t realizationCount = 0;
  std::uint64_t actorCount = 0;
};

struct SpatialMemoryOccupancyInspection final {
  ::loom::fabric::FabricMemoryOccurrenceRef occurrence;
  std::uint64_t realizationCount = 0;
  std::uint64_t operationCount = 0;
};

struct SpatialRouteInspection final {
  ::dataflow::CanonicalGraphProducerEndpointRef logicalNet;
  std::uint64_t nodeCount = 0;
  std::uint64_t traversalCount = 0;
  std::uint64_t sinkCount = 0;
};

/// Removable typed projection of one sealed D/T/F/SpatialMapping tuple. It is
/// intended for direct conformance checks and visualization joins; none of its
/// counts or grouping records participate in Mapping identity or legality.
struct SpatialMappingInspection final {
  SpatialMappingInspectionSummary summary;
  std::vector<SpatialComputeOccupancyInspection> computeOccupancy;
  std::vector<SpatialMemoryOccupancyInspection> memoryOccupancy;
  std::vector<SpatialRouteInspection> routes;
};

llvm::Expected<SpatialMappingInspection>
inspectSpatialMapping(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                      const TechMappingView &techMapping,
                      const ::loom::fabric::FabricArtifactView &fabric,
                      const SpatialMappingView &spatialMapping);

} // namespace loom::mapping

#endif // LOOM_MAPPING_INSPECTION_SPATIALMAPPINGINSPECTION_H
