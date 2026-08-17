#ifndef LOOM_PNR_SPATIALROOTSUPPLYADMISSION_H
#define LOOM_PNR_SPATIALROOTSUPPLYADMISSION_H

#include "Fabric/Artifact/FabricArtifact.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom::pnr {

enum class SpatialRootSupplyDisposition : std::uint8_t {
  Admissible,
  ProvenInfeasible,
};

/// Exact necessary root-supply closure for a TechMapping. Admissible means
/// only that every compute realization can receive a distinct resident
/// context and every memory realization has a joint occurrence assignment
/// satisfying operation-port, ingress, internal-connection, and resident-row
/// supply. Routing, attachment, and progress closure remain owned by full
/// Spatial PnR and its independent verifier.
struct SpatialRootSupplyAdmission final {
  SpatialRootSupplyDisposition disposition =
      SpatialRootSupplyDisposition::Admissible;
  std::uint64_t computeDemandCount = 0;
  std::uint64_t computeContextValueCount = 0;
  std::uint64_t computeContextEdgeCount = 0;
  std::uint64_t computeContextMaximumMatching = 0;
  std::uint64_t computeHallDemandCount = 0;
  std::uint64_t computeHallContextValueCount = 0;
  std::vector<std::uint64_t> computeHallRealizations;
  std::uint64_t memoryDemandCount = 0;
  std::uint64_t memoryOccurrenceValueCount = 0;
  std::uint64_t memoryOccurrenceChoiceCount = 0;
  std::uint64_t memoryExclusiveRelationCount = 0;
  std::uint64_t memoryAssignmentAttempts = 0;
  ::loom::mapping::SpatialMemoryOccurrenceSupplyFailureKind memoryFailure =
      ::loom::mapping::SpatialMemoryOccurrenceSupplyFailureKind::None;
  std::uint64_t deterministicWork = 0;
  std::string diagnostic;
};

llvm::Expected<SpatialRootSupplyAdmission>
analyzeSpatialRootSupply(const ::loom::mapping::TechMappingView &techMapping,
                         const ::dataflow::CanonicalDataflowProgramView &dataflow,
                         const ::loom::fabric::FabricArtifactView &fabric);

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALROOTSUPPLYADMISSION_H
