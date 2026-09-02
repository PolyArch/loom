#ifndef LOOM_TOOLS_LOOM_SYSTEM_RUN_SPATIALINVOCATIONCASE_H
#define LOOM_TOOLS_LOOM_SYSTEM_RUN_SPATIALINVOCATIONCASE_H

#include "Common/Artifact.h"
#include "Simulator/SimulationExecution.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace loom::system_run {

/// One Spatial invocation materialized from the System engines' observations.
/// Every Spatial cell (DFG, CGRA, mapped RTL) executes the same case
/// independently and is compared against the System boundary results.
struct SpatialInvocationCase final {
  std::size_t ordinal = 0;
  std::uint64_t dispatchTargetOrdinal = 0;
  std::string accCoreReference;
  std::string executionContextKey;
  std::vector<std::uint64_t> denseCoordinates;
  ArtifactRootReference dataflow;
  ArtifactRootReference workload;
  ArtifactRootReference runtimeInput;
  ArtifactRootReference hardwareImplementation;
  ArtifactRootReference fabric;
  ArtifactRootReference spatialMapping;
  sim::SpatialEngineBoundaryResult systemDfgBoundary;
  sim::SpatialEngineBoundaryResult systemCgraBoundary;
};

} // namespace loom::system_run

#endif // LOOM_TOOLS_LOOM_SYSTEM_RUN_SPATIALINVOCATIONCASE_H
