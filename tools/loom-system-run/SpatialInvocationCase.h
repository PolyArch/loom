#ifndef LOOM_TOOLS_LOOM_SYSTEM_RUN_SPATIALINVOCATIONCASE_H
#define LOOM_TOOLS_LOOM_SYSTEM_RUN_SPATIALINVOCATIONCASE_H

#include "Common/Artifact.h"
#include "Deployment/Deployment.h"
#include "Runtime/SpatialInvocationWire.h"
#include "Simulator/SimulationExecution.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace loom::system_run {

/// Exact observations captured from one System engine at a Spatial boundary.
struct ObservedSpatialInvocation final {
  std::uint64_t dispatchTargetOrdinal = 0;
  std::string accCoreReference;
  std::string executionContextKey;
  loom::ArtifactRootReference workload;
  std::vector<std::uint8_t> invocation;
  loom::runtime::SpatialInvocationRuntimeInputSnapshot runtimeInput;
  std::vector<std::uint8_t> boundaryResult;
};

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

/// Imports both engine observations, proves their effective invocation and
/// boundary results agree, and derives the standalone Spatial execution case.
llvm::Expected<SpatialInvocationCase> materializeSpatialInvocationCase(
    std::size_t ordinal, const ObservedSpatialInvocation &dfg,
    const ObservedSpatialInvocation &cgra,
    const loom::deployment::FinalizedDeployment &deployment,
    const loom::ArtifactStore &artifacts, const loom::BlobStore &blobs);

} // namespace loom::system_run

#endif // LOOM_TOOLS_LOOM_SYSTEM_RUN_SPATIALINVOCATIONCASE_H
