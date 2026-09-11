#ifndef LOOM_TOOLS_LOOM_SYSTEM_RUN_SPATIALCOMPUTEAGGREGATE_H
#define LOOM_TOOLS_LOOM_SYSTEM_RUN_SPATIALCOMPUTEAGGREGATE_H

#include "Application/SystemQor.h"
#include "SpatialInvocationCase.h"

namespace loom::system_run {

/// Aggregates the candidate System run's compute speed-of-light facts from
/// evidence that already exists: the retired Compute-kind actor firings of
/// every standalone CGRA replay by operation class, the realizations of each
/// class the selected SpatialMappings bound, the Fabric's per-class issue
/// lanes and placement slots under the TechMapping's own admission rule, the
/// distinct AccCores that received an invocation, and the SpatialCore clock
/// period in gem5 ticks. It derives no ratio; the Application QoR owner keeps
/// every formula. `cgraReplays` runs parallel to `invocations` by invocation
/// ordinal.
llvm::Expected<application::ApplicationSystemComputeInputs>
aggregateSpatialComputeInputs(
    llvm::ArrayRef<SpatialInvocationCase> invocations,
    llvm::ArrayRef<const sim::SpatialSimulationExecution *> cgraReplays,
    const std::optional<sim::SystemComputationInterval> &computation,
    const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::system_run

#endif // LOOM_TOOLS_LOOM_SYSTEM_RUN_SPATIALCOMPUTEAGGREGATE_H
