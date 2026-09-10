#ifndef LOOM_SIMULATOR_SYSTEMACTIVITY_H
#define LOOM_SIMULATOR_SYSTEMACTIVITY_H

#include "Simulator/SimulationExecution.h"

namespace loom::sim {
/// Native occupied shared-memory acceptance-service ticks divided by the
/// complete program's tick window, under the exact bound gem5 memory model.
/// Absence means the execution did not publish native service observations.
llvm::Expected<std::optional<evaluation::ExactRatio>>
projectSystemMemoryUtilization(const CanonicalSimulationExecution &execution,
                              const evaluation::CaseArtifactResolution &resolution,
                              const ArtifactStore &artifacts, const BlobStore &blobs);

/// The accelerated window of one System execution: the closed gem5-tick
/// interval from the first observed root Start lifecycle event through the last
/// observed root Completion lifecycle event. Host gaps between launches remain
/// inside the interval. `occupiedTicks` is the shared-memory acceptance service
/// consumed inside it, taken as the difference of the two bounding samples.
struct SystemAcceleratedWindow final {
  std::uint64_t firstStartTick = 0;
  std::uint64_t lastCompletionTick = 0;
  std::uint64_t occupiedTicks = 0;

  std::uint64_t elapsedTicks() const { return lastCompletionTick - firstStartTick; }
};

/// Absence means the execution completed no root launch, so it has no
/// accelerated window: a host-only run and a run whose launches never retired
/// are both unmeasured rather than fully utilized.
/// When supplied, the source computation interval selects lifecycle events
/// inside that interval; warmup cannot extend its measured activity span.
llvm::Expected<std::optional<SystemAcceleratedWindow>>
projectSystemAcceleratedWindow(
    const CanonicalSimulationExecution &execution,
    const evaluation::CaseArtifactResolution &resolution,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const SystemComputationInterval *computation = nullptr);
}
#endif
