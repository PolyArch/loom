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

}
#endif
