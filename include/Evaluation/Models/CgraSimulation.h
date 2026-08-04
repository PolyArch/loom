#ifndef LOOM_EVALUATION_MODELS_CGRASIMULATION_H
#define LOOM_EVALUATION_MODELS_CGRASIMULATION_H

#include "Evaluation/Evidence.h"
#include "Simulator/CGRAAdmission.h"
#include "Simulator/SimulationArtifacts.h"

#include <chrono>
#include <cstdint>
#include <optional>

namespace loom {
class ArtifactStore;
struct ResolvedConfig;
} // namespace loom

namespace loom::evaluation::models {

struct PreparedCgraSimulationEvaluation final {
  EvaluationRequest request;
  CaseArtifactResolution resolution;
  sim::PreparedCgraExecution execution;
  sim::CanonicalSimulationWorkload workload;
  sim::CanonicalSimulationRuntimeInput runtimeInput;
};

struct CgraSimulationAttemptLimits final {
  std::uint64_t maxEventFrames = 100000;
  std::optional<std::chrono::steady_clock::time_point> executionDeadline;
};

llvm::Error registerCgraSimulationModel();

llvm::Expected<PreparedCgraSimulationEvaluation>
prepareCgraSimulationEvaluation(const ArtifactRootReference &canonicalDataflow,
                                const ArtifactRootReference &fabric,
                                const ArtifactRootReference &spatialMapping,
                                const ArtifactRootReference &workload,
                                const ArtifactRootReference &runtimeInput,
                                const ResolvedConfig &config,
                                const ArtifactStore &artifactStore);

llvm::Expected<EvaluationEvidence>
evaluateCgraSimulation(const PreparedCgraSimulationEvaluation &prepared,
                       CgraSimulationAttemptLimits limits,
                       const ArtifactStore &artifactStore);

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_CGRASIMULATION_H
