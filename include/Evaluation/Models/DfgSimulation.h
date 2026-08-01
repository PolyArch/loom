#ifndef LOOM_EVALUATION_MODELS_DFGSIMULATION_H
#define LOOM_EVALUATION_MODELS_DFGSIMULATION_H

#include "Evaluation/Evidence.h"

#include <chrono>
#include <cstdint>
#include <optional>

namespace loom {
class ArtifactStore;
struct ResolvedConfig;
} // namespace loom

namespace loom::evaluation::models {

struct PreparedDfgSimulationEvaluation final {
  EvaluationRequest request;
  CaseArtifactResolution resolution;
};

/// Nonsemantic limits for one DFG provider attempt. They do not enter Request
/// identity and cannot select a formal result from an explored prefix.
struct DfgSimulationAttemptLimits final {
  std::uint64_t maxWavefrontSteps = 100000;
  std::optional<std::chrono::steady_clock::time_point> executionDeadline;
};

llvm::Error registerDfgSimulationModel();

llvm::Expected<PreparedDfgSimulationEvaluation>
prepareDfgSimulationEvaluation(const ArtifactRootReference &canonicalDataflow,
                               const ArtifactRootReference &workload,
                               const ArtifactRootReference &runtimeInput,
                               const ResolvedConfig &config,
                               const ArtifactStore &artifactStore);

/// Executes one exact prepared Request and returns its canonical Evidence
/// value. The caller owns Evidence publication and attempt bookkeeping.
llvm::Expected<EvaluationEvidence>
evaluateDfgSimulation(const PreparedDfgSimulationEvaluation &prepared,
                      DfgSimulationAttemptLimits limits,
                      const ArtifactStore &artifactStore);

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_DFGSIMULATION_H
