#ifndef LOOM_RUNTIME_GEM5SYSTEMEXECUTION_H
#define LOOM_RUNTIME_GEM5SYSTEMEXECUTION_H

#include "Evaluation/ModelProvider.h"
#include "Simulator/SimulationExecution.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::runtime {

struct Gem5CgraEngineAttemptProfile final {
  std::uint64_t invocationCount = 0;
  std::uint64_t activeWallNanoseconds = 0;
  std::uint64_t activeProcessCpuNanoseconds = 0;
  std::uint64_t eventFrameCount = 0;
};

struct Gem5SystemAttemptProfile final {
  std::uint64_t configurationWallNanoseconds = 0;
  std::uint64_t activeWallNanoseconds = 0;
  std::uint64_t gem5ActiveProcessCpuNanoseconds = 0;
  std::uint64_t observationWallNanoseconds = 0;
  std::uint64_t observationProcessCpuNanoseconds = 0;
  std::uint64_t engineProcessCpuNanoseconds = 0;
  std::uint64_t bridgeCallbackCpuNanoseconds = 0;
  std::uint64_t bridgeEngineWaitNanoseconds = 0;
  std::uint64_t bridgeMessageCount = 0;
  std::uint64_t acceleratorInvocationCount = 0;
  std::uint64_t bridgeCount = 0;
  std::optional<Gem5CgraEngineAttemptProfile> cgraEngine;
};

struct Gem5SpatialInvocationProjection final {
  std::uint64_t bridgeSessionOrdinal = 0;
  std::uint64_t sequence = 0;
  std::uint64_t sessionEntryOrdinal = 0;
  std::uint64_t launchOrdinal = 0;
  std::uint64_t completionGem5Tick = 0;
  sim::SpatialProgressObservations progress;
  std::uint64_t acceleratorReferenceCycles = 0;
};

/// The transient provider result plus attempt-local operational diagnostics.
/// Evaluation remains the sole owner that finalizes persistent Evidence.
struct Gem5SystemEvaluation final {
  evaluation::EvaluationModelResult result;
  std::vector<Gem5SpatialInvocationProjection> spatialInvocations;
  std::optional<Gem5SystemAttemptProfile> attemptProfile;
};

/// Prepares the exact gem5 System invocation selected by model kind 17, 18,
/// or 19. The model descriptor remains the sole engine selector.
llvm::Expected<evaluation::EvaluationModelProviderPreparation>
prepareGem5SystemInvocation(
    const evaluation::EvaluationRequest &request,
    const evaluation::CaseArtifactResolution &resolution,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const external_tool::ExternalToolPreparationContext &context);

/// Strictly imports one completed gem5 invocation and publishes its
/// Deployment-owned SystemSimulationExecution.
llvm::Expected<evaluation::EvaluationModelResult> importGem5SystemInvocation(
    const evaluation::EvaluationRequest &request,
    const evaluation::CaseArtifactResolution &resolution,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const ArtifactStore &artifacts, const BlobStore &blobs);

/// Strictly imports the same invocation while retaining typed, attempt-local
/// timing and exact Spatial progress projections for operational consumers.
llvm::Expected<Gem5SystemEvaluation> importGem5SystemEvaluationWithDiagnostics(
    const evaluation::EvaluationRequest &request,
    const evaluation::CaseArtifactResolution &resolution,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::runtime

#endif // LOOM_RUNTIME_GEM5SYSTEMEXECUTION_H
