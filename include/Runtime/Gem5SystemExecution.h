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
  std::optional<std::uint64_t> engineProcessCpuNanoseconds;
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

/// Attempt-local operational observations from one explicitly fresh gem5
/// diagnostic invocation. The ordinary Evaluation provider never requires or
/// imports these observations.
struct Gem5SystemDiagnosticEvaluation final {
  evaluation::EvaluationModelResult result;
  std::vector<Gem5SpatialInvocationProjection> spatialInvocations;
  Gem5SystemAttemptProfile attemptProfile;
};

/// Prepares the exact gem5 System invocation selected by model kind 17, 18,
/// or 19. The model descriptor remains the sole engine selector.
llvm::Expected<evaluation::EvaluationModelProviderPreparation>
prepareGem5SystemInvocation(
    const evaluation::EvaluationRequest &request,
    const evaluation::CaseArtifactResolution &resolution,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const external_tool::ExternalToolPreparationContext &context);

/// Prepares an explicitly diagnostic gem5 invocation. Performance outputs are
/// part of this dedicated bundle contract and never part of ordinary model
/// preparation.
llvm::Expected<evaluation::EvaluationModelProviderPreparation>
prepareGem5SystemDiagnosticInvocation(
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

/// Imports the dedicated diagnostic contract after proving that the caller
/// executed the exact bundle as a fresh external attempt.
llvm::Expected<Gem5SystemDiagnosticEvaluation>
importGem5SystemDiagnosticInvocation(
    const evaluation::EvaluationRequest &request,
    const evaluation::CaseArtifactResolution &resolution,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const external_tool::ExternalToolInvocationExecutionObservation &execution,
    const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::runtime

#endif // LOOM_RUNTIME_GEM5SYSTEMEXECUTION_H
