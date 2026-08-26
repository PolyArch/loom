#ifndef LOOM_RUNTIME_GEM5SYSTEMEXECUTION_H
#define LOOM_RUNTIME_GEM5SYSTEMEXECUTION_H

#include "Evaluation/ModelProvider.h"
#include "Simulator/SimulationExecution.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace loom::runtime {

inline constexpr std::uint64_t gem5MaximumSpatialWork = 1'000'000;

/// Returns the exact integral distance between launch acceptance and graph
/// retirement. Fractional endpoint coordinates are accepted when their
/// difference is an unsigned integer.
std::optional<std::uint64_t> integralSpatialReferenceCycleDistance(
    const sim::SpatialProgressObservations &progress);

enum class Gem5SystemFactsSessionMode : std::uint8_t {
  ReuseEnclosing,
  Isolated,
};

struct Gem5SystemFactsOperationStatistics final {
  std::uint64_t invocations = 0;
  std::uint64_t wallNanoseconds = 0;
  std::uint64_t selfCpuNanoseconds = 0;
  std::uint64_t selfCpuObservationCount = 0;
  std::uint64_t childCpuNanoseconds = 0;
  std::uint64_t childCpuObservationCount = 0;
};

struct Gem5SystemFactsConstructionStatistics final {
  Gem5SystemFactsOperationStatistics deriveFacts;
  Gem5SystemFactsOperationStatistics systemInputsAndDeploymentImport;
  Gem5SystemFactsOperationStatistics bindingImport;
  Gem5SystemFactsOperationStatistics fabricImport;
  Gem5SystemFactsOperationStatistics systemMappingImport;
  Gem5SystemFactsOperationStatistics runtimeImageDerivation;
};

struct Gem5SystemFactsSessionStatistics final {
  std::uint64_t requests = 0;
  std::uint64_t cacheHits = 0;
  std::uint64_t cacheMisses = 0;
  std::uint64_t constructionAttempts = 0;
  std::uint64_t uniqueConstructions = 0;
  std::uint64_t uncachedConstructions = 0;
  std::uint64_t unsupportedConstructions = 0;
  std::uint64_t failedConstructions = 0;
  std::uint64_t revalidationCount = 0;
  std::uint64_t revalidatedArtifactBytes = 0;
  std::uint64_t revalidatedBlobBytes = 0;
  std::uint64_t constructionNanoseconds = 0;
  std::uint64_t constructionNanosecondsSaved = 0;
  std::uint64_t minimumRetainedBytes = 0;
  std::uint64_t entryCount = 0;
  Gem5SystemFactsConstructionStatistics construction;
};

/// Bounded immutable facts cache for one exact ArtifactStore/BlobStore
/// verification domain. Hits revalidate the complete cold-construction
/// closure and never bypass Request verification or typed importers.
class Gem5SystemFactsSession final {
public:
  class Impl;

  Gem5SystemFactsSession(const ArtifactStore &artifacts, const BlobStore &blobs,
                         Gem5SystemFactsSessionMode mode =
                             Gem5SystemFactsSessionMode::ReuseEnclosing,
                         std::size_t entryLimit = 8);
  ~Gem5SystemFactsSession();

  Gem5SystemFactsSession(const Gem5SystemFactsSession &) = delete;
  Gem5SystemFactsSession &operator=(const Gem5SystemFactsSession &) = delete;

  Gem5SystemFactsSessionStatistics statistics() const;

private:
  std::shared_ptr<Impl> active_;
  std::shared_ptr<Impl> previous_;
};

struct Gem5CgraEngineAttemptProfile final {
  std::uint64_t invocationCount = 0;
  std::uint64_t activeWallNanoseconds = 0;
  std::uint64_t activeProcessCpuNanoseconds = 0;
  std::uint64_t eventFrameCount = 0;
};

struct Gem5SystemAttemptProfile final {
  std::uint64_t configurationWallNanoseconds = 0;
  std::uint64_t engineStartupWallNanoseconds = 0;
  std::uint64_t simulationWallNanoseconds = 0;
  std::uint64_t gem5SimulationProcessCpuNanoseconds = 0;
  std::uint64_t observationWallNanoseconds = 0;
  std::uint64_t observationProcessCpuNanoseconds = 0;
  std::optional<std::uint64_t> engineProcessCpuNanoseconds;
  std::uint64_t bridgeCallbackCpuNanoseconds = 0;
  std::uint64_t bridgeEngineWaitNanoseconds = 0;
  std::uint64_t bridgeMessageCount = 0;
  std::uint64_t acceleratorInvocationCount = 0;
  std::uint64_t bridgeClockFailureCount = 0;
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
  std::optional<std::uint64_t> acceleratorReferenceCycles;
};

/// Attempt-local operational observations from one explicitly fresh gem5
/// diagnostic invocation. The ordinary Evaluation provider never requires or
/// imports these observations.
struct Gem5SystemDiagnosticEvaluation final {
  evaluation::EvaluationEvidence evidence;
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

/// Validates the Request and prepares an explicitly diagnostic gem5
/// invocation. Performance outputs are part of this dedicated bundle contract
/// and never part of ordinary model preparation.
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

/// Validates the Request and imports the dedicated diagnostic contract after
/// proving that the caller executed the exact bundle as a fresh external
/// attempt. No execution artifact is published before both checks succeed.
llvm::Expected<Gem5SystemDiagnosticEvaluation>
importGem5SystemDiagnosticInvocation(
    const evaluation::EvaluationRequest &request,
    const evaluation::CaseArtifactResolution &resolution,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const external_tool::ExternalToolInvocationExecutionObservation &execution,
    const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::runtime

#endif // LOOM_RUNTIME_GEM5SYSTEMEXECUTION_H
