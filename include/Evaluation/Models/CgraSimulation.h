#ifndef LOOM_EVALUATION_MODELS_CGRASIMULATION_H
#define LOOM_EVALUATION_MODELS_CGRASIMULATION_H

#include "Evaluation/Evidence.h"
#include "Simulator/CGRAAdmission.h"
#include "Simulator/CGRASimulator.h"
#include "Simulator/SimulationArtifacts.h"

#include <chrono>
#include <cstdint>
#include <optional>

namespace loom {
class ArtifactStore;
class BlobStore;
struct ResolvedConfig;
} // namespace loom

namespace loom::evaluation::models {

struct ResolvedCgraSimulationCase final {
  ArtifactRootReference canonicalDataflow;
  ArtifactRootReference fabric;
  CaseArtifactResolution resolution;
};

struct PreparedCgraSimulationEvaluation final {
  EvaluationRequest request;
  CaseArtifactResolution resolution;
  sim::PreparedCgraExecution execution;
  sim::CanonicalSimulationWorkload workload;
  sim::CanonicalSimulationRuntimeInput runtimeInput;
  sim::PreparedCgraWorkloadExecution workloadExecution;
  sim::PreparedSpatialExecutionContext executionContext;
};

struct CgraSimulationAttemptLimits final {
  std::uint64_t maxEventFrames = 100000;
  std::optional<std::chrono::steady_clock::time_point> executionDeadline;
};

struct CgraSimulationAttemptProfile final {
  std::uint64_t activeWallNanoseconds = 0;
  std::optional<std::uint64_t> processCpuNanoseconds;
  std::uint64_t inputLoadWallNanoseconds = 0;
  std::optional<std::uint64_t> inputLoadCpuNanoseconds;
  std::uint64_t engineActiveWallNanoseconds = 0;
  std::optional<std::uint64_t> engineActiveCpuNanoseconds;
  std::uint64_t observationProjectionWallNanoseconds = 0;
  std::optional<std::uint64_t> observationProjectionCpuNanoseconds;
  std::uint64_t artifactPublicationWallNanoseconds = 0;
  std::optional<std::uint64_t> artifactPublicationCpuNanoseconds;
  sim::CgraSimulationCounters counters;
};

struct CgraSimulationEvaluation final {
  EvaluationEvidence evidence;
  std::optional<sim::CgraClosedWaitSetDiagnostic> closedWait;
  std::optional<sim::CgraUnsupportedMemoryContract> unsupportedMemoryContract;
  CgraSimulationAttemptProfile attemptProfile;
};

llvm::Error registerCgraSimulationModel();

EvaluationModelDescriptorRef cgraSimulationModelDescriptorRef();
CaseSubjectRoleRef cgraSimulationProgramRole();
CaseSubjectRoleRef cgraSimulationHardwareRole();
CaseSubjectRoleRef cgraSimulationSpatialMappingRole();

/// Strictly resolves one SpatialMapping-rooted CGRA case from owner lineage.
/// The returned references are ordinary Artifact roots reconstructed from the
/// imported Mapping owners; no caller-provided D/F participates.
llvm::Expected<ResolvedCgraSimulationCase>
resolveCgraSimulationCase(const ArtifactRootReference &spatialMapping,
                          const ArtifactRootReference &workload,
                          const ArtifactRootReference &runtimeInput,
                          const ArtifactStore &artifactStore);

llvm::Expected<PreparedCgraSimulationEvaluation>
prepareCgraSimulationEvaluation(const ArtifactRootReference &canonicalDataflow,
                                const ArtifactRootReference &fabric,
                                const ArtifactRootReference &spatialMapping,
                                const ArtifactRootReference &workload,
                                const ArtifactRootReference &runtimeInput,
                                const ResolvedConfig &config,
                                const ArtifactStore &artifactStore,
                                const BlobStore &blobStore);

llvm::Expected<EvaluationEvidence>
evaluateCgraSimulation(const PreparedCgraSimulationEvaluation &prepared,
                       CgraSimulationAttemptLimits limits,
                       const ArtifactStore &artifactStore,
                       const BlobStore &blobStore);

/// Same ordinary Evaluation request/evidence transaction with the exact
/// in-process halt diagnostic retained for a caller that owns a typed hardware
/// feedback loop. The diagnostic is ephemeral and cannot replace Evidence or
/// Mapping verification.
llvm::Expected<CgraSimulationEvaluation> evaluateCgraSimulationWithDiagnostics(
    const PreparedCgraSimulationEvaluation &prepared,
    CgraSimulationAttemptLimits limits, const ArtifactStore &artifactStore,
    const BlobStore &blobStore);

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_CGRASIMULATION_H
