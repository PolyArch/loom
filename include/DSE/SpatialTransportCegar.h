#ifndef LOOM_DSE_SPATIALTRANSPORTCEGAR_H
#define LOOM_DSE_SPATIALTRANSPORTCEGAR_H

#include "Common/Artifact.h"
#include "Evaluation/Models/CgraClosedWait.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "PnR/SpatialExactRepair.h"
#include "PnR/SpatialMappingWarmSeed.h"

#include "llvm/Support/Error.h"
#include "llvm/ADT/StringRef.h"

#include <chrono>
#include <cstdint>
#include <optional>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
struct ResolvedConfig;
} // namespace loom

namespace loom::dse {

struct SpatialTransportCegarPolicy final {
  std::uint64_t maximumIterations = 1;
  std::uint64_t maximumAccumulatedClauses = 1;
  std::uint64_t maximumSolverCallsPerIteration = 1;
  std::uint64_t maximumRuntimeEventFramesPerIteration = 1;
  std::optional<std::chrono::steady_clock::time_point> deadline;
};

enum class SpatialTransportCegarTermination : std::uint8_t {
  Retired,
  ProofNotEstablished,
  NoProgress,
  RepeatedCertificate,
  RepairTerminal,
  RuntimeIncomplete,
  IterationBudgetExhausted,
  ClauseBudgetExhausted,
  TimedOut,
};

llvm::StringRef spatialTransportCegarTerminationSpelling(
    SpatialTransportCegarTermination termination);

struct SpatialTransportCegarIteration final {
  struct WorkAccounting final {
    ExecutionResourceStatistics promotion;
    ExecutionResourceStatistics problemFreeze;
    ExecutionResourceStatistics warmSeed;
    ExecutionResourceStatistics exactRepair;
    ExecutionResourceStatistics childFinalization;
    ExecutionResourceStatistics runtimeEvaluation;
    ExecutionResourceStatistics evidenceVerification;
  };

  ArtifactRootReference parentMapping;
  ArtifactRootReference runtimeEvidence;
  ArtifactRootReference runtimeExecution;
  ArtifactRootReference evaluationRequest;
  ComponentViewDigest certificateDigest;
  ComponentViewDigest structureDigest;
  ArtifactRootReference accumulatedConstraints;
  pnr::SpatialMappingWarmSeedAccounting warmSeed;
  pnr::SpatialExactRepairResult repair;
  std::optional<ArtifactRootReference> childMapping;
  std::optional<ArtifactRootReference> childEvidence;
  bool retired = false;
  WorkAccounting work;
};

struct SpatialTransportCegarResult final {
  SpatialTransportCegarTermination termination =
      SpatialTransportCegarTermination::ProofNotEstablished;
  std::vector<SpatialTransportCegarIteration> iterations;
  std::optional<ArtifactRootReference> finalMapping;
  std::optional<ArtifactRootReference> finalConstraints;
  std::optional<ArtifactRootReference> finalEvidence;
};

/// Runs cumulative, Request-exact Spatial runtime CEGAR. Each iteration
/// promotes exactly one replay-verified certificate, canonical-unions it with
/// prior constraints, reconstructs the exact parent Mapping as a warm
/// Candidate, performs one finite literal-breaking repair, independently
/// finalizes/admit the child, and executes the same workload/runtime input.
llvm::Expected<SpatialTransportCegarResult> executeSpatialTransportCegar(
    const ArtifactRootReference &parentMapping,
    const ArtifactRootReference &parentConstraints,
    const evaluation::models::VerifiedCgraClosedWaitEvidence &parentEvidence,
    const ResolvedConfig &config,
    const fabric::FabricPhysicalTimingProfileView &physicalTiming,
    const SpatialTransportCegarPolicy &policy,
    const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::dse

#endif // LOOM_DSE_SPATIALTRANSPORTCEGAR_H
