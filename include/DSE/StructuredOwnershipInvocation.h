#ifndef LOOM_DSE_STRUCTUREDOWNERSHIPINVOCATION_H
#define LOOM_DSE_STRUCTUREDOWNERSHIPINVOCATION_H

#include "DSE/StructuredOwnership.h"
#include "Evaluation/Models/StructuredEvaluationInvocationCache.h"
#include "Common/ArtifactLocalReference.h"
#include "Common/ExecutionControl.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace loom {
class ArtifactStore;
struct ResolvedConfig;
} // namespace loom

namespace loom::sim {
struct NativeStructuredProgramObservations;
}

namespace loom::dse {

namespace detail {
class StructuredOwnershipInvocationAccess;
}

struct StructuredOwnershipSharedEvaluationStatistics final {
  std::uint64_t profileCacheHits = 0;
  std::uint64_t profileCacheMisses = 0;
  std::uint64_t profileSingleFlightWaits = 0;
};

/// Wall time spent inside the two model-owned promotion paths for one
/// invocation.  This is deliberately narrower than the surrounding plan
/// execution time: orchestration, artifact materialization, and deferred
/// selection are accounted by their owning frontier counters.
struct StructuredOwnershipEvaluationTiming final {
  std::uint64_t analyticCalls = 0;
  std::uint64_t analyticElapsedNanoseconds = 0;
  std::uint64_t functionalReplayCalls = 0;
  std::uint64_t functionalReplayElapsedNanoseconds = 0;
};

/// One build-local immutable source observation and exact-key Evaluation cache
/// shared by independent ownership generations. It has no Artifact identity
/// and cannot relax any model key or reuse a candidate result across builds.
class StructuredOwnershipSharedEvaluation final {
public:
  StructuredOwnershipSharedEvaluation(
      const sim::NativeStructuredProgramObservations &sourceObservations,
      evaluation::models::StructuredEvaluationInvocationCache &cache)
      : sourceObservations_(sourceObservations), cache_(cache) {}

  const sim::NativeStructuredProgramObservations &sourceObservations() const {
    return sourceObservations_;
  }
  evaluation::models::StructuredEvaluationInvocationCache &cache() const {
    return cache_;
  }
  StructuredOwnershipSharedEvaluationStatistics statistics() const;

  llvm::Expected<
      std::shared_ptr<const sim::NativeStructuredProgramObservations>>
  profiledObservations(
      const ArtifactRootReference &candidate,
      const ArtifactRootReference &source,
      const ArtifactRootReference &workload,
      const ArtifactRootReference &runtimeInput,
      const frontend::StructuredProgramCandidate &candidateProgram,
      const frontend::StructuredProgramCandidate &sourceProgram,
      const sim::CanonicalSimulationWorkload &simulationWorkload,
      const sim::CanonicalSimulationRuntimeInput &simulationRuntimeInput) const;

private:
  struct ProfileKey final {
    ArtifactRootReference candidate;
    ArtifactRootReference source;
    ArtifactRootReference workload;
    ArtifactRootReference runtimeInput;

    friend bool operator<(const ProfileKey &lhs, const ProfileKey &rhs) {
      const ArtifactRootReference *left[] = {
          &lhs.candidate, &lhs.source, &lhs.workload, &lhs.runtimeInput};
      const ArtifactRootReference *right[] = {
          &rhs.candidate, &rhs.source, &rhs.workload, &rhs.runtimeInput};
      for (std::size_t index = 0; index != 4; ++index) {
        if (artifactRootReferenceLess(*left[index], *right[index]))
          return true;
        if (artifactRootReferenceLess(*right[index], *left[index]))
          return false;
      }
      return false;
    }
  };

  struct ProfileEntry final {
    bool inFlight = true;
    std::shared_ptr<const sim::NativeStructuredProgramObservations>
        observations;
  };

  const sim::NativeStructuredProgramObservations &sourceObservations_;
  evaluation::models::StructuredEvaluationInvocationCache &cache_;
  mutable std::mutex profileMutex_;
  mutable std::condition_variable profileChanged_;
  mutable std::map<ProfileKey, ProfileEntry> profiles_;
  mutable StructuredOwnershipSharedEvaluationStatistics statistics_;
};

/// Removable typed state shared by the compiler-owned Generate and Promote
/// nodes of one synchronous Structured DSE invocation. Persistent candidate
/// and Evidence identity remains owned by the ordinary Artifact families.
class StructuredOwnershipInvocation final {
public:
  StructuredOwnershipInvocation(
      const frontend::StructuredProgramCandidate &generationParent,
      const frontend::StructuredProgramCandidate &sourceProgram,
      const sim::CanonicalSimulationWorkload &workload,
      const sim::CanonicalSimulationRuntimeInput &runtimeInput,
      const fabric::FinalizedFabricRoot &fabric, const ResolvedConfig &config,
      const lowering::CanonicalDataflowLoweringOptions &lowering,
      std::uint32_t candidateWorkerCount,
      sim::SourceBackedDfgValidationLimits functionalReplayLimits,
      llvm::ArrayRef<frontend::StructuredOperationSourceProvenance>
          sourceProvenance = {},
      const StructuredOwnershipSharedEvaluation *sharedEvaluation = nullptr,
      ExecutionControlView executionControl = {},
      bool generationParentFunctionallyVerified = true);
  ~StructuredOwnershipInvocation();

  StructuredOwnershipInvocation(const StructuredOwnershipInvocation &) = delete;
  StructuredOwnershipInvocation &
  operator=(const StructuredOwnershipInvocation &) = delete;

  llvm::ArrayRef<StructuredOwnershipCandidateDisposition> dispositions() const;

  llvm::Error prepareInputs(const ArtifactRootReference &generationParent,
                            const ArtifactRootReference &source,
                            const ArtifactRootReference &workload,
                            const ArtifactRootReference &runtimeInput,
                            const ArtifactStore &store);

  std::uint64_t sourceNativeExecutionCount() const;
  evaluation::models::StructuredEvaluationInvocationCacheStatistics
  evaluationCacheStatistics() const;

  StructuredOwnershipEvaluationTiming evaluationTiming() const;

  /// Ensures that a terminal Promote survivor has the transient replay view
  /// required by later Dataflow/application materialization. Persisted
  /// finding Evidence alone is insufficient because it does not contain the
  /// replay cases. Exact in-memory hits remain no-execution cache hits.
  llvm::Error ensureSelectedCandidateFunctionalReplay(
      const ArtifactRootReference &candidate, const ArtifactStore &store);

  llvm::Expected<SelectedStructuredOwnershipCandidate>
  materializeSelectedCandidate(const ArtifactRootReference &candidate,
                               const ArtifactStore &store);

  /// Materializes one analytically promoted candidate only as the immutable
  /// parent of a later transformation layer. It carries no functional replay
  /// and cannot be published as a terminal pre-Mapping selection.
  llvm::Expected<SelectedStructuredOwnershipCandidate>
  materializeAnalyticContinuationCandidate(
      const ArtifactRootReference &candidate, const ArtifactStore &store);

  /// Returns whether the candidate's already materialized Canonical Dataflow
  /// contains a non-scalar root thread domain. The Dataflow projection is the
  /// semantic authority for temporal execution regardless of whether the
  /// domain originated in ownership or schedule lineage.
  llvm::Expected<bool> selectedCandidateHasLogicalThreadDomain(
      const ArtifactRootReference &candidate) const;

  llvm::Expected<ArtifactRootReference>
  prepareDataflowGeneration(const ArtifactRootReference &structuredParent,
                            const ArtifactStore &store);

  llvm::Expected<SelectedStructuredOwnershipCandidate>
  materializeSelectedDataflowCandidate(
      const ArtifactRootReference &structuredParent,
      const ArtifactRootReference &dataflowCandidate,
      const ArtifactStore &store);

private:
  llvm::Expected<SelectedStructuredOwnershipCandidate>
  materializeCandidate(const ArtifactRootReference &candidate,
                       const ArtifactStore &store,
                       bool requireFunctionalReplay);

  class Impl;
  std::unique_ptr<Impl> impl_;

  friend class detail::StructuredOwnershipInvocationAccess;
};

/// Binds one Structured invocation and its Evaluation cache to the current
/// synchronous central-plan execution. Nested scopes restore both bindings.
class StructuredOwnershipInvocationScope final {
public:
  explicit StructuredOwnershipInvocationScope(
      StructuredOwnershipInvocation &invocation);
  ~StructuredOwnershipInvocationScope();

  StructuredOwnershipInvocationScope(
      const StructuredOwnershipInvocationScope &) = delete;
  StructuredOwnershipInvocationScope &
  operator=(const StructuredOwnershipInvocationScope &) = delete;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace loom::dse

#endif // LOOM_DSE_STRUCTUREDOWNERSHIPINVOCATION_H
