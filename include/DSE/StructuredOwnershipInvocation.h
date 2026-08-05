#ifndef LOOM_DSE_STRUCTUREDOWNERSHIPINVOCATION_H
#define LOOM_DSE_STRUCTUREDOWNERSHIPINVOCATION_H

#include "DSE/StructuredOwnership.h"
#include "Evaluation/Models/StructuredEvaluationInvocationCache.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>

namespace loom {
class ArtifactStore;
struct ResolvedConfig;
} // namespace loom

namespace loom::dse {

namespace detail {
class StructuredOwnershipInvocationAccess;
}

/// Removable typed state shared by the compiler-owned Generate and Promote
/// nodes of one synchronous Structured DSE invocation. Persistent candidate
/// and Evidence identity remains owned by the ordinary Artifact families.
class StructuredOwnershipInvocation final {
public:
  StructuredOwnershipInvocation(
      const frontend::StructuredProgramCandidate &sourceProgram,
      const sim::CanonicalSimulationWorkload &workload,
      const sim::CanonicalSimulationRuntimeInput &runtimeInput,
      const fabric::FinalizedFabricRoot &fabric, const ResolvedConfig &config,
      const lowering::CanonicalDataflowLoweringOptions &lowering,
      std::uint32_t candidateWorkerCount,
      sim::SourceBackedDfgValidationLimits functionalReplayLimits,
      llvm::ArrayRef<frontend::StructuredOperationSourceProvenance>
          sourceProvenance = {});
  ~StructuredOwnershipInvocation();

  StructuredOwnershipInvocation(const StructuredOwnershipInvocation &) = delete;
  StructuredOwnershipInvocation &
  operator=(const StructuredOwnershipInvocation &) = delete;

  llvm::ArrayRef<StructuredOwnershipCandidateDisposition> dispositions() const;

  llvm::Error prepareSource(const ArtifactRootReference &source,
                            const ArtifactRootReference &workload,
                            const ArtifactRootReference &runtimeInput,
                            const ArtifactStore &store);

  std::uint64_t sourceNativeExecutionCount() const;
  evaluation::models::StructuredEvaluationInvocationCacheStatistics
  evaluationCacheStatistics() const;

  llvm::Expected<SelectedStructuredOwnershipCandidate>
  materializeSelectedCandidate(const ArtifactRootReference &candidate,
                               const ArtifactStore &store);

  llvm::Expected<ArtifactRootReference>
  prepareDataflowGeneration(const ArtifactRootReference &structuredParent,
                            const ArtifactStore &store);

  llvm::Expected<SelectedStructuredOwnershipCandidate>
  materializeSelectedDataflowCandidate(
      const ArtifactRootReference &structuredParent,
      const ArtifactRootReference &dataflowCandidate,
      const ArtifactStore &store);

private:
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
