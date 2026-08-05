#ifndef LOOM_DSE_STRUCTUREDOWNERSHIPINVOCATIONINTERNAL_H
#define LOOM_DSE_STRUCTUREDOWNERSHIPINVOCATIONINTERNAL_H

#include "DSE/StructuredOwnershipInvocation.h"
#include "Evaluation/Models/StructuredEvaluationInvocationCache.h"
#include "Frontend/Compilation/StructuredSchedule.h"
#include "Simulator/NativeSimulationOracle.h"

namespace loom::dse::detail {

struct StructuredOwnershipPreparedSource final {
  const ArtifactRootReference &sourceReference;
  const ArtifactRootReference &workloadReference;
  const ArtifactRootReference &runtimeInputReference;
  const sim::NativeStructuredProgramObservations &observations;
};

struct StructuredOwnershipCandidateState final {
  ArtifactRootReference reference;
  frontend::MaterializedStructuredOwnershipCandidate candidate;
};

class StructuredOwnershipInvocationAccess final {
public:
  static StructuredOwnershipInvocation *current();
  static StructuredOwnershipInvocation *
  bind(StructuredOwnershipInvocation *invocation);

  static llvm::Error
  prepareGeneration(StructuredOwnershipInvocation &invocation,
                    const frontend::StructuredProgramCandidate &sourceProgram,
                    const sim::CanonicalSimulationWorkload &workload,
                    const sim::CanonicalSimulationRuntimeInput &runtimeInput,
                    const fabric::FinalizedFabricRoot &fabric,
                    const ArtifactStore &store,
                    StructuredOwnershipGenerationOptions &options);

  static llvm::Expected<StructuredOwnershipPreparedSource>
  preparedSource(const StructuredOwnershipInvocation &invocation);

  static const ResolvedConfig &
  config(const StructuredOwnershipInvocation &invocation);
  static const lowering::CanonicalDataflowLoweringOptions &
  loweringOptions(const StructuredOwnershipInvocation &invocation);

  static const fabric::FinalizedFabricRoot &
  fabric(const StructuredOwnershipInvocation &invocation);
  static evaluation::models::StructuredEvaluationInvocationCache &
  evaluationCache(StructuredOwnershipInvocation &invocation);
  static llvm::ArrayRef<frontend::StructuredOperationSourceProvenance>
  sourceProvenance(const StructuredOwnershipInvocation &invocation);

  static llvm::Error recordGeneration(
      StructuredOwnershipInvocation &invocation,
      ArtifactRootReference sourceReference,
      ArtifactRootReference workloadReference,
      ArtifactRootReference runtimeInputReference,
      llvm::ArrayRef<StructuredOwnershipCandidateDisposition> dispositions,
      std::vector<StructuredOwnershipCandidateState> candidates,
      const ArtifactStore &store);

  static llvm::Expected<frontend::MaterializedStructuredOwnershipCandidate>
  cloneOwnershipCandidate(StructuredOwnershipInvocation &invocation,
                          const ArtifactRootReference &reference);

  static llvm::Error recordScheduleCandidate(
      StructuredOwnershipInvocation &invocation,
      const ArtifactRootReference &parent, const ArtifactRootReference &child,
      const frontend::StructuredScheduleDecision &decision,
      frontend::MaterializedStructuredScheduleCandidate candidate,
      lowering::ProjectedCanonicalDataflow projected,
      const ArtifactStore &store);

  static llvm::Error recordExecutionShapeCandidate(
      StructuredOwnershipInvocation &invocation,
      const ArtifactRootReference &parent, const ArtifactRootReference &child,
      std::optional<frontend::StructuredExecutionShapeDecision> decision,
      frontend::MaterializedStructuredOwnershipCandidate candidate,
      lowering::ProjectedCanonicalDataflow projected,
      const ArtifactStore &store);

  static llvm::Error recordDataflowRewriteCandidate(
      StructuredOwnershipInvocation &invocation,
      const ArtifactRootReference &parent, const ArtifactRootReference &child,
      const dataflow::DataflowRewriteDecision &decision,
      const ArtifactStore &store);

  static llvm::Error
  primeAnalyticCandidate(StructuredOwnershipInvocation &invocation,
                         const ArtifactRootReference &candidate,
                         const ArtifactStore &store);

  static llvm::Error
  primeFunctionalReplay(StructuredOwnershipInvocation &invocation,
                        const ArtifactRootReference &candidate,
                        const ArtifactStore &store);

  static llvm::Error
  primeDataflowFunctionalReplay(StructuredOwnershipInvocation &invocation,
                                const ArtifactRootReference &structuredParent,
                                const ArtifactRootReference &dataflowCandidate,
                                const ArtifactStore &store);
};

} // namespace loom::dse::detail

#endif // LOOM_DSE_STRUCTUREDOWNERSHIPINVOCATIONINTERNAL_H
