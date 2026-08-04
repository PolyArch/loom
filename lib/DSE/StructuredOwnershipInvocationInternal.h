#ifndef LOOM_LIB_DSE_STRUCTUREDOWNERSHIPINVOCATIONINTERNAL_H
#define LOOM_LIB_DSE_STRUCTUREDOWNERSHIPINVOCATIONINTERNAL_H

#include "DSE/StructuredOwnershipInvocation.h"
#include "Evaluation/Models/StructuredEvaluationInvocationCache.h"
#include "Simulator/NativeSimulationOracle.h"

namespace loom::dse::detail {

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
                    StructuredOwnershipGenerationOptions &options);

  static const ResolvedConfig &
  config(const StructuredOwnershipInvocation &invocation);
  static evaluation::models::StructuredEvaluationInvocationCache &
  evaluationCache(StructuredOwnershipInvocation &invocation);
  static llvm::ArrayRef<frontend::StructuredOperationSourceProvenance>
  sourceProvenance(const StructuredOwnershipInvocation &invocation);

  static llvm::Error recordGeneration(
      StructuredOwnershipInvocation &invocation,
      ArtifactRootReference sourceReference,
      ArtifactRootReference workloadReference,
      ArtifactRootReference runtimeInputReference,
      sim::NativeStructuredProgramObservations sourceObservations,
      llvm::ArrayRef<StructuredOwnershipCandidateDisposition> dispositions);

  static llvm::Error
  primeFunctionalReplay(StructuredOwnershipInvocation &invocation,
                        const ArtifactRootReference &candidate,
                        const ArtifactStore &store);
};

} // namespace loom::dse::detail

#endif // LOOM_LIB_DSE_STRUCTUREDOWNERSHIPINVOCATIONINTERNAL_H
