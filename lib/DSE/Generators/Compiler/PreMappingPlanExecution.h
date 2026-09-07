#ifndef LOOM_LIB_DSE_COMPILER_PREMAPPINGPLANEXECUTION_H
#define LOOM_LIB_DSE_COMPILER_PREMAPPINGPLANEXECUTION_H

#include "DSE/PreMappingExploration.h"
#include "DSE/StructuredOwnershipCandidateGenerator.h"
#include "DSE/StructuredScheduleCandidateGenerator.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace loom::dse::pre_mapping {

// The coordinator consumes completed selections and typed incompleteness.
// Plan construction, acquisition, quality gates, and result interpretation
// stay together in this private execution module.
struct CompletedOwnershipSelection final {
  std::unique_ptr<StructuredOwnershipInvocation> invocation;
  std::vector<ArtifactRootReference> selected;
  std::vector<ArtifactRootReference> preferenceOrder;
  std::vector<ArtifactRootReference> evidence;
  std::vector<StructuredOwnershipCandidateDisposition> dispositions;
  std::vector<StructuredOwnershipFinalizationRejection> finalizationRejections;
  DsePlanGenerateInvocationRecords generateInvocations;
  std::optional<RetainedDsePlanIncompleteness> retainedIncompleteness;
  std::uint64_t programMaterializations = 0;
  std::uint64_t analyticEvaluations = 0;
  std::uint64_t functionalReplays = 0;
  StructuredOwnershipEvaluationTiming evaluationTiming;
};
using OwnershipSelectionOutcome =
    std::variant<CompletedOwnershipSelection, IncompletePreMappingExploration>;
struct CompletedDataflowSelection final {
  std::vector<ArtifactRootReference> selected;
  std::vector<ArtifactRootReference> preferenceOrder;
  std::vector<ArtifactRootReference> evidence;
  DsePlanGenerateInvocationRecords generateInvocations;
  std::optional<RetainedDsePlanIncompleteness> retainedIncompleteness;
};
using DataflowSelectionOutcome =
    std::variant<CompletedDataflowSelection, IncompletePreMappingExploration>;

llvm::Error invalid(const llvm::Twine &message);

bool isCancellationReason(const DsePlanIncompleteReason &reason);

PreMappingCandidatePlanningDisposition
planningDispositionForIncomplete(const DsePlanIncompleteReason &reason);

void emitOwnershipRejections(
    llvm::ArrayRef<StructuredOwnershipCandidateDisposition> dispositions);

void mergeReferences(std::vector<ArtifactRootReference> &destination,
                     llvm::ArrayRef<ArtifactRootReference> source);

llvm::Expected<OwnershipSelectionOutcome> exploreOwnershipCandidates(
    const frontend::StructuredCompilation &generationParent,
    const ArtifactRootReference &generationParentReference,
    const frontend::StructuredProgramCandidate &sourceProgram,
    const ArtifactRootReference &sourceReference,
    const sim::CanonicalSimulationWorkload &workload,
    const ArtifactRootReference &workloadReference,
    const sim::CanonicalSimulationRuntimeInput &runtimeInput,
    const ArtifactRootReference &runtimeInputReference,
    const fabric::FinalizedFabricRoot &fabric, const ResolvedConfig &config,
    const StructuredOwnershipExplorationOptions &options,
    StructuredOwnershipGenerationIntent ownershipIntent,
    StructuredScheduleGenerationIntent scheduleIntent,
    std::uint64_t ownershipMaterializationAttemptLimit,
    std::uint64_t specialMathMaterializationAttemptLimit,
    std::uint64_t expansionLimit, bool generationParentFunctionallyVerified,
    bool requireFunctionalReplay, ExecutionControlView executionControl,
    const ArtifactStore &artifactStore, const BlobStore &blobStore,
    const StructuredOwnershipSharedEvaluation *sharedEvaluation);

llvm::Expected<DataflowSelectionOutcome> exploreDataflowCandidates(
    const ArtifactRootReference &d0,
    const ArtifactRootReference &structuredParent,
    const fabric::FinalizedFabricRoot &fabric,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ResolvedConfig &config,
    const StructuredOwnershipTopKSelection &selection,
    StructuredOwnershipSelectionMode selectionMode,
    bool allowRewriteExploration, ExecutionControlView executionControl,
    const ArtifactStore &store, const BlobStore &blobs);

} // namespace loom::dse::pre_mapping

#endif // LOOM_LIB_DSE_COMPILER_PREMAPPINGPLANEXECUTION_H
