#ifndef LOOM_DSE_PREMAPPINGEXPLORATION_H
#define LOOM_DSE_PREMAPPINGEXPLORATION_H

#include "DSE/Plan.h"
#include "DSE/StructuredOwnership.h"
#include "Frontend/Compilation/PreMappingCompilation.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <variant>
#include <vector>

namespace loom::dse {

struct PreMappingExplorationOptions final {
  StructuredOwnershipExplorationOptions ownership;
};

struct SelectedPreMappingCompilation final {
  /// Invocation-local rank in the bounded software frontier. This value is
  /// derived from completed objective Evidence and does not enter any
  /// compilation or Dataflow Artifact identity.
  std::uint64_t preferenceRank = 0;
  frontend::PreMappingCompilation compilation;
  std::vector<StructuredOwnershipDerivation> derivations;
  std::vector<StructuredExecutionShapeDerivation> executionShapeDerivations;
  std::vector<StructuredSpecialMathAccuracyDerivation>
      specialMathAccuracyDerivations;
  std::vector<StructuredScheduleDerivation> scheduleDerivations;
  std::vector<StructuredMemoryCommunicationDerivation>
      memoryCommunicationDerivations;
  std::vector<DataflowRewriteDerivation> dataflowRewriteDerivations;
  std::optional<sim::SourceBackedDfgValidationResult> functionalReplay;
};

struct CompletedPreMappingSelection final {
  std::vector<SelectedPreMappingCompilation> selected;
  std::vector<ArtifactRootReference> satisfiedEvidence;
  std::vector<StructuredOwnershipCandidateDisposition> dispositions;
  std::vector<DsePlanGenerateInvocationRecords> planGenerateInvocations;
};

struct CompletedPreMappingNoFeasibleCandidate final {
  std::vector<ArtifactRootReference> satisfiedEvidence;
  std::vector<DsePlanGenerateInvocationRecords> planGenerateInvocations;
};

struct IncompletePreMappingExploration final {
  std::optional<std::uint64_t> planNodeOrdinal;
  DsePlanIncompleteReason reason;
  std::vector<ArtifactRootReference> retainedEvidence;
  std::vector<DsePlanGenerateInvocationRecords> planGenerateInvocations;
};

using PreMappingExplorationOutcome =
    std::variant<CompletedPreMappingSelection,
                 CompletedPreMappingNoFeasibleCandidate,
                 IncompletePreMappingExploration>;

llvm::Expected<PreMappingExplorationOutcome>
exploreStructuredCompilationToPreMapping(
    frontend::StructuredCompilation compilation,
    const sim::CanonicalSimulationWorkload &workload,
    const sim::CanonicalSimulationRuntimeInput &runtimeInput,
    const fabric::FinalizedFabricRoot &fabric, const ResolvedConfig &config,
    const PreMappingExplorationOptions &options,
    const ArtifactStore &artifactStore, const BlobStore &blobStore);

} // namespace loom::dse

#endif // LOOM_DSE_PREMAPPINGEXPLORATION_H
