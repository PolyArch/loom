#ifndef LOOM_DSE_PREMAPPINGEXPLORATION_H
#define LOOM_DSE_PREMAPPINGEXPLORATION_H

#include "DSE/Plan.h"
#include "DSE/StructuredOwnership.h"
#include "Frontend/Compilation/PreMappingCompilation.h"

#include "llvm/Support/Error.h"

#include <variant>
#include <vector>

namespace loom::dse {

struct PreMappingExplorationOptions final {
  StructuredOwnershipExplorationOptions ownership;
};

struct SelectedPreMappingCompilation final {
  frontend::PreMappingCompilation compilation;
  std::vector<StructuredOwnershipDerivation> derivations;
  std::vector<StructuredExecutionShapeDerivation> executionShapeDerivations;
  std::vector<StructuredScheduleDerivation> scheduleDerivations;
  std::optional<sim::SourceBackedDfgValidationResult> functionalReplay;
};

struct CompletedPreMappingSelection final {
  std::vector<SelectedPreMappingCompilation> selected;
  std::vector<ArtifactRootReference> satisfiedEvidence;
  std::vector<StructuredOwnershipCandidateDisposition> dispositions;
};

struct IncompletePreMappingExploration final {
  std::optional<std::uint64_t> planNodeOrdinal;
  DsePlanIncompleteReason reason;
  std::vector<ArtifactRootReference> retainedEvidence;
};

using PreMappingExplorationOutcome =
    std::variant<CompletedPreMappingSelection, CompletedNoFeasibleCandidate,
                 IncompletePreMappingExploration>;

llvm::Expected<PreMappingExplorationOutcome>
exploreStructuredCompilationToPreMapping(
    frontend::StructuredCompilation compilation,
    const sim::CanonicalSimulationWorkload &workload,
    const sim::CanonicalSimulationRuntimeInput &runtimeInput,
    const fabric::FinalizedFabricRoot &fabric, const ResolvedConfig &config,
    const PreMappingExplorationOptions &options,
    const ArtifactStore &artifactStore);

} // namespace loom::dse

#endif // LOOM_DSE_PREMAPPINGEXPLORATION_H
