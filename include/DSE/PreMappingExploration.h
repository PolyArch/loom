#ifndef LOOM_DSE_PREMAPPINGEXPLORATION_H
#define LOOM_DSE_PREMAPPINGEXPLORATION_H

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
};

struct CompletedPreMappingSelection final {
  std::vector<SelectedPreMappingCompilation> selected;
  std::vector<ArtifactRootReference> satisfiedEvidence;
  std::vector<StructuredOwnershipCandidateDisposition> dispositions;
};

using PreMappingExplorationOutcome =
    std::variant<CompletedPreMappingSelection, CompletedNoFeasibleCandidate,
                 IncompleteSelection>;

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
