#include "DSE/PreMappingExploration.h"

#include "Common/ArtifactStore.h"
#include "Common/ResolvedConfig.h"

#include <utility>
#include <vector>

namespace loom::dse {

llvm::Expected<PreMappingExplorationOutcome>
exploreStructuredCompilationToPreMapping(
    frontend::StructuredCompilation compilation,
    const sim::CanonicalSimulationWorkload &workload,
    const sim::CanonicalSimulationRuntimeInput &runtimeInput,
    const fabric::FinalizedFabricRoot &fabric, const ResolvedConfig &config,
    const PreMappingExplorationOptions &options,
    const ArtifactStore &artifactStore) {
  if (compilation.fabric != fabric.reference())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "pre_mapping_exploration_invalid: Structured compilation and Fabric "
        "references differ");
  auto explored = generateAndPromoteStructuredOwnership(
      compilation.structuredProgram, workload, runtimeInput, fabric, config,
      options.ownership, artifactStore);
  if (!explored)
    return explored.takeError();
  if (const auto *incomplete = std::get_if<IncompleteSelection>(&*explored))
    return PreMappingExplorationOutcome{*incomplete};
  if (std::holds_alternative<CompletedNoFeasibleCandidate>(*explored))
    return PreMappingExplorationOutcome{CompletedNoFeasibleCandidate{}};

  auto selection =
      std::get<CompletedStructuredOwnershipSelection>(std::move(*explored));
  std::vector<SelectedPreMappingCompilation> selected;
  selected.reserve(selection.selected.size());
  for (auto &candidate : selection.selected) {
    selected.push_back(SelectedPreMappingCompilation{
        frontend::PreMappingCompilation{
            compilation.fabric, compilation.staticGlobalMemory,
            std::move(candidate.candidate.structuredProgram),
            std::move(candidate.candidate.canonicalDataflow)},
        std::move(candidate.derivations)});
  }
  return PreMappingExplorationOutcome{CompletedPreMappingSelection{
      std::move(selected), std::move(selection.satisfiedEvidence),
      std::move(selection.dispositions)}};
}

} // namespace loom::dse
