#include "DSE/PreMappingExploration.h"

#include "Common/ArtifactStore.h"
#include "Common/ResolvedConfig.h"

#include "llvm/IR/Module.h"

#include <utility>
#include <vector>

namespace loom::dse {

llvm::Expected<PreMappingExplorationOutcome>
exploreLlvmModuleToPreMapping(std::unique_ptr<llvm::Module> module,
                              const fabric::FinalizedFabricRoot &fabric,
                              const ResolvedConfig &config,
                              const PreMappingExplorationOptions &options,
                              const ArtifactStore &artifactStore) {
  auto structured = frontend::raiseLlvmModuleToStructured(
      std::move(module), fabric, options.raising);
  if (!structured)
    return structured.takeError();
  auto explored = generateAndPromoteStructuredOwnership(
      structured->structuredProgram, fabric, config, options.ownership,
      artifactStore);
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
            structured->fabric, structured->staticGlobalMemory,
            std::move(candidate.candidate.structuredProgram),
            std::move(candidate.candidate.canonicalDataflow)},
        std::move(candidate.derivations)});
  }
  return PreMappingExplorationOutcome{CompletedPreMappingSelection{
      std::move(selected), std::move(selection.satisfiedEvidence)}};
}

} // namespace loom::dse
