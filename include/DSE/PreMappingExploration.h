#ifndef LOOM_DSE_PREMAPPINGEXPLORATION_H
#define LOOM_DSE_PREMAPPINGEXPLORATION_H

#include "DSE/StructuredOwnership.h"
#include "Frontend/Compilation/PreMappingCompilation.h"

#include "llvm/Support/Error.h"

#include <memory>
#include <variant>
#include <vector>

namespace llvm {
class Module;
}

namespace loom::dse {

struct PreMappingExplorationOptions final {
  raising::StructuredRaisingOptions raising;
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
exploreLlvmModuleToPreMapping(std::unique_ptr<llvm::Module> module,
                              const fabric::FinalizedFabricRoot &fabric,
                              const ResolvedConfig &config,
                              const PreMappingExplorationOptions &options,
                              const ArtifactStore &artifactStore);

} // namespace loom::dse

#endif // LOOM_DSE_PREMAPPINGEXPLORATION_H
