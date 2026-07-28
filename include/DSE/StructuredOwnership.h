#ifndef LOOM_DSE_STRUCTUREDOWNERSHIP_H
#define LOOM_DSE_STRUCTUREDOWNERSHIP_H

#include "DSE/Promotion.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"

#include "llvm/Support/Error.h"

#include <variant>
#include <vector>

namespace loom {
class ArtifactStore;
struct ResolvedConfig;
} // namespace loom

namespace loom::dse {

struct StructuredOwnershipExplorationOptions final {
  lowering::CanonicalDataflowLoweringOptions lowering;
  PointMetricTopKSelection selection;
};

struct CompletedStructuredOwnershipSelection final {
  std::vector<frontend::MaterializedOwnershipCandidate> selected;
  std::vector<ArtifactRootReference> satisfiedEvidence;
};

using StructuredOwnershipExplorationOutcome =
    std::variant<CompletedStructuredOwnershipSelection,
                 CompletedNoFeasibleCandidate, IncompleteSelection>;

/// Executes one finite Ownership Generate/Evaluate/Promote composition. Every
/// generated child is independently materialized from the immutable parent;
/// expected semantic or exact-Fabric rejection prunes only that child.
llvm::Expected<StructuredOwnershipExplorationOutcome>
generateAndPromoteStructuredOwnership(
    const frontend::StructuredProgramCandidate &parent,
    const fabric::FinalizedFabricRoot &fabric, const ResolvedConfig &config,
    const StructuredOwnershipExplorationOptions &options,
    const ArtifactStore &artifactStore);

} // namespace loom::dse

#endif // LOOM_DSE_STRUCTUREDOWNERSHIP_H
