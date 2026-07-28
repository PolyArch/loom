#ifndef LOOM_DSE_STRUCTUREDOWNERSHIP_H
#define LOOM_DSE_STRUCTUREDOWNERSHIP_H

#include "DSE/Promotion.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"

#include "llvm/Support/Error.h"

#include <cstdint>
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
  std::uint32_t candidateWorkerCount = 1;
};

/// One exact parent-local ownership decision that produced a child candidate.
/// This is invocation lineage, not part of either Artifact's identity.
struct StructuredOwnershipDerivation final {
  frontend::SpatialOwnershipScope scope;
  frontend::SpatialOwnershipDecisionPoint decision;

  friend bool operator==(const StructuredOwnershipDerivation &lhs,
                         const StructuredOwnershipDerivation &rhs) {
    return lhs.scope == rhs.scope && lhs.decision == rhs.decision;
  }
};

struct SelectedStructuredOwnershipCandidate final {
  frontend::MaterializedOwnershipCandidate candidate;
  std::vector<StructuredOwnershipDerivation> derivations;
};

struct CompletedStructuredOwnershipSelection final {
  std::vector<SelectedStructuredOwnershipCandidate> selected;
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
