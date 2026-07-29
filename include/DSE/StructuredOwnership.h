#ifndef LOOM_DSE_STRUCTUREDOWNERSHIP_H
#define LOOM_DSE_STRUCTUREDOWNERSHIP_H

#include "DSE/Promotion.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SourceBackedDfgValidation.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
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
  sim::SourceBackedDfgValidationLimits functionalReplayLimits{
      100000, 1000000, 256ULL * 1024ULL * 1024ULL};
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

/// One exact coordinate in the invocation-local finite ownership domain. An
/// absent decision denotes a definition-level scope rejection before a typed
/// decision domain could be derived.
struct StructuredOwnershipCandidateCoordinate final {
  frontend::SpatialOwnershipScope scope;
  std::optional<frontend::SpatialOwnershipDecisionPoint> decision;

  friend bool operator==(const StructuredOwnershipCandidateCoordinate &lhs,
                         const StructuredOwnershipCandidateCoordinate &rhs) {
    return lhs.scope == rhs.scope && lhs.decision == rhs.decision;
  }
};

struct StructuredOwnershipCandidateRejectionRecord final {
  frontend::SpatialOwnershipCandidateRejectionKind kind;
  std::string message;

  friend bool
  operator==(const StructuredOwnershipCandidateRejectionRecord &lhs,
             const StructuredOwnershipCandidateRejectionRecord &rhs) {
    return lhs.kind == rhs.kind && lhs.message == rhs.message;
  }
};

using StructuredOwnershipCandidateResult =
    std::variant<ArtifactRootReference,
                 StructuredOwnershipCandidateRejectionRecord>;

/// Complete invocation-local accounting for one scope or decision attempt.
/// This is a projection for InvocationManifest provenance, not an Artifact or
/// a second candidate identity.
struct StructuredOwnershipCandidateDisposition final {
  StructuredOwnershipCandidateCoordinate coordinate;
  StructuredOwnershipCandidateResult result;

  friend bool operator==(const StructuredOwnershipCandidateDisposition &lhs,
                         const StructuredOwnershipCandidateDisposition &rhs) {
    return lhs.coordinate == rhs.coordinate && lhs.result == rhs.result;
  }
};

struct SelectedStructuredOwnershipCandidate final {
  frontend::MaterializedOwnershipCandidate candidate;
  std::vector<StructuredOwnershipDerivation> derivations;
};

struct CompletedStructuredOwnershipSelection final {
  std::vector<SelectedStructuredOwnershipCandidate> selected;
  std::vector<ArtifactRootReference> satisfiedEvidence;
  std::vector<StructuredOwnershipCandidateDisposition> dispositions;
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
    const sim::CanonicalSimulationWorkload &workload,
    const sim::CanonicalSimulationRuntimeInput &runtimeInput,
    const fabric::FinalizedFabricRoot &fabric, const ResolvedConfig &config,
    const StructuredOwnershipExplorationOptions &options,
    const ArtifactStore &artifactStore);

} // namespace loom::dse

#endif // LOOM_DSE_STRUCTUREDOWNERSHIP_H
