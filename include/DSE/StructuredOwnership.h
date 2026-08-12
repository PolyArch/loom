#ifndef LOOM_DSE_STRUCTUREDOWNERSHIP_H
#define LOOM_DSE_STRUCTUREDOWNERSHIP_H

#include "DSE/Promotion.h"
#include "Dataflow/Transforms/DataflowRewrite.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Frontend/Compilation/StructuredExecutionShape.h"
#include "Frontend/Compilation/StructuredMemoryCommunication.h"
#include "Frontend/Compilation/StructuredSchedule.h"
#include "Frontend/Compilation/StructuredSpecialMathAccuracy.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SourceBackedDfgValidation.h"

#include "llvm/ADT/ArrayRef.h"
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

enum class StructuredOwnershipSelectionMode : std::uint8_t {
  BenefitQualified,
  SemanticConformance,
};

struct StructuredOwnershipTopKSelection final {
  evaluation::MetricRequestOrdinal metricRequest;
  ResolvedObjectiveDirection direction;
  std::uint64_t k;
};

struct StructuredOwnershipExplorationOptions final {
  lowering::CanonicalDataflowLoweringOptions lowering;
  StructuredOwnershipTopKSelection selection;
  std::uint32_t candidateWorkerCount = 1;
  sim::SourceBackedDfgValidationLimits functionalReplayLimits{
      100000, 1000000, 256ULL * 1024ULL * 1024ULL};
  /// Optional invocation-local operator protocol roots. When nonempty, the
  /// ownership domain contains only these exact defined callables and their
  /// statically resolved direct callees. The Structured Program remains the
  /// sole program authority.
  std::vector<frontend::StructuredEntityRef> protocolCallableRoots{};
  StructuredOwnershipSelectionMode selectionMode =
      StructuredOwnershipSelectionMode::BenefitQualified;
};

struct StructuredOwnershipGenerationOptions final {
  lowering::CanonicalDataflowLoweringOptions lowering;
  std::uint64_t scopeExpansionLimit = 64;
  std::uint32_t candidateWorkerCount = 1;
  std::vector<frontend::StructuredEntityRef> protocolCallableRoots{};
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

struct StructuredScheduleDerivation final {
  ArtifactRootReference parent;
  frontend::StructuredScheduleDecision decision;

  friend bool operator==(const StructuredScheduleDerivation &lhs,
                         const StructuredScheduleDerivation &rhs) {
    return lhs.parent == rhs.parent && lhs.decision == rhs.decision;
  }
};

struct StructuredExecutionShapeDerivation final {
  ArtifactRootReference parent;
  frontend::StructuredExecutionShapeDecision decision;

  friend bool operator==(const StructuredExecutionShapeDerivation &lhs,
                         const StructuredExecutionShapeDerivation &rhs) {
    return lhs.parent == rhs.parent && lhs.decision == rhs.decision;
  }
};

struct StructuredSpecialMathAccuracyDerivation final {
  ArtifactRootReference parent;
  frontend::StructuredSpecialMathAccuracyDecision decision;

  friend bool operator==(const StructuredSpecialMathAccuracyDerivation &lhs,
                         const StructuredSpecialMathAccuracyDerivation &rhs) {
    return lhs.parent == rhs.parent && lhs.decision == rhs.decision;
  }
};

struct StructuredMemoryCommunicationDerivation final {
  ArtifactRootReference parent;
  frontend::StructuredMemoryCommunicationDecision decision;

  friend bool operator==(const StructuredMemoryCommunicationDerivation &lhs,
                         const StructuredMemoryCommunicationDerivation &rhs) {
    return lhs.parent == rhs.parent && lhs.decision == rhs.decision;
  }
};

struct DataflowRewriteDerivation final {
  ArtifactRootReference parent;
  ArtifactRootReference child;
  dataflow::DataflowRewriteDecision decision;

  friend bool operator==(const DataflowRewriteDerivation &lhs,
                         const DataflowRewriteDerivation &rhs) {
    return lhs.parent == rhs.parent && lhs.child == rhs.child &&
           lhs.decision == rhs.decision;
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

/// Complete finite Generate result. Candidate identity remains the ordinary
/// Structured Program Artifact identity; dispositions are invocation-local
/// lineage used by tracing and never become a second candidate authority.
struct CompletedStructuredOwnershipGeneration final {
  CandidateSet candidates;
  std::vector<StructuredOwnershipCandidateDisposition> dispositions;
  std::uint64_t plannedScopeCount = 0;
  std::uint64_t decisionAttemptCount = 0;
};

struct SelectedStructuredOwnershipCandidate final {
  frontend::MaterializedOwnershipCandidate candidate;
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

/// Generates and publishes one finite canonical ownership candidate set. It
/// performs no metric acquisition, quality gate, or candidate promotion.
llvm::Expected<CompletedStructuredOwnershipGeneration>
generateStructuredOwnershipCandidates(
    const frontend::StructuredProgramCandidate &parent,
    const frontend::StructuredProgramCandidate &sourceProgram,
    const sim::CanonicalSimulationWorkload &workload,
    const sim::CanonicalSimulationRuntimeInput &runtimeInput,
    const fabric::FinalizedFabricRoot &fabric,
    const StructuredOwnershipGenerationOptions &options,
    const ArtifactStore &artifactStore,
    llvm::ArrayRef<frontend::StructuredOperationSourceProvenance>
        sourceProvenance = {});

} // namespace loom::dse

#endif // LOOM_DSE_STRUCTUREDOWNERSHIP_H
