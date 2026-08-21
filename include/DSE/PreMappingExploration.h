#ifndef LOOM_DSE_PREMAPPINGEXPLORATION_H
#define LOOM_DSE_PREMAPPINGEXPLORATION_H

#include "DSE/Plan.h"
#include "DSE/StructuredOwnership.h"
#include "Frontend/Analysis/StructuredProtocolDependencies.h"
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

enum class PreMappingCandidatePlanningDisposition : std::uint8_t {
  Retained,
  BoundedFrontierBudget,
};

struct PreMappingProtocolRootActivity final {
  frontend::StructuredEntityRef root;
  std::uint64_t dynamicActivations = 0;
  std::uint64_t dynamicLeafExecutions = 0;

  friend bool operator==(const PreMappingProtocolRootActivity &lhs,
                         const PreMappingProtocolRootActivity &rhs) {
    return lhs.root == rhs.root &&
           lhs.dynamicActivations == rhs.dynamicActivations &&
           lhs.dynamicLeafExecutions == rhs.dynamicLeafExecutions;
  }
};

/// One invocation-local candidate planning record. Exact dependencies and
/// source activity are stored once on CompletedPreMappingSelection and remain
/// mechanically joinable through `ownedProtocolRoots`; this record does not
/// duplicate cut projections or claim Mapping legality.
struct PreMappingCandidatePlanningRecord final {
  ArtifactRootReference structuredProgram;
  std::vector<frontend::StructuredEntityRef> ownedProtocolRoots;
  std::optional<std::uint64_t> estimatedRuntimePicoseconds;
  PreMappingCandidatePlanningDisposition disposition =
      PreMappingCandidatePlanningDisposition::BoundedFrontierBudget;
  std::optional<std::uint64_t> preferenceRank;

  friend bool operator==(const PreMappingCandidatePlanningRecord &lhs,
                         const PreMappingCandidatePlanningRecord &rhs) {
    return lhs.structuredProgram == rhs.structuredProgram &&
           lhs.ownedProtocolRoots == rhs.ownedProtocolRoots &&
           lhs.estimatedRuntimePicoseconds == rhs.estimatedRuntimePicoseconds &&
           lhs.disposition == rhs.disposition &&
           lhs.preferenceRank == rhs.preferenceRank;
  }
};

struct CompletedPreMappingSelection final {
  /// A completed promotion can select a verified incumbent without exhausting
  /// every Generate or Evidence domain. Retained plan records remain the sole
  /// authority for search completeness.
  std::vector<SelectedPreMappingCompilation> selected;
  std::vector<ArtifactRootReference> satisfiedEvidence;
  std::vector<StructuredOwnershipCandidateDisposition> dispositions;
  std::vector<PreMappingProtocolRootActivity> protocolRootActivity;
  std::vector<frontend::analysis::StructuredProtocolDependency>
      protocolDependencies;
  std::vector<PreMappingCandidatePlanningRecord> candidateInventory;
  std::vector<DsePlanGenerateInvocationRecords> planGenerateInvocations;
  std::vector<RetainedDsePlanIncompleteness> retainedPlanIncompleteness;

  bool searchComplete() const { return retainedPlanIncompleteness.empty(); }
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
