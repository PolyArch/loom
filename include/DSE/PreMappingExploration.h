#ifndef LOOM_DSE_PREMAPPINGEXPLORATION_H
#define LOOM_DSE_PREMAPPINGEXPLORATION_H

#include "DSE/Plan.h"
#include "DSE/PreMappingFrontier.h"
#include "DSE/StructuredOwnership.h"
#include "DSE/StructuredOwnershipInvocation.h"
#include "Common/ExecutionControl.h"
#include "Frontend/Analysis/StructuredProtocolDependencies.h"
#include "Frontend/Compilation/PreMappingCompilation.h"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

namespace loom::dse {

struct PreMappingExplorationOptions final {
  StructuredOwnershipExplorationOptions ownership;
  PreMappingFrontierPolicy frontier;
  ExecutionControlView executionControl;

  explicit PreMappingExplorationOptions(
      StructuredOwnershipExplorationOptions ownership,
      PreMappingFrontierPolicy frontier = {})
      : ownership(std::move(ownership)), frontier(std::move(frontier)) {}
};

struct SelectedPreMappingCompilation final {
  /// Invocation-local rank in the bounded software frontier. This value is
  /// derived from completed objective Evidence and does not enter any
  /// compilation or Dataflow Artifact identity.
  std::uint64_t preferenceRank = 0;
  std::optional<std::size_t> planningRecordOrdinal;
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
  HeuristicPruned,
  CoordinateBudget,
  ProgramMaterializationBudget,
  AnalyticEvaluationBudget,
  FunctionalReplayBudget,
  DataflowPromotionBudget,
  MappingPairBudget,
  ExactGateRejected,
  Unsupported,
  Unknown,
  CancelledOrTimeout,
};

llvm::StringRef toString(PreMappingCandidatePlanningDisposition value);

struct PreMappingSearchCompleteness final {
  bool domainComplete = false;
  bool budgetComplete = false;
  bool providerComplete = false;
  bool evidenceComplete = false;
  bool selectionComplete = false;

  bool exactComplete() const {
    return domainComplete && budgetComplete && providerComplete &&
           evidenceComplete && selectionComplete;
  }
};

enum class PreMappingCheckpointBoundary : std::uint8_t {
  CoordinatePlanning,
  OwnershipGeneration,
  EvidencePromotion,
  DataflowPromotion,
  MappingAdmission,
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
  std::optional<ArtifactRootReference> structuredProgram;
  std::optional<ArtifactRootReference> canonicalDataflow;
  std::vector<frontend::StructuredEntityRef> ownedProtocolRoots;
  std::vector<PreMappingSpectrumSeedKind> seedKinds;
  std::optional<PreMappingCandidateProjection> projection;
  std::optional<std::uint64_t> estimatedRuntimePicoseconds;
  PreMappingCandidatePlanningDisposition disposition =
      PreMappingCandidatePlanningDisposition::HeuristicPruned;
  std::optional<std::uint64_t> preferenceRank;
  std::optional<PreMappingMaterializedProjection> materializedProjection;
  std::optional<PreMappingTemporalWitness> temporalWitness;
  std::optional<DsePlanIncompleteReason> incompleteReason;
  /// A verified endpoint classification is written only after a real
  /// SystemMapping schedule has been imported and checked. It is not derived
  /// from seed kinds or the logical-domain fact.
  std::optional<PreMappingSpectrumClass> verifiedSpectrum;
  /// Digest of the immutable semantic coordinate and transformation lineage.
  /// Invocation policy, evaluation, disposition, and preference rank are
  /// excluded; downstream Mapping joins this value instead of array order.
  std::optional<ComponentViewDigest> candidateIdentity;
  /// Generation intent is provenance only. It is never an endpoint
  /// classification and is excluded from candidateIdentity.
  std::optional<PreMappingScheduleIntent> scheduleIntent;

  friend bool operator==(const PreMappingCandidatePlanningRecord &lhs,
                         const PreMappingCandidatePlanningRecord &rhs) {
    return lhs.structuredProgram == rhs.structuredProgram &&
           lhs.canonicalDataflow == rhs.canonicalDataflow &&
           lhs.ownedProtocolRoots == rhs.ownedProtocolRoots &&
           lhs.seedKinds == rhs.seedKinds &&
           lhs.temporalWitness == rhs.temporalWitness &&
           lhs.projection == rhs.projection &&
           lhs.estimatedRuntimePicoseconds == rhs.estimatedRuntimePicoseconds &&
           lhs.disposition == rhs.disposition &&
           lhs.preferenceRank == rhs.preferenceRank &&
           lhs.materializedProjection == rhs.materializedProjection &&
           lhs.incompleteReason == rhs.incompleteReason &&
           lhs.verifiedSpectrum == rhs.verifiedSpectrum &&
           lhs.candidateIdentity == rhs.candidateIdentity &&
           lhs.scheduleIntent == rhs.scheduleIntent;
  }
};

/// Derives the stable identity used to join one planning record across
/// pre-Mapping, Mapping, and application diagnostics. The supplied context
/// roots and policy digest remain invocation provenance at this API boundary;
/// only semantic candidate coordinate/transformation lineage enters the
/// digest. Mutable evaluation fields are not part of the identity.
llvm::Expected<ComponentViewDigest> computePreMappingCandidateIdentity(
    const PreMappingCandidatePlanningRecord &record,
    const ArtifactRootReference &sourceProgram,
    const ArtifactRootReference &fabric,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput,
    const ComponentViewDigest &frontierPolicyDigest);

/// Invocation-local best-so-far checkpoint. It is deliberately separate from
/// an Artifact identity: a cancelled or timed-out search may resume only from
/// the named parent/provenance after revalidation.
struct PreMappingCheckpoint final {
  PreMappingCheckpointBoundary boundary =
      PreMappingCheckpointBoundary::CoordinatePlanning;
  DsePlanIncompleteReason reason{
      CandidateGeneratorIncompleteReason::CancelledOrTimeout};
  ArtifactRootReference sourceProgram;
  ArtifactRootReference fabric;
  ArtifactRootReference workload;
  ArtifactRootReference runtimeInput;
  ComponentViewDigest frontierPolicyDigest;
  PreMappingWorkAccounting workAccounting;
  PreMappingSearchCompleteness completeness;
  std::uint64_t eligibleCoordinateCount = 0;
  bool coordinateFrontierTruncated = false;
  std::vector<ArtifactRootReference> retainedCandidates;
  /// Candidate records observed before the checkpoint boundary. This is a
  /// snapshot of planning provenance, not a claim that the records were
  /// materialized or admitted to Mapping.
  std::vector<PreMappingCandidatePlanningRecord> candidateInventory;
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
  frontend::analysis::StructuredProtocolDependencyProjection
      protocolDependencyProjection;
  std::vector<PreMappingCandidatePlanningRecord> candidateInventory;
  PreMappingFrontierPolicy frontierPolicy;
  std::uint64_t eligibleCoordinateCount = 0;
  bool coordinateFrontierTruncated = false;
  PreMappingWorkAccounting frontierAccounting;
  StructuredOwnershipSharedEvaluationStatistics sharedEvaluationStatistics;
  StructuredOwnershipEvaluationTiming evaluationTiming;
  evaluation::models::StructuredEvaluationInvocationCacheStatistics
      evaluationCacheStatistics;
  std::vector<DsePlanGenerateInvocationRecords> planGenerateInvocations;
  std::vector<RetainedDsePlanIncompleteness> retainedPlanIncompleteness;
  ArtifactRootReference sourceProgram;
  ArtifactRootReference fabric;
  ArtifactRootReference workload;
  ArtifactRootReference runtimeInput;
  ComponentViewDigest frontierPolicyDigest;
  StructuredOwnershipSelectionMode requestedPlannerMode =
      StructuredOwnershipSelectionMode::SemanticConformance;
  StructuredOwnershipSelectionMode resolvedPlannerMode =
      StructuredOwnershipSelectionMode::SemanticConformance;
  PreMappingSearchCompleteness completeness;
  std::optional<PreMappingShadowRecall> shadowRecall;

  bool searchComplete() const {
    return completeness.exactComplete();
  }
};

struct CompletedPreMappingNoFeasibleCandidate final {
  std::vector<ArtifactRootReference> satisfiedEvidence;
  std::vector<DsePlanGenerateInvocationRecords> planGenerateInvocations;
};

struct IncompletePreMappingExploration final {
  std::optional<std::uint64_t> planNodeOrdinal;
  DsePlanIncompleteReason reason{
      CandidateGeneratorIncompleteReason::CancelledOrTimeout};
  std::vector<ArtifactRootReference> retainedEvidence{};
  std::vector<DsePlanGenerateInvocationRecords> planGenerateInvocations{};
  std::uint64_t programMaterializations = 0;
  std::uint64_t analyticEvaluations = 0;
  std::uint64_t functionalReplays = 0;
  StructuredOwnershipEvaluationTiming evaluationTiming{};
  PreMappingSearchCompleteness completeness{};
  std::optional<PreMappingCheckpoint> checkpoint = std::nullopt;
  /// Present when incompleteness came from a resolved nested DSE plan. It
  /// lets the enclosing frontier retain the exact Plan-node provenance while
  /// continuing independent coordinates.
  std::optional<ComponentViewDigest> resolvedDseConfigViewDigest =
      std::nullopt;
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
