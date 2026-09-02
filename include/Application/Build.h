#ifndef LOOM_APPLICATION_BUILD_H
#define LOOM_APPLICATION_BUILD_H

#include "Application/Manifest.h"
#include "Application/RuntimeManifest.h"
#include "Config/ResolvedConfig.h"
#include "DSE/InvocationManifest.h"
#include "DSE/JointDesignExploration.h"
#include "DSE/JointHardwareReopen.h"
#include "DSE/JointMappingMigration.h"
#include "DSE/PreMappingExploration.h"
#include "DSE/ResourceTimeFrontier.h"
#include "DSE/ResourceTimeSpectrum.h"
#include "Deployment/Deployment.h"
#include "Evaluation/Case.h"
#include "Frontend/Compilation/PreMappingCompilation.h"
#include "Frontend/Executable/CompilerTargetBinding.h"
#include "Frontend/Executable/CompilerTargetLinker.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace llvm {
class Module;
}

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::application {

inline constexpr llvm::StringLiteral applicationBuildProducerIdentity{
    "loom.application.build.v2"};

struct ApplicationPointerMemoryObservable final {
  std::uint64_t argumentOrdinal = 0;
  sim::MemoryObservationForm form = sim::MemoryObservationForm::FullState;
};

/// One exact source invocation expressed in the existing Structured Program
/// workload and runtime-input domains. Symbol resolution is invocation-local;
/// all persistent inputs remain owned by SimulationWorkload and
/// SimulationRuntimeInput.
struct ApplicationSourceInvocation final {
  std::string entrySymbol;
  std::vector<sim::StructuredProgramArgumentSource> argumentPlan;
  bool observeReturnValue = false;
  std::vector<ApplicationPointerMemoryObservable> memoryObservables;
  std::vector<sim::StructuredRuntimeValueEntry> runtimeValues;
  std::vector<sim::RuntimeMemoryObject> memoryObjects;
  std::vector<sim::StructuredPointerBindingDraft> pointerBindings;
};

struct ApplicationBuildRequest final {
  ApplicationSourceInvocation sourceInvocation;
  std::vector<std::string> operatorProtocolSymbols;
  ArtifactRootReference system;
  std::vector<ArtifactRootReference> physicalTimingProfiles;
  ResolvedConfig resolvedConfig;
  dse::JointDesignPolicy jointPolicy;
  frontend::PreMappingCompilationOptions compilationOptions;
  dse::PreMappingExplorationOptions preMappingOptions;
  dse::ResourceTimeFrontierPolicy resourceTimePolicy;
  std::optional<SelectedApplicationInput> portfolioInput;
  std::optional<ArtifactRootReference> edaPredictionModelWeight;
  std::vector<evaluation::EvaluationCondition> fpaOperatingConditions;
};

struct PreparedApplicationSoftware final {
  std::uint64_t preferenceRank = 0;
  std::size_t preMappingCandidateRecordOrdinal = 0;
  ComponentViewDigest candidateIdentity;
  frontend::PublishedPreMappingCompilation compilation;
  std::vector<ArtifactRootReference> workloads;
  std::vector<sim::SourceBackedDfgReplayCaseReference> replayCases;
};

struct PreparedApplicationMappingAlternative final {
  std::uint64_t preferenceRank = 0;
  std::size_t preMappingCandidateRecordOrdinal = 0;
  ComponentViewDigest candidateIdentity;
  ComponentViewDigest resourceTimeScheduleHintDigest;
  /// Schedule provenance with identical static Mapping inputs shares the
  /// canonical plan. Each digest remains independently checked by Spectrum;
  /// this vector is derived application evidence, not candidate identity.
  std::vector<ComponentViewDigest> equivalentScheduleHintDigests;
  ArtifactRootReference dataflow;
  std::vector<dse::ResourceTimeRegionFeature> resourceTimeRegions;
  std::vector<dse::ResourceTimeRegionResourceBound> resourceTimeRegionBounds;
  dse::JointDesignExplorationPlan plan;
};

struct PreparedApplicationBuild final {
  ApplicationSourceInvocation sourceInvocation;
  dse::JointDesignPolicy jointPolicy;
  std::vector<PreparedApplicationSoftware> software;
  std::vector<ArtifactRootReference> satisfiedEvidence;
  std::vector<dse::DsePlanGenerateInvocationRecords>
      preMappingGenerateInvocations;
  frontend::analysis::StructuredProtocolDependencyProjection
      protocolDependencyProjection;
  std::vector<dse::PreMappingCandidatePlanningRecord> candidateInventory;
  dse::PreMappingFrontierPolicy preMappingFrontierPolicy;
  std::uint64_t preMappingEligibleCoordinateCount = 0;
  bool preMappingCoordinateFrontierTruncated = false;
  dse::PreMappingWorkAccounting preMappingWorkAccounting;
  dse::StructuredOwnershipEvaluationTiming preMappingEvaluationTiming;
  dse::StructuredOwnershipSharedEvaluationStatistics
      preMappingSharedEvaluationStatistics;
  evaluation::models::StructuredEvaluationInvocationCacheStatistics
      preMappingEvaluationCacheStatistics;
  std::vector<dse::RetainedDsePlanIncompleteness>
      retainedPreMappingIncompleteness;
  /// Best-first bounded alternatives. Each plan owns exactly one software
  /// candidate so completed infeasibility can fall through without executing
  /// unrelated lower-ranked Mapping work.
  std::vector<PreparedApplicationMappingAlternative> mappingAlternatives;
  dse::ResourceTimeFrontierPolicy resourceTimePolicy;
  dse::ResourceTimeMappingFunnel resourceTimeFunnel;
  dse::StructuredOwnershipSelectionMode preMappingRequestedPlannerMode =
      dse::StructuredOwnershipSelectionMode::SemanticConformance;
  dse::StructuredOwnershipSelectionMode preMappingResolvedPlannerMode =
      dse::StructuredOwnershipSelectionMode::SemanticConformance;
  dse::PreMappingSearchCompleteness preMappingCompleteness;
  std::optional<dse::PreMappingShadowRecall> preMappingShadowRecall;
  std::optional<std::uint64_t> preMappingSourceHostOnlyWork;
  ArtifactRootReference preMappingSourceProgram;
  ArtifactRootReference preMappingFabric;
  ArtifactRootReference preMappingWorkload;
  ArtifactRootReference preMappingRuntimeInput;
  ComponentViewDigest preMappingFrontierPolicyDigest;
  std::uint64_t preMappingFabricAccCoreCount = 0;
  /// The InvocationManifest-owned semantic closure for the pre-Mapping
  /// planning invocation.  It is present even when cancellation happens
  /// before a Mapping provider is dispatched, so the pair decision never
  /// loses its provenance join.
  std::optional<std::array<std::uint8_t, 32>> preMappingInvocationRunKey;
  std::optional<SelectedApplicationInput> portfolioInput;
  std::optional<ArtifactRootReference> edaPredictionModelWeight;
  std::vector<evaluation::EvaluationCondition> fpaOperatingConditions;
};

struct ApplicationDeploymentRequest final {
  CompilerTargetPolicy compilerTargetPolicy;
  CompilerTargetLinkWorkspace linkerWorkspace;
  ExecutionControlView executionControl;
};

/// Exact provider ledger projection for one Mapping execution. Invocation is
/// reconciled as either a provider dispatch or an execution-journal replay;
/// the three provider domains remain distinct.
struct ApplicationMappingProviderWorkObservation final {
  std::uint64_t techMappingInvocations = 0;
  std::uint64_t spatialPnrInvocations = 0;
  std::uint64_t systemPnrInvocations = 0;
  std::uint64_t techMappingDispatches = 0;
  std::uint64_t spatialPnrDispatches = 0;
  std::uint64_t systemPnrDispatches = 0;
  std::uint64_t techMappingJournalReplays = 0;
  std::uint64_t spatialPnrJournalReplays = 0;
  std::uint64_t systemPnrJournalReplays = 0;
};

/// Immutable Deployment-bound snapshot of the Mapping repair observation.
/// The compiler-side observation remains the semantic owner; this derived
/// record makes the exact repair cone and paired evidence replayable with the
/// preverified runtime transition.
struct ApplicationResourceTimeRepairEvidence final {
  std::vector<dataflow::RootThreadLaunchRef> reopenedRoots;
  dse::JointMappingReuseDisposition reuseDisposition =
      dse::JointMappingReuseDisposition::ColdFallback;
  std::uint64_t preservedTechMappings = 0;
  std::uint64_t preservedSpatialMappings = 0;
  std::uint64_t repairedTechMappings = 0;
  std::uint64_t repairedSpatialMappings = 0;
  std::uint64_t preservedSystemBindings = 0;
  std::uint64_t reopenedSystemBindings = 0;
  std::uint64_t coldWallTimeNanoseconds = 0;
  std::uint64_t incrementalWallTimeNanoseconds = 0;
  std::uint64_t coldVerifierRetainedBytes = 0;
  std::uint64_t incrementalVerifierRetainedBytes = 0;
  std::uint64_t coldVerifierWork = 0;
  std::uint64_t incrementalVerifierWork = 0;
  ApplicationMappingProviderWorkObservation coldProviderWork;
  ApplicationMappingProviderWorkObservation incrementalProviderWork;
  std::optional<std::uint64_t> coldDfgCycles;
  std::optional<std::uint64_t> coldCgraCycles;
  std::optional<std::uint64_t> incrementalDfgCycles;
  std::optional<std::uint64_t> incrementalCgraCycles;
};

struct ApplicationResourceTimeTransitionEvidence final {
  pnr::ResourceTimeTransition transition;
  dse::ResourceTimeSpectrumFunnelResult parentSpectrum;
  dse::ResourceTimeSpectrumFunnelResult childSpectrum;
  ApplicationResourceTimeRepairEvidence repair;
};

struct ApplicationDeploymentArtifacts final {
  ArtifactRootReference configurationAbi;
  hardware::ConfigurationABIConstructionStatistics configurationAbiConstruction;
  std::vector<deployment::DeploymentHardwareBinding> hardwareBindings;
  std::vector<ArtifactRootReference> instructionCoreBinaries;
  /// Every compiler-preverified adjacency together with the parent completion
  /// schedule and the child schedule which carries real active work.
  std::vector<ApplicationResourceTimeTransitionEvidence>
      resourceTimeTransitions;
  /// Independent replay of the selected schedule after endpoint Deployment
  /// construction. A missing closure remains typed incomplete here.
  std::optional<dse::ResourceTimeSpectrumFunnelResult> resourceTimeSpectrum;
  FinalizedApplicationRuntimeManifest runtimeManifest;
  deployment::FinalizedDeployment deployment;
};

struct ApplicationMappingExecutionRequest final {
  dse::DseProducerSemanticBuildIdentity producer;
  std::string journalRoot;
  std::vector<ArtifactRootReference> preexistingEvidence;
  std::optional<dse::JointBoundedQualityPolicy> boundedQuality;
  dse::SiteCapacity siteCapacity;
  dse::PlanExecutionPolicy executionPolicy;
  /// External user-selected Systems are fixed invocation targets. Builtin
  /// product targets may opt into the existing bounded hardware-feedback
  /// loop without changing Mapping legality or candidate identity.
  dse::JointHardwareExplorationScope hardwareExplorationScope =
      dse::JointHardwareExplorationScope::BoundedHardwareReopen;
  /// Cooperative application deadline shared with Spectrum verification.
  /// PlanExecutionPolicy remains the Mapping-dispatch owner; this view keeps
  /// verifier work inside the same invocation boundary.
  ExecutionControlView executionControl;
  /// Mapping-repair admission per exact runtime witness. Absent keeps the
  /// joint repair owner's default.
  std::optional<std::uint64_t> mappingRepairCandidateLimit;
};

enum class ApplicationMappingRuntimeDisposition : std::uint8_t {
  NotRequested,
  Completed,
  Unsupported,
  ProofNotEstablished,
  ExecutionFailed,
  CancelledOrTimeout,
};

/// The terminal decision for one complete application/workload and
/// Fabric/System pair. This is an application-level projection of the
/// existing planning, Mapping, and runtime records; it is not a second
/// Mapping legality or candidate identity authority.
enum class ApplicationPairDecisionDisposition : std::uint8_t {
  VerifiedAcceleration,
  VerifiedFeasibleButNotBeneficial,
  NoPromisingCandidate,
  ExactHardwareIncompatible,
  MappingProofNotEstablished,
  CancelledOrTimeout,
  BudgetExhausted,
  UnsupportedSemantic,
  ImplementationFailure,
  HardwareDseAlternative,
};

llvm::StringRef toString(ApplicationPairDecisionDisposition value);

enum class ApplicationPairManifestJoinStatus : std::uint8_t {
  OwnerScopedPlanningClosure,
  /// The ApplicationBuild owner stopped before source/workload/runtime roots
  /// existed, so no InvocationManifest run-key could be derived. The public
  /// boundary still publishes a typed, auditable decision instead of a bare
  /// missing join.
  OwnerVerifiedPreAdmission,
  Missing,
};

llvm::StringRef toString(ApplicationPairManifestJoinStatus value);

enum class ApplicationPortfolioExecutionBinding : std::uint8_t {
  NotSelected,
  DeclaredOnly,
  CanonicalSimulation,
  CanonicalSimulationAndOracle,
};

llvm::StringRef toString(ApplicationPortfolioExecutionBinding value);

/// Fixed application-QoR dimensions. The ordering is stable for reports, but
/// the existing DSE ObjectiveProgram remains the sole ordering authority.
/// Every value is a non-negative integer in the unit named here; the owning
/// Evidence root retains the exact decimal observation.
enum class ApplicationObjectiveDimension : std::uint8_t {
  /// Host-only baseline work: measured host cycles or the analytic picosecond
  /// estimate of the source program.
  HostOnlyWork,
  /// Measured DFG simulation cycles of the selected Mapping.
  DfgCycles,
  /// Measured CGRA simulation cycles of the selected Mapping.
  CgraCycles,
  /// Dynamic leaf executions left on the host by the selected ownership.
  HostResidualWork,
  /// Bytes crossing the host/accelerator cut per invocation.
  CutTransferWork,
  /// Launch and synchronization cost units of the selected ownership.
  LaunchSynchronizationWork,
  /// Exact System resource-core count of the selected Mapping.
  ResourceCoreCost,
  /// Mapping dispatch count of the selected invocation.
  MappingWork,
  /// Calibrated total area in square micrometers.
  Area,
  /// Calibrated dynamic plus leakage power in microwatts.
  Power,
  /// Calibrated energy of one CGRA execution in picojoules.
  Energy,
};

enum class ApplicationObjectiveEvidence : std::uint8_t {
  Exact,
  SoundBound,
  Analytic,
  Calibrated,
  RuntimeMeasured,
  Unsupported,
};

struct ApplicationObjectiveObservation final {
  ApplicationObjectiveDimension dimension =
      ApplicationObjectiveDimension::HostOnlyWork;
  std::optional<std::uint64_t> value;
  ApplicationObjectiveEvidence evidence =
      ApplicationObjectiveEvidence::Unsupported;
  /// A value in [0, 1000]. Exact and runtime-measured values use 1000,
  /// calibrated point estimates inside their training envelope 500, analytic
  /// estimates 250, and unsupported values 0. This is evidence metadata, not
  /// an objective score and never participates in candidate identity.
  std::uint16_t confidencePermille = 0;
  bool outOfDistribution = false;
};

/// Compact derived join for one candidate's existing planning and Mapping
/// observations. The detailed records and work ledgers remain owned by DSE;
/// this view only makes their stable references mechanically available from a
/// pair decision.
struct ApplicationPairMappingObservation final {
  std::uint64_t planOrdinal = 0;
  ComponentViewDigest scheduleHintDigest;
  ArtifactRootReference system;
  dse::JointDesignAttemptDisposition mappingDisposition =
      dse::JointDesignAttemptDisposition::Incomplete;
  ApplicationMappingRuntimeDisposition runtimeDisposition =
      ApplicationMappingRuntimeDisposition::NotRequested;
  std::optional<dse::DsePlanIncompleteReason> incompleteReason;
  std::vector<ArtifactRootReference> systemMappings;
  std::vector<ArtifactRootReference> runtimeEvidence;
  /// Completed SimulationComparison Evidence against the source-backed native
  /// oracle. Runtime Evidence remains the owner; this is its typed subset.
  std::vector<ArtifactRootReference> oracleEvidence;
  std::optional<std::uint64_t> dfgCycles;
  std::optional<std::uint64_t> cgraCycles;
  std::optional<std::uint64_t> resourceCoreCost;
  std::optional<dse::PreMappingSpectrumClass> verifiedSpectrum;
  std::optional<dse::ResourceTimeSpectrumIncompleteReason>
      resourceTimeSpectrumIncompleteReason;
  /// The analytic funnel's best schedule prediction for this candidate before
  /// exact Mapping, retained next to the exact outcome so prediction and
  /// backend result stay distinguishable.
  std::optional<std::uint64_t> predictedMakespanPicoseconds;
  dse::ResourceTimeEstimateSupport predictedSupport =
      dse::ResourceTimeEstimateSupport::Unsupported;
  dse::ResourceTimeEstimateSupport physicalModelSupport =
      dse::ResourceTimeEstimateSupport::Unsupported;
};

/// Exact comparison of the analytic funnel against the real Mapping/PnR
/// outcomes of every candidate that entered Mapping in this invocation. Counts
/// describe only the mapped sample; ranking recall compares the funnel's
/// lowest predicted makespan with the lowest measured CGRA cycle count.
struct ApplicationFunnelExactComparison final {
  std::uint64_t mappedCandidates = 0;
  std::uint64_t predictedFeasibleCandidates = 0;
  std::uint64_t verifiedCandidates = 0;
  std::uint64_t measuredCandidates = 0;
  std::uint64_t outOfDistributionCandidates = 0;
  std::optional<bool> bestRankingMatch;
};

/// Quality facts from one exact joint-design invocation. Application runtime
/// retries may execute distinct tail plans, so their facts retain their own
/// InvocationManifest key and local plan-ordinal base instead of being folded
/// into the final JointDesignExecutionSummary.
struct ApplicationPairQualityInvocationRecord final {
  std::uint64_t planOrdinalBase = 0;
  std::optional<std::array<std::uint8_t, 32>> invocationRunKey;
  dse::JointDesignQualityDisposition qualityDisposition =
      dse::JointDesignQualityDisposition::NotRequested;
  std::optional<ArtifactRootReference> qualityIncompleteCandidate;
  std::vector<std::string> qualityObjectiveDimensionLabels;
  std::vector<dse::JointDesignQualityObservation> qualityObservations;
  std::vector<std::string> hardwarePromotionObjectiveDimensionLabels;
  std::vector<dse::JointHardwarePromotionObservation>
      hardwarePromotionObservations;
  std::optional<std::uint64_t> selectedPlanOrdinal;
  std::optional<ArtifactRootReference> selectedMapping;
};

/// One identity-based reference into the existing candidate inventory and
/// Mapping outcome vectors. It intentionally stores only derived application
/// evidence; the planning record and JointDesignAttemptRecord remain the
/// owners of detailed gates, witnesses, checkpoints, and work ledgers.
struct ApplicationPairCandidateRecord final {
  std::optional<ComponentViewDigest> candidateIdentity;
  std::optional<ArtifactRootReference> structuredProgram;
  std::optional<ArtifactRootReference> canonicalDataflow;
  std::optional<ComponentViewDigest> planningProjectionIdentity;
  std::optional<ComponentViewDigest> materializedProjectionIdentity;
  std::optional<dse::PreMappingCandidatePlanningDisposition>
      planningDisposition;
  std::optional<dse::PreMappingScheduleIntent> scheduleIntent;
  std::optional<dse::DsePlanIncompleteReason> planningIncompleteReason;
  std::optional<dse::PreMappingSpectrumClass> verifiedSpectrum;
  std::size_t planningRecordOrdinal = 0;
  /// Selected Mapping plan for this candidate. Mapping observations own the
  /// complete per-plan inventory when one candidate is evaluated more than
  /// once.
  std::optional<std::uint64_t> planOrdinal;
  bool enteredMapping = false;
  bool selected = false;
  std::vector<ApplicationObjectiveObservation> objective;
  std::vector<ApplicationPairMappingObservation> mappingObservations;
};

/// Stable pair-level decision record. The pair identity is derived from the
/// exact source/workload/runtime/Fabric roots and is independent of frontier
/// policy, ranking, cache state, and disposition. Detailed candidate facts
/// remain in ApplicationMappingCandidateOutcome, JointDesignAttemptRecord,
/// PreMappingCandidatePlanningRecord, checkpoints, and their ledgers.
struct ApplicationPairDecisionRecord final {
  std::optional<SelectedApplicationInput> portfolioInput;
  ApplicationFunnelExactComparison funnelExactComparison;
  ApplicationPortfolioExecutionBinding portfolioExecutionBinding =
      ApplicationPortfolioExecutionBinding::NotSelected;
  std::optional<ComponentViewDigest> pairIdentity;
  /// Exact DSE InvocationManifest run-key join for the Mapping attempt.
  std::optional<std::array<std::uint8_t, 32>> invocationRunKey;
  /// A pre-admission decision may have no semantic roots yet. In that narrow
  /// case the application-build owner identifies the explicit contract that
  /// authorizes a keyless record; this is not a substitute run key.
  std::optional<std::string> manifestJoinOwner;
  std::optional<std::string> manifestJoinContract;
  bool manifestJoinOwnerVerified = false;
  ApplicationPairManifestJoinStatus manifestJoinStatus =
      ApplicationPairManifestJoinStatus::Missing;
  ApplicationPairDecisionDisposition disposition =
      ApplicationPairDecisionDisposition::ImplementationFailure;
  std::optional<ArtifactRootReference> sourceProgram;
  std::optional<ArtifactRootReference> fabric;
  std::optional<ArtifactRootReference> workload;
  std::optional<ArtifactRootReference> runtimeInput;
  /// Planning records include budget/pruning bookkeeping that never became a
  /// semantic candidate. They remain owned by PreMapping; the pair candidate
  /// list contains only records with a complete stable candidate identity.
  std::uint64_t planningRecordCount = 0;
  std::uint64_t nonCandidatePlanningRecordCount = 0;
  /// Exact final-invocation quality state. ObjectiveProgram remains the
  /// ordering owner; codes retain their SystemMapping and Evidence joins.
  std::vector<std::string> qualityObjectiveDimensionLabels;
  dse::JointDesignQualityDisposition qualityDisposition =
      dse::JointDesignQualityDisposition::NotRequested;
  std::optional<ArtifactRootReference> qualityIncompleteCandidate;
  std::vector<dse::JointDesignQualityObservation> qualityObservations;
  /// Exact labels and observations from pre-Mapping hardware promotion. The
  /// promoted bit identifies the bounded finalist set that entered ordinary
  /// Mapping/PnR work; it is not a physical feasibility claim.
  std::vector<std::string> hardwarePromotionObjectiveDimensionLabels;
  std::vector<dse::JointHardwarePromotionObservation>
      hardwarePromotionObservations;
  std::vector<ApplicationPairQualityInvocationRecord> qualityInvocations;
  std::vector<ApplicationObjectiveObservation> hostOnlyBaseline;
  std::vector<ApplicationPairCandidateRecord> candidates;
  /// Causal decisions retain every dimension as an explicit null residual.
  std::vector<ApplicationObjectiveObservation> selectedObjective;
  std::optional<ComponentViewDigest> selectedCandidateIdentity;
  std::optional<ArtifactRootReference> selectedSystem;
  std::optional<ArtifactRootReference> selectedSystemMapping;
  bool hostOnlyBaselineComplete = false;
  bool finalApplicationQorComplete = false;
  std::optional<std::string> detail;
};

/// Publishes the typed pair boundary when an exact portfolio profile cannot
/// enter the product source-binding runner. No source/workload/runtime roots
/// are invented before compilation, and the host-profile report remains a
/// separate operational input.
ApplicationPairDecisionRecord makeUnsupportedPortfolioProfilePairDecision(
    SelectedApplicationInput selection,
    const ArtifactRootReference &requestedSystem, llvm::StringRef detail);

/// Exact join between one bounded pre-Mapping planning record and one joint
/// Mapping attempt. Verified outcomes name the independently imported
/// SystemMapping roots; incomplete and infeasible outcomes remain distinct.
struct ApplicationMappingCandidateOutcome final {
  std::size_t preMappingCandidateRecordOrdinal = 0;
  std::uint64_t planOrdinal = 0;
  ComponentViewDigest resourceTimeScheduleHintDigest;
  ArtifactRootReference dataflow;
  ArtifactRootReference system;
  dse::JointDesignAttemptDisposition disposition =
      dse::JointDesignAttemptDisposition::Incomplete;
  std::optional<std::uint64_t> incompleteNodeOrdinal;
  std::optional<dse::DsePlanIncompleteReason> incompleteReason;
  std::vector<ArtifactRootReference> systemMappings;
  /// The exact planning record used to admit this Mapping attempt. Keeping
  /// this invocation evidence beside the outcome prevents reports from
  /// reconstructing candidate provenance by array position.
  std::optional<dse::PreMappingCandidatePlanningRecord> planningRecord;
  std::vector<pnr::SystemBindingPartitionIntent> systemBindingPartitions;
  ApplicationMappingRuntimeDisposition runtimeDisposition =
      ApplicationMappingRuntimeDisposition::NotRequested;
  std::vector<ArtifactRootReference> runtimeEvidence;
  std::vector<std::uint64_t> qualityObjectiveCodes;
  std::optional<dse::ResourceTimeSpectrumFunnelResult> resourceTimeSpectrum;
  /// Raw application runtime observations, when the existing replay provider
  /// supplied them. Objective codes remain the DSE ordering projection and are
  /// never treated as physical values when these observations are available.
  std::optional<std::uint64_t> dfgCycles;
  std::optional<std::uint64_t> cgraCycles;
  std::optional<std::uint64_t> resourceCoreCost;
  std::vector<ArtifactRootReference> oracleEvidence;
};

/// Evidence for one application-level resource-time transition attempt. The
/// Mapping migration owner remains responsible for preservation and repair
/// counts; this record only joins that result to the application schedule
/// which caused the typed delta.
struct ApplicationIncrementalMappingObservation final {
  ArtifactRootReference parentMapping;
  ArtifactRootReference childSystem;
  std::optional<ArtifactRootReference> childMapping;
  std::optional<ArtifactRootReference> coldMapping;
  std::uint64_t parentPlanOrdinal = 0;
  std::uint64_t childPlanOrdinal = 0;
  ComponentViewDigest parentScheduleHintDigest;
  ComponentViewDigest childScheduleHintDigest;
  std::vector<dataflow::RootThreadLaunchRef> reopenedRoots;
  dse::JointMappingReuseDisposition reuseDisposition =
      dse::JointMappingReuseDisposition::ColdFallback;
  std::uint64_t preservedTechMappings = 0;
  std::uint64_t preservedSpatialMappings = 0;
  std::uint64_t repairedTechMappings = 0;
  std::uint64_t repairedSpatialMappings = 0;
  std::uint64_t preservedSystemBindings = 0;
  std::uint64_t reopenedSystemBindings = 0;
  dse::JointDesignAttemptDisposition disposition =
      dse::JointDesignAttemptDisposition::Incomplete;
  std::optional<dse::DsePlanIncompleteReason> incompleteReason;
  std::uint64_t coldWallTimeNanoseconds = 0;
  std::uint64_t incrementalWallTimeNanoseconds = 0;
  std::uint64_t wallTimeNanoseconds = 0;
  std::uint64_t coldVerifierRetainedBytes = 0;
  std::uint64_t incrementalVerifierRetainedBytes = 0;
  std::uint64_t coldVerifierWork = 0;
  std::uint64_t incrementalVerifierWork = 0;
  ApplicationMappingProviderWorkObservation coldProviderWork;
  ApplicationMappingProviderWorkObservation incrementalProviderWork;
  std::optional<std::uint64_t> coldDfgCycles;
  std::optional<std::uint64_t> coldCgraCycles;
  std::optional<std::uint64_t> incrementalDfgCycles;
  std::optional<std::uint64_t> incrementalCgraCycles;
  bool verified = false;
};

/// Ordered join from one selected schedule to its bounded adjacent Mapping
/// repairs. Observation ordinals are the canonical edge lineage; a Deployment
/// may publish only the longest prefix which closes as one independently
/// verified spectrum scenario.
struct ApplicationResourceTimeMappingPath final {
  std::uint64_t scheduleOwnerPlanOrdinal = 0;
  ComponentViewDigest scheduleHintDigest;
  std::vector<std::uint64_t> observationOrdinals;
};

struct ApplicationMappingProvenance final {
  std::optional<ArtifactRootReference> sourceProgram;
  std::optional<ArtifactRootReference> fabric;
  std::optional<ArtifactRootReference> workload;
  std::optional<ArtifactRootReference> runtimeInput;
  std::optional<ComponentViewDigest> frontierPolicyDigest;
  std::optional<dse::ResourceTimeMappingFunnelAccounting>
      resourceTimeFunnelAccounting;
  bool resourceTimeFunnelTruncated = false;
  std::optional<dse::ResourceTimeFrontierIncompleteReason>
      resourceTimeFunnelIncompleteReason;
  dse::PreMappingSearchCompleteness preMappingCompleteness;
  dse::StructuredOwnershipSelectionMode requestedPlannerMode =
      dse::StructuredOwnershipSelectionMode::SemanticConformance;
  dse::StructuredOwnershipSelectionMode resolvedPlannerMode =
      dse::StructuredOwnershipSelectionMode::SemanticConformance;
  std::vector<ApplicationIncrementalMappingObservation>
      incrementalMappingObservations;
  std::optional<ApplicationResourceTimeMappingPath> resourceTimeMappingPath;
  /// Derived application decision view. All detailed evidence remains owned by
  /// the records referenced above; this field only closes the pair-level join.
  std::optional<ApplicationPairDecisionRecord> pairDecision;
};

struct ApplicationMappingExecution final {
  dse::JointDesignExecution execution;
  std::vector<ApplicationMappingCandidateOutcome> candidateOutcomes;
  ApplicationMappingProvenance provenance;
};

enum class ApplicationBuildUnsupportedKind : std::uint8_t {
  RootCoordinates,
  DirectInvocationBoundary,
  DynamicInvocationBoundary,
};

struct UnsupportedApplicationBuild final {
  ApplicationBuildUnsupportedKind kind;
  ArtifactRootReference canonicalDataflow;
  dataflow::RootThreadLaunchRef root;
};

struct IncompleteApplicationResourceTimePlanning final {
  dse::ResourceTimeFrontierIncompleteReason reason =
      dse::ResourceTimeFrontierIncompleteReason::CancelledOrTimeout;
  dse::ResourceTimeMappingFunnel funnel;
  std::vector<dse::PreMappingCandidatePlanningRecord> candidateInventory;
  ArtifactRootReference sourceProgram;
  ArtifactRootReference fabric;
  ArtifactRootReference workload;
  ArtifactRootReference runtimeInput;
  ComponentViewDigest frontierPolicyDigest;
  std::optional<std::uint64_t> sourceHostOnlyWork;
};

using ApplicationBuildPreparationOutcome = std::variant<
    PreparedApplicationBuild, dse::CompletedPreMappingNoFeasibleCandidate,
    dse::IncompletePreMappingExploration, UnsupportedApplicationBuild,
    IncompleteApplicationResourceTimePlanning>;

/// Composes final-link, compiler, Simulation input, and joint-DSE owners
/// without creating another persistent application or candidate identity.
llvm::Expected<ApplicationBuildPreparationOutcome>
prepareApplicationBuild(const llvm::Module &finalLinkedModule,
                        ApplicationBuildRequest request,
                        const ArtifactStore &artifacts, const BlobStore &blobs);

/// Executes or resumes a prepared Mapping plan through the shared bounded
/// journal, scheduler, exact repair, and independent Mapping verifiers.
llvm::Expected<ApplicationMappingExecution>
executeApplicationMapping(const PreparedApplicationBuild &prepared,
                          ApplicationMappingExecutionRequest request,
                          const ArtifactStore &artifacts,
                          const BlobStore &blobs);

/// Builds the application-owned bounded-quality adapter. Each selected
/// SystemMapping is replayed through the existing DFG/CGRA evidence models;
/// only completed cycle observations and the imported Fabric AccCore count
/// become candidate measures. When the prepared build carries an admitted
/// frozen FPA weight, the same objective also evaluates exact Decimal FPA
/// predictions and uses them to promote bounded hardware parents. Missing,
/// OOD, or non-completed evidence remains a typed bounded-quality outcome.
llvm::Expected<dse::JointBoundedQualityPolicy>
makeApplicationBoundedQualityPolicy(
    const PreparedApplicationBuild &prepared,
    const dse::PlanExecutionPolicy &executionPolicy,
    const ArtifactStore &artifacts, const BlobStore &blobs);

/// Requires one uniquely selected, independently imported SystemMapping and
/// derives the complete declarative Deployment closure. The host executable
/// and InstructionCore executables are generated from the exact final-linked
/// module and target bindings; no RTL generation or compilation occurs here.
llvm::Expected<ApplicationDeploymentArtifacts> buildApplicationDeployment(
    const PreparedApplicationBuild &prepared,
    const ApplicationMappingExecution &mappingExecution,
    const llvm::Module &finalLinkedModule, ApplicationDeploymentRequest request,
    const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::application

#endif // LOOM_APPLICATION_BUILD_H
