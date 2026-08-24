#ifndef LOOM_APPLICATION_BUILD_H
#define LOOM_APPLICATION_BUILD_H

#include "Config/ResolvedConfig.h"
#include "DSE/InvocationManifest.h"
#include "DSE/JointDesignExploration.h"
#include "DSE/JointMappingMigration.h"
#include "DSE/PreMappingExploration.h"
#include "DSE/ResourceTimeFrontier.h"
#include "DSE/ResourceTimeSpectrum.h"
#include "Deployment/Deployment.h"
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
};

struct ApplicationDeploymentRequest final {
  CompilerTargetPolicy compilerTargetPolicy;
  CompilerTargetLinkWorkspace linkerWorkspace;
};

struct ApplicationDeploymentArtifacts final {
  ArtifactRootReference configurationAbi;
  hardware::ConfigurationABIConstructionStatistics configurationAbiConstruction;
  std::vector<deployment::DeploymentHardwareBinding> hardwareBindings;
  std::vector<ArtifactRootReference> instructionCoreBinaries;
  deployment::FinalizedDeployment deployment;
};

struct ApplicationMappingExecutionRequest final {
  dse::DseProducerSemanticBuildIdentity producer;
  std::string journalRoot;
  std::vector<ArtifactRootReference> preexistingEvidence;
  std::optional<dse::JointBoundedQualityPolicy> boundedQuality;
  dse::SiteCapacity siteCapacity;
  dse::PlanExecutionPolicy executionPolicy;
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
  Exact,
  OwnerScopedPlanningClosure,
  /// The ApplicationBuild owner stopped before source/workload/runtime roots
  /// existed, so no InvocationManifest run-key could be derived. The public
  /// boundary still publishes a typed, auditable decision instead of a bare
  /// missing join.
  OwnerVerifiedPreAdmission,
  NotStartedBeforeMapping,
  Missing,
};

llvm::StringRef toString(ApplicationPairManifestJoinStatus value);

/// Fixed application-QoR dimensions. The ordering is stable for reports, but
/// the existing DSE ObjectiveProgram remains the sole ordering authority.
enum class ApplicationObjectiveDimension : std::uint8_t {
  HostOnlyWork,
  DfgCycles,
  CgraCycles,
  HostResidualWork,
  CutTransferWork,
  LaunchSynchronizationWork,
  ResourceCoreCost,
  MappingWork,
  Area,
  Power,
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
  /// A value in [0, 1000]. Exact and runtime-measured values use 1000;
  /// unsupported values use 0. This is evidence metadata, not an objective
  /// score and never participates in candidate identity.
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
  std::optional<dse::PreMappingSpectrumClass> verifiedSpectrum;
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
  std::optional<ComponentViewDigest> pairIdentity;
  /// Exact DSE InvocationManifest run-key join for the Mapping attempt.
  std::optional<std::array<std::uint8_t, 32>> invocationRunKey;
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
  std::vector<ApplicationObjectiveObservation> hostOnlyBaseline;
  std::vector<ApplicationPairCandidateRecord> candidates;
  std::vector<ApplicationObjectiveObservation> selectedObjective;
  std::optional<ComponentViewDigest> selectedCandidateIdentity;
  std::optional<ArtifactRootReference> selectedSystem;
  bool hostOnlyBaselineComplete = false;
  bool finalApplicationQorComplete = false;
  std::optional<std::string> detail;
};

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
};

/// Evidence for one application-level resource-time transition attempt. The
/// Mapping migration owner remains responsible for preservation and repair
/// counts; this record only joins that result to the application schedule
/// which caused the typed delta.
struct ApplicationIncrementalMappingObservation final {
  ArtifactRootReference parentMapping;
  ArtifactRootReference childSystem;
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
  std::uint64_t wallTimeNanoseconds = 0;
  bool verified = false;
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
/// become candidate measures. Missing or non-completed evidence remains a
/// typed bounded-quality outcome rather than an estimate.
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
