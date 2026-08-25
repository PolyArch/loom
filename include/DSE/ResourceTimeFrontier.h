#ifndef LOOM_DSE_RESOURCETIMEFRONTIER_H
#define LOOM_DSE_RESOURCETIMEFRONTIER_H

#include "Common/Artifact.h"
#include "Common/ComponentViewDigest.h"
#include "Common/ExecutionControl.h"
#include "DSE/PreMappingFrontier.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "PnR/System/SystemMappingMigration.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace loom::dse {

/// Exact invocation inputs that bound an invocation-local resource-time
/// frontier and its removable memoization. None of these observations become
/// schedule-candidate identity.
struct ResourceTimeInvocationKey final {
  ArtifactRootReference sourceLineage;
  ArtifactRootReference dataflow;
  ArtifactRootReference fabric;
  ArtifactRootReference workload;
  ArtifactRootReference runtimeInput;
  ComponentViewDigest resolvedConfigDigest;
  ComponentViewDigest modelSnapshotDigest;
  /// These observations affect the reachable region set and the analytic
  /// speedup curve even when the underlying Dataflow artifact is unchanged.
  std::string entrySymbol;
  std::optional<std::uint64_t> estimatedRuntimePicoseconds;
};

/// Exact semantic context for removable transition-result memoization. The
/// transition itself supplies the parent/child Mapping and Deployment, safe
/// point, trigger, active/live state, and resource/configuration/route deltas.
/// Result status and measured reprogramming and migration times are
/// deliberately excluded from the key.
struct ResourceTimeTransitionCacheKeyInput final {
  ArtifactRootReference constraints;
  ComponentViewDigest algorithmIdentity;
  ArtifactRootReference childTarget;
  ComponentViewDigest scheduleDeltaDigest;
  ComponentViewDigest hardwareDeltaDigest;
};

/// Derives an invocation-local removable cache key. This never admits a
/// Mapping, transition, or endpoint and a cache hit still requires the owner
/// verifier before publication.
llvm::Expected<ComponentViewDigest> deriveResourceTimeTransitionCacheKey(
    const pnr::ResourceTimeTransition &transition,
    const ResourceTimeTransitionCacheKeyInput &input);

enum class ResourceTimeEstimateSupport : std::uint8_t {
  Exact,
  Analytic,
  Calibrated,
  OutOfDomain,
  Unsupported,
};

enum class ResourceTimeEstimateConfidence : std::uint8_t {
  None,
  Low,
  Calibrated,
  OutOfDomain,
};

llvm::StringRef
resourceTimeEstimateSupportSpelling(ResourceTimeEstimateSupport support);
llvm::StringRef resourceTimeEstimateConfidenceSpelling(
    ResourceTimeEstimateConfidence confidence);

struct ResourceTimeDependencyFeature final {
  ::dataflow::RootThreadLaunchRef producer;
  pnr::ResourceTimeReadinessKind readiness =
      pnr::ResourceTimeReadinessKind::Completion;
};

/// One provider-owned point on a region's frozen resource-speedup curve.
/// Resource vectors use the invocation's canonical resource-class order.
/// These estimates rank schedules; they never prove Mapping legality.
struct ResourceTimeSpeedupPoint final {
  std::vector<std::uint64_t> resourceUnits;
  std::uint64_t executionTimePicoseconds = 0;
  std::optional<std::uint64_t> firstTokenLatencyPicoseconds;
  std::optional<std::uint64_t> initiationIntervalPicoseconds;
  std::uint64_t hostTransferTimePicoseconds = 0;
  std::uint64_t configurationTimePicoseconds = 0;
  std::uint64_t liveStateMigrationTimePicoseconds = 0;
  ResourceTimeEstimateSupport support =
      ResourceTimeEstimateSupport::Unsupported;
};

/// Provider-owned structural features used only for analytic ranking. Values
/// are derived from the immutable Dataflow projection; they never assert
/// physical Mapping legality or a complete QoR metric.
struct ResourceTimeRegionAnalyticFeatures final {
  std::uint64_t actorCount = 0;
  std::uint64_t computeActorCount = 0;
  std::uint64_t controlActorCount = 0;
  std::uint64_t memoryActorCount = 0;
  std::uint64_t graphCount = 0;
  std::uint64_t launchSynchronizationCost = 0;
  std::uint64_t parallelismLowerBound = 0;
  std::uint64_t topologyCongestionProxy = 0;
};

struct ResourceTimeRegionFeature final {
  ::dataflow::RootThreadLaunchRef region;
  std::vector<ResourceTimeDependencyFeature> dependencies;
  std::vector<ResourceTimeSpeedupPoint> speedupCurve;
  /// One means the region describes one logical epoch. A value greater than
  /// one denotes a partitioned logical domain; it cannot by itself establish
  /// temporal reuse, because System PnR realizes those partitions as distinct
  /// concurrent cells.
  std::uint64_t logicalEpochCount = 0;
  /// True only when `speedupCurve` enumerates every legal allocation count in
  /// this analytic scheduling domain. A no-fit result is a sound resource
  /// rejection only under this explicit completeness fact.
  bool allocationDomainExhaustive = false;
  ResourceTimeRegionAnalyticFeatures analyticFeatures;
};

struct ResourceTimeRegionResourceBound final {
  ::dataflow::RootThreadLaunchRef region;
  std::uint64_t maximumUsefulResourceUnits = 0;
  ResourceTimeEstimateSupport support =
      ResourceTimeEstimateSupport::Unsupported;
  /// The provider-owned exact lower feasibility boundary. Zero with
  /// Unsupported support means the projection has no lower-bound proof; an
  /// observed one-core Mapping must not manufacture one.
  std::uint64_t minimumFeasibleResourceUnits = 0;
  ResourceTimeEstimateSupport minimumSupport =
      ResourceTimeEstimateSupport::Unsupported;
};

/// One immutable, Dataflow-owned projection shared by every schedule state in
/// an invocation. The single aggregate AccCore class is a conservative
/// analytic capacity; heterogeneous compatibility remains a Mapping fact.
struct ResourceTimeDataflowProjection final {
  std::uint64_t acceleratedGraphCount = 0;
  std::uint64_t acceleratedActorCount = 0;
  std::vector<ArtifactRootReference> resourceClasses;
  std::vector<std::uint64_t> availableResourceUnits;
  std::vector<ResourceTimeRegionFeature> regions;
  std::vector<ResourceTimeRegionResourceBound> regionBounds;
};

llvm::Expected<ComponentViewDigest> resourceTimeAnalyticModelSnapshotDigest();

/// Derives the invocation-local key for the immutable Dataflow-to-resource-
/// time projection. The key binds the complete semantic invocation context;
/// it is a removable derived-cache key, not a candidate identity.
llvm::Expected<ComponentViewDigest> deriveResourceTimeProjectionCacheKey(
    const ResourceTimeInvocationKey &invocation);

/// Returns a saturating estimate of the retained bytes of one immutable
/// projection. The estimate is used only for deterministic invocation-local
/// cache admission and diagnostics.
std::uint64_t resourceTimeProjectionRetainedBytes(
    const ResourceTimeDataflowProjection &projection);

/// Projects static logical-domain capacity, root causality, and a bounded
/// analytic speedup curve once for one canonical Dataflow. Completion
/// dependencies are exact; an earlier token relation without a completion
/// proof remains a FIFO dependency whose latency is unsupported.
llvm::Expected<ResourceTimeDataflowProjection> projectResourceTimeDataflow(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &system,
    llvm::StringRef entrySymbol,
    std::optional<std::uint64_t> estimatedRuntimePicoseconds = std::nullopt);

struct ResourceTimeFrontierPolicy final {
  std::vector<std::uint64_t> availableResourceUnits;
  std::uint64_t maximumStatesGenerated = 4096;
  std::uint64_t maximumActionsGenerated = 16384;
  std::uint64_t maximumStateCacheEntries = 4096;
  std::uint64_t maximumRetainedBytes = 64ULL * 1024ULL * 1024ULL;
  /// Invocation-local exact frontier memo capacity. This memo is not a
  /// Mapping cache and never survives the enclosing funnel invocation; any
  /// concurrent single-flight sharing belongs to the enclosing DSE owner.
  std::uint64_t maximumInvocationMemoEntries = 64;
  std::uint64_t maximumInvocationMemoBytes = 8ULL * 1024ULL * 1024ULL;
  std::uint64_t beamWidth = 32;
  std::uint64_t maximumFinalists = 8;
  std::uint64_t maximumMappingFinalists = 4;
  /// Ranking focus for one bounded endpoint experiment. This is a hint only;
  /// the Spectrum verifier still owns the final class.
  PreMappingSpectrumEndpoint spectrumEndpoint =
      PreMappingSpectrumEndpoint::Automatic;
};

struct ResourceTimeWorkCounter final {
  std::uint64_t limit = 0;
  std::uint64_t planned = 0;
  std::uint64_t reserved = 0;
  std::uint64_t consumed = 0;
  std::uint64_t rejected = 0;
  std::uint64_t cancelled = 0;
  std::uint64_t elapsedNanoseconds = 0;
};

struct ResourceTimeFrontierAccounting final {
  ResourceTimeWorkCounter sourceProjections;
  ResourceTimeWorkCounter actions;
  ResourceTimeWorkCounter states;
  ResourceTimeWorkCounter estimates;
  ResourceTimeWorkCounter finalists;
  std::uint64_t stateMemoHits = 0;
  std::uint64_t stateMemoMisses = 0;
  /// Existing semantic future states admitted as a non-dominated path point.
  std::uint64_t stateMemoParetoInsertions = 0;
  std::uint64_t stateMemoDominatedStates = 0;
  std::uint64_t stateMemoHitCapacityRejections = 0;
  std::uint64_t stateMemoMissCapacityRejections = 0;
  std::uint64_t statesPrunedByBeam = 0;
  std::uint64_t terminalHintsGenerated = 0;
  std::uint64_t terminalHintsRetained = 0;
  std::uint64_t terminalHintsPruned = 0;
  /// Number of admitted states whose lower bound was updated from the parent
  /// event/action delta rather than by rescanning the frozen graph.
  std::uint64_t incrementalLowerBoundUpdates = 0;
  std::uint64_t maximumRetainedBytes = 0;
};

struct ResourceTimeConcurrencyBounds final {
  std::uint64_t minimumPeakConcurrentRegions = 0;
  std::uint64_t maximumPeakConcurrentRegions = 0;
  ResourceTimeEstimateSupport support =
      ResourceTimeEstimateSupport::Unsupported;
};

enum class ResourceTimeActionKind : std::uint8_t {
  AdmitRegion,
  AdvanceEvent,
};

/// Incremental mutation applied to one event-relative frontier state. This is
/// planning provenance, not a Mapping delta or a runtime transition ABI.
struct ResourceTimeActionDelta final {
  ResourceTimeActionKind kind = ResourceTimeActionKind::AdvanceEvent;
  std::optional<::dataflow::RootThreadLaunchRef> admittedRegion;
  std::optional<std::uint64_t> speedupPointOrdinal;
  std::uint64_t beforeTimePicoseconds = 0;
  std::uint64_t afterTimePicoseconds = 0;
  std::vector<::dataflow::RootThreadLaunchRef> completedRegions;
  std::vector<::dataflow::RootThreadLaunchRef> tokenReadyProducers;
  std::vector<::dataflow::RootThreadLaunchRef> newlyReadyRegions;

  friend bool operator==(const ResourceTimeActionDelta &lhs,
                         const ResourceTimeActionDelta &rhs) {
    return lhs.kind == rhs.kind && lhs.admittedRegion == rhs.admittedRegion &&
           lhs.speedupPointOrdinal == rhs.speedupPointOrdinal &&
           lhs.beforeTimePicoseconds == rhs.beforeTimePicoseconds &&
           lhs.afterTimePicoseconds == rhs.afterTimePicoseconds &&
           lhs.completedRegions == rhs.completedRegions &&
           lhs.tokenReadyProducers == rhs.tokenReadyProducers &&
           lhs.newlyReadyRegions == rhs.newlyReadyRegions;
  }
};

struct ResourceTimeHintAllocation final {
  ::dataflow::RootThreadLaunchRef region;
  std::uint64_t speedupPointOrdinal = 0;
  std::vector<std::uint64_t> resourceUnits;
  std::uint64_t completionTimePicoseconds = 0;
};

struct ResourceTimeHintState final {
  std::uint64_t timePicoseconds = 0;
  std::vector<ResourceTimeHintAllocation> active;
  std::vector<::dataflow::RootThreadLaunchRef> ready;
  std::vector<::dataflow::RootThreadLaunchRef> completed;
  std::uint64_t optimisticMakespanLowerBoundPicoseconds = 0;
};

/// A fast-model schedule proposal. It deliberately carries neither physical
/// bindings nor a MaxSpatial/MaxTemporal/intermediate classification.
struct ResourceTimeScheduleHint final {
  std::vector<ResourceTimeActionDelta> actions;
  std::vector<ResourceTimeHintState> states;
  std::uint64_t estimatedMakespanPicoseconds = 0;
  std::uint64_t optimisticMakespanLowerBoundPicoseconds = 0;
  std::uint64_t peakConcurrentRegions = 0;
  std::uint64_t totalAllocatedResourceTime = 0;
  ResourceTimeEstimateSupport support =
      ResourceTimeEstimateSupport::Unsupported;
};

/// Stable evaluation provenance for one schedule proposal. This digest is not
/// software candidate identity and never supplies Mapping legality.
llvm::Expected<ComponentViewDigest>
deriveResourceTimeScheduleHintDigest(const ResourceTimeScheduleHint &hint);

enum class ResourceTimeFrontierIncompleteReason : std::uint8_t {
  BudgetExhausted,
  CancelledOrTimeout,
  ProofNotEstablished,
  Unsupported,
};

llvm::StringRef resourceTimeFrontierIncompleteReasonSpelling(
    ResourceTimeFrontierIncompleteReason reason);

enum class ResourceTimeFrontierInfeasibleReason : std::uint8_t {
  CompletionDependencyCycle,
  ResourceCapacity,
};

llvm::StringRef resourceTimeFrontierInfeasibleReasonSpelling(
    ResourceTimeFrontierInfeasibleReason reason);

struct CompletedResourceTimeFrontier final {
  ResourceTimeInvocationKey invocation;
  std::vector<ResourceTimeScheduleHint> finalists;
  /// True only when no configured beam truncation discarded a reachable
  /// state. A false value is a completed bounded search, not an exhaustive
  /// proof about the schedule domain.
  bool domainExhaustive = false;
  std::optional<ResourceTimeConcurrencyBounds> concurrencyBounds;
  ResourceTimeFrontierAccounting accounting;
};

struct IncompleteResourceTimeFrontier final {
  ResourceTimeInvocationKey invocation;
  ResourceTimeFrontierIncompleteReason reason =
      ResourceTimeFrontierIncompleteReason::ProofNotEstablished;
  std::vector<ResourceTimeScheduleHint> retainedFinalists;
  ResourceTimeFrontierAccounting accounting;
};

struct ProvenInfeasibleResourceTimeFrontier final {
  ResourceTimeInvocationKey invocation;
  ResourceTimeFrontierInfeasibleReason reason =
      ResourceTimeFrontierInfeasibleReason::ResourceCapacity;
  ResourceTimeFrontierAccounting accounting;
};

using ResourceTimeFrontierOutcome =
    std::variant<CompletedResourceTimeFrontier, IncompleteResourceTimeFrontier,
                 ProvenInfeasibleResourceTimeFrontier>;

struct ResourceTimeMappingCandidateInput;
struct ResourceTimeMappingFunnel;

struct ResourceTimeFrontierSessionStatistics final {
  std::uint64_t requests = 0;
  std::uint64_t cacheHits = 0;
  std::uint64_t cacheMisses = 0;
  std::uint64_t singleFlightWaits = 0;
  std::uint64_t coalescedUncachedResults = 0;
  std::uint64_t cancelledWaits = 0;
  std::uint64_t capacityBypasses = 0;
  std::uint64_t entryCount = 0;
  std::uint64_t retainedBytes = 0;
};

/// Bounded removable state shared by workers of one resource-time funnel
/// invocation. It owns neither Mapping decisions nor persistent evidence.
/// Complete analytic results and sound fixed-input rejections may be retained;
/// incomplete results exist only long enough to coalesce an active flight.
class ResourceTimeFrontierSession final {
public:
  ResourceTimeFrontierSession(std::uint64_t maximumEntries,
                              std::uint64_t maximumRetainedBytes);
  ~ResourceTimeFrontierSession();

  ResourceTimeFrontierSession(const ResourceTimeFrontierSession &) = delete;
  ResourceTimeFrontierSession &
  operator=(const ResourceTimeFrontierSession &) = delete;

  ResourceTimeFrontierSessionStatistics statistics() const;

private:
  struct LookupResult final {
    std::shared_ptr<const ResourceTimeFrontierOutcome> outcome;
    bool cacheHit = false;
    bool cacheMiss = false;
    bool waited = false;
    bool coalescedUncachedResult = false;
    bool cancelledWait = false;
    bool capacityBypass = false;
  };

  using Compute = std::function<llvm::Expected<ResourceTimeFrontierOutcome>()>;

  llvm::Expected<LookupResult>
  lookupOrCompute(std::string key, Compute compute,
                  ExecutionControlView executionControl);

  class Impl;
  std::unique_ptr<Impl> impl_;

  friend llvm::Expected<ResourceTimeMappingFunnel>
  selectResourceTimeMappingFinalists(
      llvm::ArrayRef<ResourceTimeMappingCandidateInput>,
      const ResourceTimeFrontierPolicy &, ExecutionControlView,
      ResourceTimeFrontierSession *);
};

struct ResourceTimeMappingCandidateInput final {
  ComponentViewDigest candidateIdentity;
  std::uint64_t inputPreferenceRank = 0;
  std::uint64_t acceleratedRegionCount = 0;
  std::uint64_t acceleratedGraphCount = 0;
  std::uint64_t acceleratedActorCount = 0;
  std::uint64_t maximumUsefulResourceUnits = 0;
  ResourceTimeInvocationKey invocation;
  std::vector<ArtifactRootReference> resourceClasses;
  std::vector<ResourceTimeRegionFeature> regions;
};

enum class ResourceTimeCandidateFunnelDisposition : std::uint8_t {
  Estimated,
  SoundGateRejected,
  Incomplete,
};

llvm::StringRef resourceTimeCandidateFunnelDispositionSpelling(
    ResourceTimeCandidateFunnelDisposition disposition);

struct ResourceTimeCandidateFunnelEvaluation final {
  ComponentViewDigest candidateIdentity;
  std::uint64_t inputPreferenceRank = 0;
  std::uint64_t acceleratedRegionCount = 0;
  std::uint64_t acceleratedGraphCount = 0;
  std::uint64_t acceleratedActorCount = 0;
  std::uint64_t maximumUsefulResourceUnits = 0;
  ResourceTimeCandidateFunnelDisposition disposition =
      ResourceTimeCandidateFunnelDisposition::Incomplete;
  std::uint64_t screeningLowerBoundPicoseconds = 0;
  std::uint64_t screeningFeatureScore = 0;
  ResourceTimeEstimateSupport screeningSupport =
      ResourceTimeEstimateSupport::Unsupported;
  ResourceTimeEstimateConfidence screeningConfidence =
      ResourceTimeEstimateConfidence::None;
  bool detailedFrontierEvaluated = false;
  std::optional<ResourceTimeConcurrencyBounds> concurrencyBounds;
  std::optional<ResourceTimeScheduleHint> bestHint;
  std::vector<ResourceTimeScheduleHint> retainedHints;
  std::optional<ResourceTimeFrontierIncompleteReason> incompleteReason;
  std::optional<ResourceTimeFrontierInfeasibleReason> infeasibleReason;
  ResourceTimeFrontierAccounting frontierAccounting;
};

struct ResourceTimeMappingFunnelAccounting final {
  std::uint64_t generatedCandidates = 0;
  std::uint64_t screenedCandidates = 0;
  std::uint64_t detailedFrontierCandidates = 0;
  std::uint64_t successiveHalvingDeferredCandidates = 0;
  std::uint64_t soundGateRejectedCandidates = 0;
  std::uint64_t estimatedCandidates = 0;
  std::uint64_t incompleteCandidates = 0;
  std::uint64_t mappingEligibleScheduleHints = 0;
  std::uint64_t mappingFinalists = 0;
  /// Counts are filled by the application owner after the analytic funnel.
  /// They make the cheap-to-expensive boundary auditable without implying
  /// that workload materialization or plan construction proves legality.
  std::uint64_t functionalReplayCandidates = 0;
  /// Application-owner timing for the immutable Dataflow-to-resource-time
  /// projection. This is separate from frontier `sourceProjections`, which
  /// measures freezing one already projected feature set.
  std::uint64_t dataflowProjectionRequests = 0;
  std::uint64_t dataflowProjectionCacheHits = 0;
  std::uint64_t dataflowProjectionCacheMisses = 0;
  std::uint64_t dataflowProjectionCacheCapacityBypasses = 0;
  std::uint64_t dataflowProjectionCacheEntries = 0;
  std::uint64_t dataflowProjectionCacheRetainedBytes = 0;
  std::uint64_t dataflowProjectionElapsedNanoseconds = 0;
  std::uint64_t dataflowMaterializedCandidates = 0;
  std::uint64_t mappingPlanCandidates = 0;
  std::uint64_t unsupportedBeforeMappingCandidates = 0;
  std::uint64_t unsupportedBeforeMappingScheduleHints = 0;
  /// Application owner sets this after schedule finalists have been promoted
  /// or typed-unsupported before PnR. The analytic funnel alone has no plan
  /// disposition to validate.
  bool applicationPromotionAccountingComplete = false;
  /// Comparison of cheap screening facts with the detailed schedule frontier
  /// already computed for the bounded sample. Neither side is Mapping/PnR
  /// legality evidence.
  std::uint64_t screeningComparisonCandidates = 0;
  std::uint64_t detailedScheduleFeasibleCandidates = 0;
  std::uint64_t screeningAdmissibleCandidates = 0;
  std::uint64_t screeningDetailedFeasibleIntersection = 0;
  std::uint64_t screeningDetailedBestRankMatches = 0;
  std::uint64_t screeningOutOfDomainCandidates = 0;
  std::uint64_t maximumScreeningLowerBoundGapPicoseconds = 0;
  std::uint64_t screeningLowerBoundViolations = 0;
  /// Exact static Mapping inputs may be shared by several schedule hints. This
  /// counts avoided plan constructions, not skipped owner verification; the
  /// shared plan still undergoes ordinary Mapping and Spectrum verification.
  std::uint64_t mappingPlanConstructionsAvoidedByExactMemo = 0;
  std::uint64_t mappingCallsDeferredByModel = 0;
  std::uint64_t mappingCallsWithheldByIncomplete = 0;
  /// Exact memoization is invocation-local and only applies to a completely
  /// identical semantic frontier input. A hit never implies Mapping legality.
  std::uint64_t exactInvocationMemoHits = 0;
  std::uint64_t exactInvocationMemoMisses = 0;
  std::uint64_t exactInvocationMemoSingleFlightWaits = 0;
  std::uint64_t exactInvocationMemoCoalescedUncachedResults = 0;
  std::uint64_t exactInvocationMemoCancelledWaits = 0;
  std::uint64_t exactInvocationMemoCapacityBypasses = 0;
  std::uint64_t exactInvocationMemoEntries = 0;
  std::uint64_t exactInvocationMemoRetainedBytes = 0;
  /// Aggregate ledger for every candidate frontier that was actually
  /// evaluated. Unevaluated candidates have no reserved work and therefore do
  /// not contribute a synthetic budget entry.
  ResourceTimeFrontierAccounting frontierAccounting;
  std::uint64_t elapsedNanoseconds = 0;
};

struct ResourceTimeMappingFinalist final {
  ComponentViewDigest candidateIdentity;
  ComponentViewDigest scheduleHintDigest;

  friend bool operator==(const ResourceTimeMappingFinalist &lhs,
                         const ResourceTimeMappingFinalist &rhs) {
    return lhs.candidateIdentity == rhs.candidateIdentity &&
           lhs.scheduleHintDigest == rhs.scheduleHintDigest;
  }
};

struct ResourceTimeMappingFunnel final {
  std::vector<ResourceTimeCandidateFunnelEvaluation> evaluations;
  std::vector<ResourceTimeMappingFinalist> finalists;
  ResourceTimeMappingFunnelAccounting accounting;
  bool truncated = false;
  std::optional<ResourceTimeFrontierIncompleteReason> incompleteReason;
};

/// Applies the same bounded resource-time owner to several semantic software
/// candidates and returns only the identities admitted to real Mapping. An
/// unsupported estimate can retain a deterministic fallback but cannot reject
/// a candidate as infeasible.
llvm::Expected<ResourceTimeMappingFunnel> selectResourceTimeMappingFinalists(
    llvm::ArrayRef<ResourceTimeMappingCandidateInput> candidates,
    const ResourceTimeFrontierPolicy &policy,
    ExecutionControlView executionControl = {},
    ResourceTimeFrontierSession *session = nullptr);

llvm::Error validateResourceTimeMappingFunnelAccounting(
    const ResourceTimeMappingFunnelAccounting &accounting);

/// Explores a finite event-driven schedule frontier. The dependency/source
/// projection is consumed once; ready sets and resource pressure are updated
/// only through typed action deltas. Only sound fixed-input necessary
/// conditions can return ProvenInfeasible.
llvm::Expected<ResourceTimeFrontierOutcome> exploreResourceTimeFrontier(
    const ResourceTimeInvocationKey &invocation,
    llvm::ArrayRef<ArtifactRootReference> resourceClasses,
    llvm::ArrayRef<ResourceTimeRegionFeature> regions,
    const ResourceTimeFrontierPolicy &policy,
    ExecutionControlView executionControl = {});

llvm::Error validateResourceTimeFrontierAccounting(
    const ResourceTimeFrontierAccounting &accounting);

} // namespace loom::dse

#endif // LOOM_DSE_RESOURCETIMEFRONTIER_H
