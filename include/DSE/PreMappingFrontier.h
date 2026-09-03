#ifndef LOOM_DSE_PREMAPPINGFRONTIER_H
#define LOOM_DSE_PREMAPPINGFRONTIER_H

#include "Common/Artifact.h"
#include "Common/ComponentViewDigest.h"
#include "DSE/JointDesignPolicy.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Frontend/Analysis/StructuredProtocolDependencies.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace loom::dse {

/// Invocation-local work bounds for compiler candidate planning. Every bound
/// is consumed before its named expensive boundary. A zero value is invalid;
/// callers that need less work must omit the invocation instead of encoding an
/// ambiguous disabled mode.
struct PreMappingFrontierBudget final {
  /// The source observation is one invocation-wide immutable projection. It
  /// has its own ledger so its wall time cannot disappear into a provider
  /// counter or be repeated silently.
  std::uint64_t maximumSourceObservations = 1;
  std::uint64_t maximumCoordinatesGenerated = 64;
  std::uint64_t maximumProgramsMaterialized = 2048;
  std::uint64_t maximumAnalyticEvaluations = 512;
  std::uint64_t maximumFunctionalReplays = 128;
  std::uint64_t maximumDataflowPromotions = 64;
  std::uint64_t maximumMappingPairs = 32;
};

enum class PreMappingSpectrumEndpoint : std::uint8_t {
  Automatic,
  MaxTemporal,
  MaxSpatial,
  Intermediate,
};

llvm::StringRef toString(PreMappingSpectrumEndpoint value);

/// A schedule intent constrains candidate generation only. It is not an
/// endpoint classification and must never be reported as MaxTemporal or
/// MaxSpatial without a verified SystemMapping schedule.
enum class PreMappingScheduleIntent : std::uint8_t {
  Unconstrained,
  TemporalReuse,
  SpatialParallel,
};

llvm::StringRef toString(PreMappingScheduleIntent value);

enum class PreMappingSpectrumClass : std::uint8_t {
  MaxTemporal,
  MaxSpatial,
  Intermediate,
};

llvm::StringRef toString(PreMappingSpectrumClass value);

/// The exact verified class requested by a focused endpoint policy. Automatic
/// leaves class selection to the ordinary objective and diversity policy.
std::optional<PreMappingSpectrumClass>
spectrumClassForEndpoint(PreMappingSpectrumEndpoint endpoint);

enum class PreMappingSpectrumSeedKind : std::uint8_t {
  /// Legacy spellings remain readable in invocation evidence but are no
  /// longer emitted by the planner. Endpoint labels belong to verified
  /// SystemMapping schedules, never to a coordinate seed.
  MaxSpatial,
  MaxTemporal,
  HighActivitySingleton,
  DependencyEdge,
  ProducerGroup,
  PipelineGroup,
  ConnectedComponent,
  CanonicalFallback,
  Intermediate,
};

struct PreMappingFrontierPolicy final {
  PreMappingFrontierBudget budget;
  std::vector<std::uint64_t> beamWidthByExpansionDepth{4};
  std::uint64_t diversityCandidateCount = 3;
  /// Maximum graph-adjacency expansion depth used to form compositional
  /// ownership roots. The planner never expands a powerset beyond this
  /// bounded frontier.
  std::uint64_t maximumExpansionDepth = 2;
  std::uint64_t maximumCompositionalGroups = 32;
  /// Selects one bounded spectrum endpoint when an invocation is collecting a
  /// focused witness. Automatic retains the ordinary objective/diversity
  /// policy; the other values never increase TopK or any work bound.
  PreMappingSpectrumEndpoint spectrumEndpoint =
      PreMappingSpectrumEndpoint::Automatic;
  JointDesignStoppingPolicy stoppingPolicy =
      JointDesignStoppingPolicy::FirstVerified;

  std::uint64_t beamWidth(std::size_t expansionDepth) const;
  llvm::Expected<ComponentViewDigest> digest() const;
};

llvm::Error validatePreMappingFrontierPolicy(
    const PreMappingFrontierPolicy &policy);

struct PreMappingWorkCounter final {
  std::uint64_t limit = 0;
  std::uint64_t planned = 0;
  std::uint64_t consumed = 0;
  std::uint64_t reserved = 0;
  std::uint64_t rejected = 0;
  std::uint64_t cancelled = 0;
  std::uint64_t elapsedNanoseconds = 0;

  friend bool operator==(const PreMappingWorkCounter &lhs,
                         const PreMappingWorkCounter &rhs) {
    return lhs.limit == rhs.limit && lhs.planned == rhs.planned &&
           lhs.consumed == rhs.consumed && lhs.reserved == rhs.reserved &&
           lhs.rejected == rhs.rejected && lhs.cancelled == rhs.cancelled &&
           lhs.elapsedNanoseconds == rhs.elapsedNanoseconds;
  }
};

struct PreMappingWorkAccounting final {
  PreMappingWorkCounter sourceObservations;
  PreMappingWorkCounter coordinates;
  PreMappingWorkCounter programMaterializations;
  PreMappingWorkCounter analyticEvaluations;
  PreMappingWorkCounter functionalReplays;
  PreMappingWorkCounter dataflowPromotions;
  PreMappingWorkCounter mappingPairs;

  friend bool operator==(const PreMappingWorkAccounting &lhs,
                         const PreMappingWorkAccounting &rhs) {
    return lhs.sourceObservations == rhs.sourceObservations &&
           lhs.coordinates == rhs.coordinates &&
           lhs.programMaterializations == rhs.programMaterializations &&
           lhs.analyticEvaluations == rhs.analyticEvaluations &&
           lhs.functionalReplays == rhs.functionalReplays &&
           lhs.dataflowPromotions == rhs.dataflowPromotions &&
           lhs.mappingPairs == rhs.mappingPairs;
  }
};

PreMappingWorkAccounting
makePreMappingWorkAccounting(const PreMappingFrontierBudget &budget);

/// Returns an error when the invocation-local ledger cannot be reconciled.
/// Reservation is deliberately checked separately from consumption: a
/// rejected or cancelled unit must never look like consumed work.
llvm::Error validatePreMappingWorkAccounting(
    const PreMappingWorkAccounting &accounting);

enum class PreMappingLogicalDomainSupport : std::uint8_t {
  Exact,
  Partial,
  Unsupported,
};

enum class PreMappingExactGateDisposition : std::uint8_t {
  Admitted,
  Rejected,
};

enum class PreMappingEstimateSupport : std::uint8_t {
  Supported,
  Unsupported,
};

enum class PreMappingEstimateConfidence : std::uint8_t {
  None,
  Low,
  Calibrated,
  OutOfDistribution,
};

llvm::StringRef toString(PreMappingSpectrumSeedKind value);
llvm::StringRef toString(PreMappingLogicalDomainSupport value);
llvm::StringRef toString(PreMappingExactGateDisposition value);
llvm::StringRef toString(PreMappingEstimateSupport value);
llvm::StringRef toString(PreMappingEstimateConfidence value);

struct PreMappingTemporalWitness final {
  /// Logical epochs are the finite temporal points induced by the rooted
  /// Dataflow launches. They are provenance, not a second executable graph.
  std::uint64_t logicalEpochCount = 0;
  std::uint64_t accCoreOccupancy = 0;
  std::uint64_t launchCount = 0;
  std::uint64_t synchronizationCount = 0;
  std::uint64_t liveStateBytes = 0;
  bool liveStateKnown = false;
  bool exact = false;

  friend bool operator==(const PreMappingTemporalWitness &lhs,
                         const PreMappingTemporalWitness &rhs) {
    return lhs.logicalEpochCount == rhs.logicalEpochCount &&
           lhs.accCoreOccupancy == rhs.accCoreOccupancy &&
           lhs.launchCount == rhs.launchCount &&
           lhs.synchronizationCount == rhs.synchronizationCount &&
           lhs.liveStateBytes == rhs.liveStateBytes &&
           lhs.liveStateKnown == rhs.liveStateKnown && lhs.exact == rhs.exact;
  }
};

/// Candidate-specific projection mechanically derived from one immutable
/// source relation and one owned root set. Known and unknown communication are
/// deliberately separate. `estimatedCutTrafficBytes` is absent on overflow or
/// when any cut object extent is unknown.
struct PreMappingCandidateProjection final {
  ComponentViewDigest identity;

  explicit PreMappingCandidateProjection(ComponentViewDigest identity)
      : identity(std::move(identity)) {}
  std::uint64_t ownedRegionCount = 0;
  std::uint64_t hostRegionCount = 0;
  std::uint64_t internalDependencyCount = 0;
  std::uint64_t internalKnownBytes = 0;
  std::uint64_t internalUnknownObjectCount = 0;
  std::uint64_t cutDependencyCount = 0;
  std::uint64_t cutKnownBytes = 0;
  std::uint64_t cutUnknownObjectCount = 0;
  std::uint64_t unknownInternalPairCount = 0;
  std::uint64_t unknownCutPairCount = 0;
  std::uint64_t channelOpportunityCount = 0;
  std::uint64_t maximumProducerFanout = 0;
  std::uint64_t ownedDynamicActivations = 0;
  std::uint64_t ownedDynamicLeafExecutions = 0;
  std::uint64_t hostDynamicActivations = 0;
  std::uint64_t hostDynamicLeafExecutions = 0;
  std::optional<std::uint64_t> estimatedCutTrafficBytes;
  std::uint64_t producerRateLowerBound = 0;
  std::uint64_t consumerRateLowerBound = 0;
  std::uint64_t channelDepthLowerBound = 0;
  std::uint64_t launchSynchronizationCost = 0;
  std::uint64_t parallelismLowerBound = 0;
  std::uint64_t topologyCongestionProxy = 0;
  std::uint64_t reconfigurationLiveStateBytes = 0;
  bool reconfigurationLiveStateKnown = false;
  PreMappingExactGateDisposition exactGate =
      PreMappingExactGateDisposition::Admitted;
  PreMappingEstimateSupport estimateSupport =
      PreMappingEstimateSupport::Unsupported;
  PreMappingEstimateConfidence estimateConfidence =
      PreMappingEstimateConfidence::None;

  friend bool operator==(const PreMappingCandidateProjection &lhs,
                         const PreMappingCandidateProjection &rhs) {
    return lhs.identity == rhs.identity &&
           lhs.ownedRegionCount == rhs.ownedRegionCount &&
           lhs.hostRegionCount == rhs.hostRegionCount &&
           lhs.internalDependencyCount == rhs.internalDependencyCount &&
           lhs.internalKnownBytes == rhs.internalKnownBytes &&
           lhs.internalUnknownObjectCount ==
               rhs.internalUnknownObjectCount &&
           lhs.cutDependencyCount == rhs.cutDependencyCount &&
           lhs.cutKnownBytes == rhs.cutKnownBytes &&
           lhs.cutUnknownObjectCount == rhs.cutUnknownObjectCount &&
           lhs.unknownInternalPairCount == rhs.unknownInternalPairCount &&
           lhs.unknownCutPairCount == rhs.unknownCutPairCount &&
           lhs.channelOpportunityCount == rhs.channelOpportunityCount &&
           lhs.maximumProducerFanout == rhs.maximumProducerFanout &&
           lhs.ownedDynamicActivations == rhs.ownedDynamicActivations &&
           lhs.ownedDynamicLeafExecutions ==
               rhs.ownedDynamicLeafExecutions &&
           lhs.hostDynamicActivations == rhs.hostDynamicActivations &&
           lhs.hostDynamicLeafExecutions ==
               rhs.hostDynamicLeafExecutions &&
           lhs.estimatedCutTrafficBytes == rhs.estimatedCutTrafficBytes &&
           lhs.producerRateLowerBound == rhs.producerRateLowerBound &&
           lhs.consumerRateLowerBound == rhs.consumerRateLowerBound &&
           lhs.channelDepthLowerBound == rhs.channelDepthLowerBound &&
           lhs.launchSynchronizationCost ==
               rhs.launchSynchronizationCost &&
           lhs.parallelismLowerBound == rhs.parallelismLowerBound &&
           lhs.topologyCongestionProxy == rhs.topologyCongestionProxy &&
           lhs.reconfigurationLiveStateBytes ==
               rhs.reconfigurationLiveStateBytes &&
           lhs.reconfigurationLiveStateKnown ==
               rhs.reconfigurationLiveStateKnown &&
           lhs.exactGate == rhs.exactGate &&
           lhs.estimateSupport == rhs.estimateSupport &&
           lhs.estimateConfidence == rhs.estimateConfidence;
  }
};

struct PreMappingCoordinate final {
  std::vector<std::size_t> ownedProtocolOrdinals;
  std::vector<PreMappingSpectrumSeedKind> seedKinds;
  PreMappingCandidateProjection projection;
  PreMappingScheduleIntent scheduleIntent =
      PreMappingScheduleIntent::Unconstrained;
  std::optional<PreMappingTemporalWitness> temporalWitness;

  friend bool operator==(const PreMappingCoordinate &lhs,
                         const PreMappingCoordinate &rhs) {
    return lhs.ownedProtocolOrdinals == rhs.ownedProtocolOrdinals &&
           lhs.seedKinds == rhs.seedKinds &&
           lhs.projection == rhs.projection &&
           lhs.scheduleIntent == rhs.scheduleIntent &&
           lhs.temporalWitness == rhs.temporalWitness;
  }
};

/// Candidate-specific facts available only after bounded Dataflow
/// materialization. Static-domain fields are exact when support is Exact;
/// Partial keeps the exact known subset and an explicit unknown count. The
/// wave count is an optimistic scheduling lower bound, never a legality gate.
struct PreMappingMaterializedProjection final {
  ComponentViewDigest identity;

  explicit PreMappingMaterializedProjection(ComponentViewDigest identity)
      : identity(std::move(identity)) {}
  std::uint64_t rootThreadLaunchCount = 0;
  std::uint64_t rootedGraphLaunchCount = 0;
  std::uint64_t staticLogicalDomainPointCount = 0;
  std::uint64_t unknownLogicalDomainCount = 0;
  std::uint64_t availableAccCoreCount = 0;
  std::optional<std::uint64_t> minimumExecutionWaves;
  std::optional<std::uint64_t> maximumParallelAccCoreCount;
  std::uint64_t actorCount = 0;
  std::uint64_t computeActorCount = 0;
  std::uint64_t controlActorCount = 0;
  std::uint64_t memoryActorCount = 0;
  std::uint64_t graphEdgeCount = 0;
  std::uint64_t logicalMemoryRootCount = 0;
  std::uint64_t streamActorCount = 0;
  std::uint64_t systemTransportResourceCount = 0;
  std::uint64_t systemTransferPatternCount = 0;
  PreMappingTemporalWitness temporalWitness;
  PreMappingLogicalDomainSupport logicalDomainSupport =
      PreMappingLogicalDomainSupport::Unsupported;

  friend bool operator==(const PreMappingMaterializedProjection &lhs,
                         const PreMappingMaterializedProjection &rhs) {
    return lhs.identity == rhs.identity &&
           lhs.rootThreadLaunchCount == rhs.rootThreadLaunchCount &&
           lhs.rootedGraphLaunchCount == rhs.rootedGraphLaunchCount &&
           lhs.staticLogicalDomainPointCount ==
               rhs.staticLogicalDomainPointCount &&
           lhs.unknownLogicalDomainCount == rhs.unknownLogicalDomainCount &&
           lhs.availableAccCoreCount == rhs.availableAccCoreCount &&
           lhs.minimumExecutionWaves == rhs.minimumExecutionWaves &&
           lhs.maximumParallelAccCoreCount ==
               rhs.maximumParallelAccCoreCount &&
           lhs.actorCount == rhs.actorCount &&
           lhs.computeActorCount == rhs.computeActorCount &&
           lhs.controlActorCount == rhs.controlActorCount &&
           lhs.memoryActorCount == rhs.memoryActorCount &&
           lhs.graphEdgeCount == rhs.graphEdgeCount &&
           lhs.logicalMemoryRootCount == rhs.logicalMemoryRootCount &&
           lhs.streamActorCount == rhs.streamActorCount &&
           lhs.systemTransportResourceCount ==
               rhs.systemTransportResourceCount &&
           lhs.systemTransferPatternCount == rhs.systemTransferPatternCount &&
           lhs.temporalWitness == rhs.temporalWitness &&
           lhs.logicalDomainSupport == rhs.logicalDomainSupport;
  }
};

struct PreMappingCoordinatePlan final {
  std::vector<PreMappingCoordinate> coordinates;
  std::uint64_t eligibleCoordinateCount = 0;
  bool truncated = false;
};

struct PreMappingShadowRecall final {
  std::uint64_t eligibleSubsets = 0;
  std::uint64_t generatedSubsets = 0;
  std::uint64_t coveredSubsets = 0;
  std::vector<std::vector<std::size_t>> missingSubsets;

  double recall() const {
    return eligibleSubsets == 0
               ? 1.0
               : static_cast<double>(coveredSubsets) /
                     static_cast<double>(eligibleSubsets);
  }
};

/// Exhaustive shadow oracle for small root domains. It is intentionally
/// bounded and must never be used as the production planner.
llvm::Expected<PreMappingShadowRecall> evaluatePreMappingShadowRecall(
    std::size_t rootCount, const PreMappingCoordinatePlan &plan,
    std::size_t maximumRoots = 4);

struct PreMappingFrontierCandidate final {
  ArtifactRootReference candidate;
  PreMappingCandidateProjection projection;
  std::optional<std::uint64_t> estimatedRuntimePicoseconds;
  PreMappingScheduleIntent scheduleIntent =
      PreMappingScheduleIntent::Unconstrained;
  std::vector<PreMappingSpectrumSeedKind> seedKinds;
  /// Set only by a verified SystemMapping schedule. Planning hints and
  /// logical-domain facts deliberately do not populate this field.
  std::optional<PreMappingSpectrumClass> verifiedSpectrum;
};

struct PreMappingFrontierSelection final {
  std::vector<ArtifactRootReference> paretoFrontier;
  std::vector<ArtifactRootReference> preferenceOrder;
  /// Projection identities paired with `preferenceOrder`. A final Structured
  /// Artifact may be reached by more than one planning coordinate; the
  /// central selector keeps one objective-best representative and callers
  /// must use this identity when recovering its lineage.
  std::vector<ComponentViewDigest> preferenceProjectionIdentities;
};

/// Applies the central Objective/Pareto/TopK owners to one already bounded
/// candidate set. Unsupported estimates remain rankable but cannot become an
/// exact rejection. The preference order retains objective-best, smallest
/// ownership, and largest ownership representatives before filling from the
/// common total order.
llvm::Expected<PreMappingFrontierSelection> selectPreMappingFrontier(
    llvm::ArrayRef<PreMappingFrontierCandidate> candidates,
    std::uint64_t maximumRetained, std::uint64_t diversityCandidateCount,
    PreMappingSpectrumEndpoint endpoint =
        PreMappingSpectrumEndpoint::Automatic);

struct PreMappingRootActivity final {
  frontend::StructuredEntityRef root;
  std::uint64_t dynamicActivations = 0;
  std::uint64_t dynamicLeafExecutions = 0;
};

/// Builds deterministic spectrum coordinates without materializing a program
/// or enumerating the root powerset. Work is linear in the explicit relation
/// plus the bounded coordinate payload.
llvm::Expected<PreMappingCoordinatePlan> buildPreMappingCoordinatePlan(
    llvm::ArrayRef<frontend::StructuredEntityRef> roots,
    const frontend::analysis::StructuredProtocolDependencyProjection
        &dependencies,
    llvm::ArrayRef<PreMappingRootActivity> activity,
    const PreMappingFrontierPolicy &policy,
    PreMappingWorkAccounting &accounting);

llvm::Expected<PreMappingMaterializedProjection>
projectPreMappingMaterializedCandidate(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const fabric::FabricSystemRootView &system, llvm::StringRef entrySymbol);

} // namespace loom::dse

#endif // LOOM_DSE_PREMAPPINGFRONTIER_H
