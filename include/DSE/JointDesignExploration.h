#ifndef LOOM_DSE_JOINTDESIGNEXPLORATION_H
#define LOOM_DSE_JOINTDESIGNEXPLORATION_H

#include "Config/ResolvedConfig.h"
#include "DSE/JointDesignPolicy.h"
#include "DSE/PlanExecutor.h"
#include "DSE/Promotion.h"
#include "DSE/SpatialRuntimeFeedback.h"
#include "Fabric/Identity/FabricRefs.h"
#include "PnR/PnrConfig.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::dse {

struct JointDesignPlanPair final {
  JointDesignPair pair;
  std::vector<PlanOutputRef> techMappings;
  std::vector<PlanOutputRef> spatialMappings;
  std::vector<ArtifactRootReference> immutableTechMappings;
  std::vector<ArtifactRootReference> immutableSpatialMappings;
  PlanOutputRef systemMappings;
};

/// Exact child-owned Mapping roots retained across one hardware derivation.
/// The ordinary plan builder validates their Dataflow and Module ownership;
/// unmatched target Modules continue through the canonical Generate path.
struct JointDesignMappingSeed final {
  std::vector<ArtifactRootReference> techMappings;
  std::vector<ArtifactRootReference> spatialMappings;
  struct SpatialRepairConstraint final {
    ArtifactRootReference techMapping;
    ArtifactRootReference constraintSet;
  };
  std::vector<SpatialRepairConstraint> spatialRepairConstraints;
};

struct JointDesignExplorationPlan final {
  ResolvedConfig resolvedConfig;
  BoundedJointFrontier frontier;
  std::vector<JointDesignPlanPair> pairOutputs;
  std::vector<pnr::SystemBindingPartitionIntent> systemBindingPartitions;
};

/// Builds one ordinary finite Generate plan. Each explicit application/System
/// pair traverses application-scoped TechMapping, SpatialMapping, and
/// SystemMapping under one exact System MappingConstraintSet.
llvm::Expected<JointDesignExplorationPlan> buildJointDesignExplorationPlan(
    JointDesignInputs inputs,
    llvm::ArrayRef<ArtifactRootReference> physicalTimingProfiles,
    const JointDesignPolicy &policy, const ResolvedConfig &baseConfig,
    const ArtifactStore &artifactStore,
    const JointDesignMappingSeed *mappingSeed = nullptr,
    llvm::ArrayRef<pnr::SystemBindingPartitionIntent> systemBindingPartitions =
        {});

/// Projects the canonical unique Module roots selected by every AccCore in one
/// exact System. Mapping-plan construction and bounded hardware reopen share
/// this projection rather than reconstructing System attachment ownership.
llvm::Expected<std::vector<ArtifactRootReference>>
projectJointDesignTargetModules(const ArtifactRootReference &system,
                                const ArtifactStore &artifactStore);

/// Projects the complete persistent semantic input closure of an authored
/// joint plan. The result includes frontier workloads and every exact Artifact
/// binding embedded in the resolved plan, and is canonical and unique.
std::vector<ArtifactRootReference>
projectJointDesignSemanticInputs(const JointDesignExplorationPlan &plan);

struct JointMappedPair final {
  JointDesignPair pair;
  std::vector<ArtifactRootReference> systemMappings;
};

enum class JointDesignQualityDisposition : std::uint8_t {
  NotRequested,
  Complete,
  Unsupported,
  ProofNotEstablished,
  ExecutionFailed,
  CancelledOrTimeout,
};

enum class JointDesignAttemptDisposition : std::uint8_t {
  Verified,
  ProvenNoFeasibleCandidate,
  Incomplete,
};

enum class JointDesignQualityIncompleteReason : std::uint8_t {
  Unsupported,
  ProofNotEstablished,
  ExecutionFailed,
  CancelledOrTimeout,
};

JointDesignQualityDisposition jointDesignQualityDisposition(
    JointDesignQualityIncompleteReason reason);

llvm::StringRef jointDesignQualityIncompleteReasonSpelling(
    JointDesignQualityIncompleteReason reason);

/// Runtime completion is an acquisition-owner fact, independent of whether a
/// later quality stage can publish a complete objective. In particular, an
/// FPA refusal must not erase an already completed Application replay.
enum class JointDesignQualityRuntimeCompletion : std::uint8_t {
  NotEstablished,
  Completed,
};

llvm::StringRef jointDesignQualityRuntimeCompletionSpelling(
    JointDesignQualityRuntimeCompletion completion);

/// Support of the calibrated model evaluated by the acquisition owner. This
/// is distinct from analytic Mapping support and from the quality outcome: an
/// out-of-domain model produces typed incomplete quality after runtime may
/// already have completed.
enum class JointDesignCalibratedModelSupport : std::uint8_t {
  NotEvaluated,
  InDomain,
  OutOfDomain,
};

llvm::StringRef jointDesignCalibratedModelSupportSpelling(
    JointDesignCalibratedModelSupport support);

/// Exact reusable facts returned by a quality acquisition before Objective
/// quantization. Supporting Evidence establishes the measures; verification
/// Evidence independently checks the candidate and may be a typed subset.
struct JointDesignQualityProvenance final {
  JointDesignQualityProvenance() = default;
  JointDesignQualityProvenance(
      std::vector<ResolvedObjectiveScalar> rawMeasures,
      std::vector<ArtifactRootReference> supportingEvidence,
      std::vector<ArtifactRootReference> verificationEvidence,
      std::optional<SpatialFifoRuntimeFeedback> spatialFifoFeedback = {},
      std::optional<SpatialOperandQueueRuntimeFeedback>
          spatialOperandQueueFeedback = {},
      std::optional<SpatialTransportRuntimeFeedback>
          spatialTransportFeedback = {},
      std::optional<std::uint64_t> resourceCoreCost = {},
      JointDesignQualityRuntimeCompletion runtimeCompletion =
          JointDesignQualityRuntimeCompletion::NotEstablished,
      JointDesignCalibratedModelSupport calibratedModelSupport =
          JointDesignCalibratedModelSupport::NotEvaluated)
      : rawMeasures(std::move(rawMeasures)),
        supportingEvidence(std::move(supportingEvidence)),
        verificationEvidence(std::move(verificationEvidence)),
        spatialFifoFeedback(std::move(spatialFifoFeedback)),
        spatialOperandQueueFeedback(std::move(spatialOperandQueueFeedback)),
        spatialTransportFeedback(std::move(spatialTransportFeedback)),
        resourceCoreCost(resourceCoreCost),
        runtimeCompletion(runtimeCompletion),
        calibratedModelSupport(calibratedModelSupport) {}

  /// Exact pre-quantization measures returned by the acquisition owner. An
  /// empty vector means that the policy did not publish reusable measures;
  /// otherwise ObjectiveProgram must reproduce objectiveCodes from it.
  std::vector<ResolvedObjectiveScalar> rawMeasures;
  /// Additional Evaluation Evidence consumed while producing rawMeasures.
  /// The optional evidence above remains the acquisition's primary Evidence.
  std::vector<ArtifactRootReference> supportingEvidence;
  /// Completed Evidence which independently verifies the acquired candidate.
  std::vector<ArtifactRootReference> verificationEvidence;
  std::optional<SpatialFifoRuntimeFeedback> spatialFifoFeedback;
  std::optional<SpatialOperandQueueRuntimeFeedback> spatialOperandQueueFeedback;
  std::optional<SpatialTransportRuntimeFeedback> spatialTransportFeedback;
  /// Exact System resource count imported by an ApplicationRuntime policy.
  /// It remains available when runtime or FPA acquisition is incomplete and
  /// therefore cannot publish a complete raw Objective vector.
  std::optional<std::uint64_t> resourceCoreCost;
  JointDesignQualityRuntimeCompletion runtimeCompletion =
      JointDesignQualityRuntimeCompletion::NotEstablished;
  JointDesignCalibratedModelSupport calibratedModelSupport =
      JointDesignCalibratedModelSupport::NotEvaluated;

  friend bool operator==(const JointDesignQualityProvenance &lhs,
                         const JointDesignQualityProvenance &rhs) {
    return lhs.rawMeasures == rhs.rawMeasures &&
           lhs.supportingEvidence == rhs.supportingEvidence &&
           lhs.verificationEvidence == rhs.verificationEvidence &&
           lhs.spatialFifoFeedback == rhs.spatialFifoFeedback &&
           lhs.spatialOperandQueueFeedback ==
               rhs.spatialOperandQueueFeedback &&
           lhs.spatialTransportFeedback == rhs.spatialTransportFeedback &&
           lhs.resourceCoreCost == rhs.resourceCoreCost &&
           lhs.runtimeCompletion == rhs.runtimeCompletion &&
           lhs.calibratedModelSupport == rhs.calibratedModelSupport;
  }
  friend bool operator!=(const JointDesignQualityProvenance &lhs,
                         const JointDesignQualityProvenance &rhs) {
    return !(lhs == rhs);
  }
};

/// Invocation-local QoR observation for one concrete SystemMapping. A
/// missing objective is explicit typed evidence; it is never represented by a
/// sentinel score.
struct JointDesignQualityObservation final {
  ArtifactRootReference candidate;
  std::vector<std::uint64_t> objectiveCodes;
  std::optional<JointDesignQualityIncompleteReason> incompleteReason;
  std::optional<ArtifactRootReference> evidence;
  JointDesignQualityProvenance provenance{};
};

/// Pre-Mapping quality observation for one exact software/System plan. The
/// promoted bit records admission to additional exact Mapping/PnR work; it is
/// never a feasibility claim for the parent or any generated child.
struct JointHardwarePromotionObservation final {
  std::uint64_t planOrdinal = 0;
  ArtifactRootReference system;
  std::vector<std::uint64_t> objectiveCodes;
  std::optional<JointDesignQualityIncompleteReason> incompleteReason;
  std::optional<ArtifactRootReference> evidence;
  bool promotedToExactMapping = false;
  JointDesignQualityProvenance provenance{};
};

/// One exact software-plan outcome retained independently of the final
/// stopping-policy winner. The plan ordinal joins mechanically to the
/// caller's bounded software frontier.
struct JointDesignAttemptRecord final {
  std::uint64_t planOrdinal = 0;
  ArtifactRootReference system;
  JointDesignAttemptDisposition disposition =
      JointDesignAttemptDisposition::Incomplete;
  std::optional<std::uint64_t> incompleteNodeOrdinal;
  std::optional<DsePlanIncompleteReason> incompleteReason;
  std::vector<ArtifactRootReference> systemMappings;
  /// Exact pre-Mapping parent whose bounded promotion caused this hardware
  /// attempt. Base Mapping attempts leave it absent; the attempt's System and
  /// Mapping roots remain the actual child outcome.
  std::optional<ArtifactRootReference> hardwarePromotionParentSystem;
};

/// The bounded analytic observations for retained pairs. These observations
/// are ranking provenance only; the pair still requires the ordinary Mapping
/// providers and independent verifier before it can be selected.
struct JointPairAnalyticObservation final {
  ArtifactRootReference dataflow;
  ArtifactRootReference system;
  JointPairAnalyticProjection projection;
};

struct JointDesignExecutionSummary final {
  JointDesignStoppingPolicy stoppingPolicy =
      JointDesignStoppingPolicy::FirstVerified;
  /// Frontier accounting is kept separate from Mapping outcomes. Deferred
  /// pairs have not been verified and must not be reported as infeasible.
  std::uint64_t eligibleJointPairCount = 0;
  std::uint64_t analyticEvaluatedJointPairCount = 0;
  std::uint64_t analyticDeferredJointPairCount = 0;
  std::uint64_t retainedJointPairCount = 0;
  bool jointFrontierTruncated = false;
  std::vector<JointPairAnalyticObservation> retainedJointPairAnalytics;
  std::uint64_t attemptedSoftwarePlans = 0;
  std::uint64_t hardwareReopenSearches = 0;
  std::uint64_t hardwareParentPromotions = 0;
  std::uint64_t hardwareReopensDeferredByQuality = 0;
  std::uint64_t hardwareReopensWithheldWithoutExactFeedback = 0;
  std::uint64_t hardwareRepairProbeLimit = 0;
  std::uint64_t hardwareRepairProbesPlanned = 0;
  std::uint64_t hardwareRepairProbesReserved = 0;
  std::uint64_t hardwareRepairProbesConsumed = 0;
  std::uint64_t hardwareRepairProbesRejected = 0;
  std::uint64_t hardwareRepairProbesCancelled = 0;
  std::uint64_t spatialMappingRepairCandidateLimit = 0;
  std::uint64_t spatialMappingRepairsPlanned = 0;
  std::uint64_t spatialMappingRepairsReserved = 0;
  std::uint64_t spatialMappingRepairsConsumed = 0;
  std::uint64_t spatialMappingRepairsRejected = 0;
  std::uint64_t spatialMappingRepairsCancelled = 0;
  /// Provider invocation and dispatch accounting. A journal replay is an
  /// available invocation result without a provider dispatch; it remains
  /// distinct from exact resource-time memoization.
  std::uint64_t techMappingInvocationCount = 0;
  std::uint64_t spatialPnrInvocationCount = 0;
  std::uint64_t systemPnrInvocationCount = 0;
  std::uint64_t techMappingDispatchCount = 0;
  std::uint64_t spatialPnrDispatchCount = 0;
  std::uint64_t systemPnrDispatchCount = 0;
  std::uint64_t techMappingJournalReplayCount = 0;
  std::uint64_t spatialPnrJournalReplayCount = 0;
  std::uint64_t systemPnrJournalReplayCount = 0;
  /// Wall time of this exact plan execution. The enclosing workflow classifies
  /// it as a cold or incremental attempt; the generic executor cannot infer
  /// that semantic distinction from the plan alone.
  std::uint64_t executionWallTimeNanoseconds = 0;
  std::uint64_t coldReopenWallTimeNanoseconds = 0;
  std::uint64_t incrementalReopenWallTimeNanoseconds = 0;
  std::optional<std::uint64_t> timeToFirstFeasibleWallTimeNanoseconds;
  std::optional<std::uint64_t> timeToBestWallTimeNanoseconds;
  std::uint64_t preservedTechMappings = 0;
  std::uint64_t preservedSpatialMappings = 0;
  std::uint64_t repairedTechMappings = 0;
  std::uint64_t repairedSpatialMappings = 0;
  std::uint64_t invalidatedTechMappings = 0;
  std::uint64_t invalidatedSpatialMappings = 0;
  std::uint64_t parentTechDecisions = 0;
  std::uint64_t parentSpatialDecisions = 0;
  std::uint64_t preservedTechDecisions = 0;
  std::uint64_t preservedSpatialDecisions = 0;
  std::uint64_t reopenedTechDecisions = 0;
  std::uint64_t reopenedSpatialDecisions = 0;
  std::uint64_t repairedTechDecisions = 0;
  std::uint64_t repairedSpatialDecisions = 0;
  std::uint64_t invalidationRootCount = 0;
  std::uint64_t invalidationConeDecisionCount = 0;
  std::uint64_t parentRouteNodeCount = 0;
  std::uint64_t preservedRouteNodeCount = 0;
  std::uint64_t reopenedRouteNodeCount = 0;
  std::uint64_t repairedRouteNodeCount = 0;
  std::uint64_t parentServiceLegCount = 0;
  std::uint64_t preservedServiceLegCount = 0;
  std::uint64_t reopenedServiceLegCount = 0;
  std::uint64_t parentThreadBindingCount = 0;
  std::uint64_t preservedThreadBindingCount = 0;
  std::uint64_t reopenedThreadBindingCount = 0;
  std::uint64_t parentGraphBindingCount = 0;
  std::uint64_t preservedGraphBindingCount = 0;
  std::uint64_t reopenedGraphBindingCount = 0;
  std::uint64_t parentResourceUseCount = 0;
  std::uint64_t preservedResourceUseCount = 0;
  std::uint64_t reopenedResourceUseCount = 0;
  std::uint64_t parentServiceRealizationCount = 0;
  std::uint64_t preservedServiceRealizationCount = 0;
  std::uint64_t reopenedServiceRealizationCount = 0;
  std::uint64_t verifiedAlternatives = 0;
  std::optional<std::uint64_t> selectedPlanOrdinal;
  std::optional<ArtifactRootReference> selectedMapping;
  JointDesignQualityDisposition qualityDisposition =
      JointDesignQualityDisposition::NotRequested;
  std::optional<ArtifactRootReference> qualityIncompleteCandidate;
  std::vector<std::string> qualityObjectiveDimensionLabels;
  std::vector<JointDesignQualityObservation> qualityObservations;
  std::vector<std::string> hardwarePromotionObjectiveDimensionLabels;
  std::vector<JointHardwarePromotionObservation> hardwarePromotionObservations;
  bool declaredWorkExhausted = false;
  std::vector<JointDesignAttemptRecord> attempts;
};

using JointDesignInvocationManifestReference = InvocationManifestReference;

llvm::Expected<JointDesignInvocationManifestReference>
publishJointDesignInvocationManifest(const InvocationManifest &manifest,
                                     const ResolvedConfig &resolvedConfig,
                                     const ArtifactStore &artifacts,
                                     const BlobStore &blobs);

llvm::Expected<InvocationManifest> importJointDesignInvocationManifest(
    const JointDesignInvocationManifestReference &reference,
    const ArtifactStore &artifacts, const BlobStore &blobs);

class JointDesignExecutionManifestBinder;

class JointDesignExecution final {
public:
  JointDesignExecution(DsePlanExecutionResult planExecution,
                       std::vector<JointMappedPair> mappedPairs,
                       JointDesignExecutionSummary summary)
      : planExecution(std::move(planExecution)),
        mappedPairs(std::move(mappedPairs)), summary(std::move(summary)) {}

  DsePlanExecutionResult planExecution;
  std::vector<JointMappedPair> mappedPairs;
  JointDesignExecutionSummary summary;

  const std::optional<JointDesignInvocationManifestReference> &
  invocationManifest() const {
    return invocationManifest_;
  }
  llvm::ArrayRef<JointDesignInvocationManifestReference>
  supportingInvocationManifests() const {
    return supportingInvocationManifests_;
  }
  std::optional<std::array<std::uint8_t, 32>> invocationRunKey() const {
    if (!invocationManifest_)
      return std::nullopt;
    return invocationManifest_->occurrence().runKey.bytes();
  }

private:
  std::optional<JointDesignInvocationManifestReference> invocationManifest_;
  std::vector<JointDesignInvocationManifestReference>
      supportingInvocationManifests_;

  friend class JointDesignExecutionManifestBinder;
};

/// Recovers the exact Spatial MappingConstraintSet that admitted one generated
/// SpatialMapping. Root-complete generation mechanically yields the canonical
/// empty set; explicit repair generation quotes its resolved constraint input.
/// An immutable seed with no retained constraint lineage, or an artifact
/// reproduced under more than one constraint root in the same execution,
/// returns `nullopt` rather than inventing a current owner.
llvm::Expected<std::optional<ArtifactRootReference>>
projectJointSpatialMappingConstraintSet(
    const JointDesignExecution &execution,
    const ArtifactRootReference &spatialMapping,
    const ArtifactStore &artifactStore);

struct IncompleteJointDesignQuality final {
  JointDesignQualityIncompleteReason reason =
      JointDesignQualityIncompleteReason::ProofNotEstablished;
  std::optional<ArtifactRootReference> candidate;
  std::optional<ArtifactRootReference> evidence;
  JointDesignQualityProvenance provenance{};
};

struct JointDesignQualityCandidate final {
  CandidateObjectiveVector objective;
  std::optional<ArtifactRootReference> evidence;
  JointDesignQualityProvenance provenance{};
};

using JointDesignQualityAcquisition =
    std::variant<std::vector<JointDesignQualityCandidate>,
                 IncompleteJointDesignQuality>;

using JointDesignQualityAcquirer =
    std::function<llvm::Expected<JointDesignQualityAcquisition>(
        const JointDesignExecution &, std::uint64_t planOrdinal)>;

using JointHardwarePromotionQualityAcquirer =
    std::function<llvm::Expected<JointDesignQualityAcquisition>(
        const JointDesignExplorationPlan &, std::uint64_t planOrdinal)>;

enum class JointDesignQualityProvenanceDomain : std::uint8_t {
  ObjectiveOnly,
  ApplicationRuntime,
};

/// In-process, pre-Mapping objective used only to rank which bounded hardware
/// parents may consume additional exact Mapping/PnR work. Candidate identity
/// and Mapping legality remain owned by the plan and Mapping providers.
struct JointHardwarePromotionQualityPolicy final {
  std::shared_ptr<const ObjectiveProgram> objectiveProgram;
  std::vector<std::string> objectiveDimensionLabels;
  std::uint32_t totalOrdering = 0;
  JointHardwarePromotionQualityAcquirer acquire;
};

/// Invocation-local adapter to the shared Objective/Pareto owner. The
/// The acquirer is invoked once per selected SystemMapping (the invocation
/// summary temporarily names that mapping) and must return exactly one
/// complete application ObjectiveVector for it. It may not substitute
/// pre-PnR feasibility scores for completed application QoR.
struct JointBoundedQualityPolicy final {
  std::shared_ptr<const ObjectiveProgram> objectiveProgram;
  /// Invocation-local labels for the objective vector. They are provenance,
  /// not a second objective definition; the ObjectiveProgram remains the
  /// ordering authority.
  std::vector<std::string> objectiveDimensionLabels;
  JointDesignQualityProvenanceDomain provenanceDomain =
      JointDesignQualityProvenanceDomain::ObjectiveOnly;
  std::vector<std::uint32_t> paretoDimensions;
  std::uint32_t finalTotalOrdering = 0;
  JointDesignQualityAcquirer acquire;
  std::optional<JointHardwarePromotionQualityPolicy> hardwarePromotion;
  /// Immutable ranking inputs included in every Mapping and hardware-reopen
  /// invocation closure. The referenced owner artifacts remain authoritative.
  std::vector<ArtifactRootReference> semanticInputs;
  /// Maximum verified base mappings promoted to hardware-spectrum expansion
  /// after the bounded software frontier has completed. Base application QoR
  /// and final selection remain owned by this policy; zero is invalid.
  std::uint64_t maximumHardwareSpectrumParents = 1;
  /// Maximum monotonic child probes within one promoted hardware parent.
  /// These probes close typed feedback and are not additional parent
  /// alternatives. Zero is invalid.
  std::uint64_t maximumHardwareRepairProbes = 16;
};

/// Validates facts whose shape is owned by the selected provenance domain.
/// Objective code reproduction and Evidence existence remain the callers'
/// responsibility because they require their respective semantic owners.
llvm::Error validateJointDesignQualityProvenanceDomain(
    const JointBoundedQualityPolicy &policy,
    const JointDesignQualityProvenance &provenance, bool objectiveComplete);
llvm::Error validateJointDesignQualityObjective(
    const ObjectiveProgram &program,
    const JointDesignQualityProvenance &provenance,
    llvm::ArrayRef<std::uint64_t> objectiveCodes);

/// Executes or resumes the exact plan through the shared Journal and
/// scheduler. Missing Mapping support remains the underlying typed incomplete
/// plan outcome; this layer never substitutes an estimate.
llvm::Expected<JointDesignExecution> executeJointDesignExploration(
    const JointDesignExplorationPlan &plan, const DseRunClosure &closure,
    ExecutionJournal &journal, SiteScheduler &scheduler,
    const PlanExecutionPolicy &executionPolicy,
    const ArtifactStore &artifactStore, const BlobStore &blobStore);

/// One workload member's already gate-qualified Mapping set. The existing
/// Promotion contract remains the sole correctness and acceleration gate
/// authority.
struct JointMemberPromotion final {
  ArtifactRootReference software;
  CompletedSelection promotion;
};

struct JointEligibleSystem final {
  ArtifactRootReference system;
  std::vector<ArtifactRootReference> acceptedMappings;
};

struct JointSystemMissingMember final {
  ArtifactRootReference system;
  ArtifactRootReference member;
};

struct JointSystemUnusedAccCore final {
  ArtifactRootReference system;
  fabric::AccCoreOccurrenceRef accCore;
};

using JointSystemGateOutcome =
    std::variant<JointEligibleSystem, JointSystemMissingMember,
                 JointSystemUnusedAccCore>;

struct JointDesignSelection final {
  std::vector<ArtifactRootReference> selectedSystems;
  std::vector<ArtifactRootReference> acceptedMappings;
  std::vector<ArtifactRootReference> satisfiedEvidence;
  std::vector<JointSystemGateOutcome> systemOutcomes;
};

struct JointDesignNoFeasibleSystem final {
  std::vector<ArtifactRootReference> satisfiedEvidence;
  std::vector<JointSystemGateOutcome> systemOutcomes;
};

using JointDesignSelectionOutcome =
    std::variant<JointDesignSelection, JointDesignNoFeasibleSystem>;

/// Applies member-local Promotion results and AccCore-use coverage before the
/// existing aggregate candidate selection policy.
llvm::Expected<JointDesignSelectionOutcome> selectJointDesignSystems(
    llvm::ArrayRef<ArtifactRootReference> systems,
    llvm::ArrayRef<JointMemberPromotion> memberPromotions,
    llvm::ArrayRef<CandidateObjectiveVector> systemObjectives,
    const CandidateSelectionPolicy &selection,
    const ObjectiveProgram *objectiveProgram,
    const ArtifactStore &artifactStore);

} // namespace loom::dse

#endif // LOOM_DSE_JOINTDESIGNEXPLORATION_H
