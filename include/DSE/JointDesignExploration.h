#ifndef LOOM_DSE_JOINTDESIGNEXPLORATION_H
#define LOOM_DSE_JOINTDESIGNEXPLORATION_H

#include "Config/ResolvedConfig.h"
#include "DSE/JointDesignPolicy.h"
#include "DSE/PlanExecutor.h"
#include "DSE/Promotion.h"
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

/// Invocation-local QoR observation for one concrete SystemMapping. A
/// missing objective is explicit typed evidence; it is never represented by a
/// sentinel score.
struct JointDesignQualityObservation final {
  ArtifactRootReference candidate;
  std::vector<std::uint64_t> objectiveCodes;
  std::optional<JointDesignQualityIncompleteReason> incompleteReason;
};

/// Pre-Mapping quality observation for one exact software/System plan. The
/// promoted bit records admission to additional exact Mapping/PnR work; it is
/// never a feasibility claim for the parent or any generated child.
struct JointHardwarePromotionObservation final {
  std::uint64_t planOrdinal = 0;
  ArtifactRootReference system;
  std::vector<std::uint64_t> objectiveCodes;
  std::optional<JointDesignQualityIncompleteReason> incompleteReason;
  bool promotedToExactMapping = false;
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
  /// Exact InvocationManifest run key for this joint execution. It is a
  /// provenance join only; application decisions must not derive identity
  /// from mutable ranking or cache state.
  std::optional<std::array<std::uint8_t, 32>> invocationRunKey;
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

struct JointDesignExecution final {
  DsePlanExecutionResult planExecution;
  std::vector<JointMappedPair> mappedPairs;
  JointDesignExecutionSummary summary;
};

struct IncompleteJointDesignQuality final {
  JointDesignQualityIncompleteReason reason =
      JointDesignQualityIncompleteReason::ProofNotEstablished;
  std::optional<ArtifactRootReference> candidate;
};

using JointDesignQualityAcquisition =
    std::variant<std::vector<CandidateObjectiveVector>,
                 IncompleteJointDesignQuality>;

using JointDesignQualityAcquirer =
    std::function<llvm::Expected<JointDesignQualityAcquisition>(
        const JointDesignExecution &, std::uint64_t planOrdinal)>;

using JointHardwarePromotionQualityAcquirer =
    std::function<llvm::Expected<JointDesignQualityAcquisition>(
        const JointDesignExplorationPlan &, std::uint64_t planOrdinal)>;

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
  std::vector<std::uint32_t> paretoDimensions;
  std::uint32_t finalTotalOrdering = 0;
  JointDesignQualityAcquirer acquire;
  std::optional<JointHardwarePromotionQualityPolicy> hardwarePromotion;
  /// Maximum verified base mappings promoted to hardware-spectrum expansion
  /// after the bounded software frontier has completed. Base application QoR
  /// and final selection remain owned by this policy; zero is invalid.
  std::uint64_t maximumHardwareSpectrumParents = 1;
  /// Maximum monotonic child probes within one promoted hardware parent.
  /// These probes close typed feedback and are not additional parent
  /// alternatives. Zero is invalid.
  std::uint64_t maximumHardwareRepairProbes = 16;
};

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
