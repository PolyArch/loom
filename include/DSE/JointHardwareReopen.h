#ifndef LOOM_DSE_JOINTHARDWAREREOPEN_H
#define LOOM_DSE_JOINTHARDWAREREOPEN_H

#include "DSE/HardwareDecision.h"
#include "DSE/JointDesignExploration.h"
#include "DSE/JointMappingMigration.h"
#include "DSE/PreMappingFrontier.h"
#include "DSE/SpatialRuntimeFeedback.h"
#include "DSE/SpatialTransportCegar.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "PnR/SpatialExactRepair.h"
#include "PnR/SpatialMappingWarmSeed.h"
#include "PnR/SpatialProgressState.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
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

/// Declares whether one joint Mapping invocation may materialize hardware
/// children beyond its exact input System frontier. This is search policy,
/// not hardware identity or Mapping legality.
enum class JointHardwareExplorationScope : std::uint8_t {
  FixedSystemFrontier,
  BoundedHardwareReopen,
};

struct JointSpatialFifoHardwareRepair final {
  SpatialFifoRuntimeFeedback feedback;
  std::vector<ArtifactRootReference> childSystems;
  std::vector<JointDesignExecution> executions;
  std::vector<JointMappingReuseDisposition> reuseDispositions;
  bool bypassAlternativeUnsupported = false;
  std::uint64_t candidateLimit = 0;
  std::uint64_t candidatesPlanned = 0;
  std::uint64_t candidatesReserved = 0;
  std::uint64_t candidatesConsumed = 0;
  std::uint64_t candidatesRejected = 0;
  std::uint64_t candidatesCancelled = 0;
};

struct JointSpatialOperandBufferHardwareRepair final {
  SpatialOperandQueueRuntimeFeedback feedback;
  std::vector<ArtifactRootReference> childSystems;
  std::vector<JointDesignExecution> executions;
  std::vector<JointMappingReuseDisposition> reuseDispositions;
  std::uint64_t candidateLimit = 0;
  std::uint64_t candidatesPlanned = 0;
  std::uint64_t candidatesReserved = 0;
  std::uint64_t candidatesConsumed = 0;
  std::uint64_t candidatesRejected = 0;
  std::uint64_t candidatesCancelled = 0;
};

struct JointSpatialTransportMappingRepair final {
  SpatialTransportRuntimeFeedback feedback;
  std::optional<SpatialTransportCegarResult> cegar;
  std::optional<pnr::SpatialMappingWarmSeedAccounting> warmSeedAccounting;
  std::optional<pnr::SpatialExactRepairResult> exactRepair;
  std::vector<ArtifactRootReference> constraintSets;
  std::vector<ArtifactRootReference> repairedSpatialMappings;
  std::vector<ArtifactRootReference> childSystems;
  std::vector<JointDesignExecution> executions;
  std::vector<JointMappingReuseDisposition> reuseDispositions;
  std::uint64_t candidateLimit = 0;
  std::uint64_t candidatesPlanned = 0;
  std::uint64_t candidatesReserved = 0;
  std::uint64_t candidatesConsumed = 0;
  std::uint64_t candidatesRejected = 0;
  std::uint64_t candidatesCancelled = 0;
  bool preparedSeedHandoff = false;
  std::uint64_t coldFallbackCount = 0;
};

struct JointHardwareReopenRequest final {
  JointHardwareReopenRequest(
      DseProducerSemanticBuildIdentity producer, std::string journalRoot,
      std::vector<ArtifactRootReference> evidence,
      JointDesignStoppingPolicy stoppingPolicy,
      std::optional<JointBoundedQualityPolicy> boundedQuality,
      std::optional<std::uint64_t> maximumUsefulAccCoreCount,
      SiteCapacity siteCapacity, PlanExecutionPolicy executionPolicy,
      PreMappingSpectrumEndpoint spectrumEndpoint =
          PreMappingSpectrumEndpoint::Automatic,
      JointHardwareExplorationScope hardwareExplorationScope =
          JointHardwareExplorationScope::BoundedHardwareReopen,
      std::vector<ArtifactRootReference> invocationSemanticInputs = {})
      : producer(std::move(producer)), journalRoot(std::move(journalRoot)),
        evidence(std::move(evidence)), stoppingPolicy(stoppingPolicy),
        boundedQuality(std::move(boundedQuality)),
        maximumUsefulAccCoreCount(maximumUsefulAccCoreCount),
        siteCapacity(std::move(siteCapacity)),
        executionPolicy(std::move(executionPolicy)),
        spectrumEndpoint(spectrumEndpoint),
        hardwareExplorationScope(hardwareExplorationScope),
        invocationSemanticInputs(std::move(invocationSemanticInputs)) {}

  DseProducerSemanticBuildIdentity producer;
  std::string journalRoot;
  std::vector<ArtifactRootReference> evidence;
  JointDesignStoppingPolicy stoppingPolicy =
      JointDesignStoppingPolicy::FirstVerified;
  std::optional<JointBoundedQualityPolicy> boundedQuality;
  /// Sound Dataflow logical-domain upper bound for the invocation's software
  /// frontier. AddAccCore spectrum points beyond it cannot improve useful
  /// parallelism; other typed hardware feedback remains unaffected.
  std::optional<std::uint64_t> maximumUsefulAccCoreCount;
  SiteCapacity siteCapacity;
  PlanExecutionPolicy executionPolicy;
  /// Endpoint focus is ranking provenance only. It may prioritize an exact
  /// feedback parent for a bounded repair, but never supplies endpoint
  /// legality or a Spectrum label.
  PreMappingSpectrumEndpoint spectrumEndpoint =
      PreMappingSpectrumEndpoint::Automatic;
  JointHardwareExplorationScope hardwareExplorationScope =
      JointHardwareExplorationScope::BoundedHardwareReopen;
  /// Invocation-level immutable inputs consumed by ranking or stopping policy
  /// but not owned by the Mapping plan. They join every parent and repair
  /// closure and remain ordinary semantic inputs rather than Evidence.
  std::vector<ArtifactRootReference> invocationSemanticInputs;
  /// Executes an additional unseeded Mapping plan beside the preserve-first
  /// plan so a caller can compare cold and incremental repair as independent
  /// oracles. The cold result is never the repaired Mapping: when the rebase
  /// preserves nothing the preserve-first plan is already unseeded, so the two
  /// plans are identical and the work is paid twice. Callers that only need the
  /// repaired Mapping must leave this disabled.
  bool coldComparisonBaseline = false;
  /// Mapping-repair admission: the cumulative CEGAR children one exact runtime
  /// witness may spend on the immutable parent System. It is a separate owner
  /// from the bounded-quality hardware probe bound, so a non-retiring repair
  /// sequence can never consume the hardware reopen the same witness admits.
  std::uint64_t maximumMappingRepairCandidates = 8;
};

/// Returns `policy` with its dispatch deadline moved earlier by
/// `reservedNanoseconds`, never later and never before the present. A policy
/// without a deadline has no window to reserve and is returned unchanged.
llvm::Expected<PlanExecutionPolicy>
reserveDispatchWindow(const PlanExecutionPolicy &policy,
                      std::uint64_t reservedNanoseconds);

struct JointRepairQualitySelection final {
  std::size_t executionOrdinal = 0;
  ArtifactRootReference mapping;
};

struct JointRepairQualityIncomplete final {
  std::size_t executionOrdinal = 0;
  IncompleteJointDesignQuality incomplete;
};

using JointRepairQualitySelectionOutcome =
    std::variant<JointRepairQualitySelection, JointRepairQualityIncomplete>;

/// Selects one already quality-assessed repair Mapping through the same
/// ObjectiveProgram, Pareto dimensions, and total ordering as the parent
/// bounded application request. A typed incomplete result blocks selection;
/// it is never treated as an inferior objective.
llvm::Expected<JointRepairQualitySelectionOutcome>
selectJointRepairMappingByQuality(
    llvm::ArrayRef<JointDesignExecution> executions,
    const JointBoundedQualityPolicy &quality, const ArtifactStore &artifacts);

llvm::Expected<std::uint64_t>
deriveApplicationRuntimeResourceCoreCost(const JointDesignExecution &execution,
                                         const ArtifactRootReference &mapping,
                                         const ArtifactStore &artifacts);

struct JointResourceTimeAdjacentRepair final {
  ArtifactRootReference parentMapping;
  ArtifactRootReference migrationSeed;
  JointDesignExplorationPlan plan;
  std::optional<ArtifactRootReference> coldMapping;
  std::optional<ArtifactRootReference> incrementalMapping;
  JointDesignExecution coldExecution;
  JointDesignExecution execution;
  JointMappingReuseDisposition reuseDisposition =
      JointMappingReuseDisposition::ColdFallback;
  mapping::SystemMappingImportSessionStatistics coldVerification;
  mapping::SystemMappingImportSessionStatistics incrementalVerification;
};

/// One already-materialized hardware child and the exact lineage needed to
/// project its parent Mapping frontier. Candidate generators remain the sole
/// owners of child identity and impact; this value only joins their outputs to
/// the shared repair executor.
struct JointHardwareMutationChild final {
  ArtifactRootReference system;
  ResolvedConfig config;
  std::optional<pnr::SystemExecutionBindingCorrespondence>
      executionBindingCorrespondence;
  /// Ordered candidate lineage. A single entry is eligible for typed local
  /// repair. Multiple entries retain the exact component changes but use a
  /// conservative cold fallback until a composed parent-to-child entity
  /// correspondence is available.
  std::vector<HardwareImpactProjection> impacts;
};

/// Materializes one exact Module rewrite through the canonical candidate
/// generator, replaces every matching SpatialCore attachment in the parent
/// System, and returns the composed System correspondence and typed impact.
llvm::Expected<JointHardwareMutationChild>
materializeJointModuleHardwareMutation(
    ResolvedConfig config, const ArtifactRootReference &parentSystem,
    const ArtifactRootReference &parentModule,
    SpatialMicroarchitectureDecisionDomain decision,
    const ArtifactStore &artifacts, const BlobStore &blobs);

/// Materializes one exact Module topology rewrite through the canonical
/// topology candidate generator and the same System replacement path. The
/// adopted lineage decision owns the `SpatialTopology` impact family.
llvm::Expected<JointHardwareMutationChild>
materializeJointModuleHardwareMutation(
    ResolvedConfig config, const ArtifactRootReference &parentSystem,
    const ArtifactRootReference &parentModule,
    SpatialTopologyDecisionDomain decision, const ArtifactStore &artifacts,
    const BlobStore &blobs);

/// Materializes one exact System rewrite through the canonical composition
/// candidate generator. Module lineage is derived from the finalized parent
/// and child AccCore targets. Non-bijective attachment lineage retains the
/// legal child but deliberately omits incremental System correspondence.
llvm::Expected<JointHardwareMutationChild>
materializeJointSystemHardwareMutation(
    ResolvedConfig config, const ArtifactRootReference &parentSystem,
    SystemCompositionDecisionDomain decision,
    llvm::ArrayRef<ArtifactRootReference> admissibleModules,
    const ArtifactStore &artifacts, const BlobStore &blobs);

/// Composes two consecutive exact child lineages. Component impacts remain
/// ordered and lossless; the repair executor conservatively uses cold fallback
/// for the combined Mapping until a composed local entity cone is available.
llvm::Expected<JointHardwareMutationChild>
composeJointHardwareMutationChildren(JointHardwareMutationChild first,
                                     JointHardwareMutationChild second,
                                     const ArtifactStore &artifacts);

enum class JointSystemMappingReuseDisposition : std::uint8_t {
  Preserved,
  Reopened,
  ColdFallback,
};

llvm::StringRef jointSystemMappingReuseDispositionSpelling(
    JointSystemMappingReuseDisposition disposition);

/// Paired proof for one typed hardware mutation. The cold and preserve-first
/// plans execute in independent journals and PnR sessions against the same
/// child. Every returned Mapping has also passed a fresh strict import after
/// provider execution; import accounting is kept per side so memory and work
/// are not inferred from plan counts.
struct JointHardwareMutationRepair final {
  /// Durable per-family evidence published by the executor
  /// (`loom.dse.hardware_mutation_repair_record`): affected cones, typed
  /// dispositions, cold and preserve-first Mapping roots, dispatch and
  /// verifier accounting, and quality observations.
  ArtifactRootReference record;
  ArtifactRootReference parentMapping;
  JointHardwareMutationChild child;
  JointMappingRebaseResult rebase;
  JointSystemMappingReuseDisposition systemDisposition =
      JointSystemMappingReuseDisposition::ColdFallback;
  JointDesignExplorationPlan coldPlan;
  JointDesignExplorationPlan incrementalPlan;
  /// Empty unless `JointHardwareReopenRequest::coldComparisonBaseline` asked
  /// for the independent cold oracle. The repaired Mapping is always the
  /// preserve-first result.
  std::vector<ArtifactRootReference> coldMappings;
  std::vector<ArtifactRootReference> incrementalMappings;
  std::optional<JointDesignExecution> coldExecution;
  JointDesignExecution incrementalExecution;
  mapping::SystemMappingImportSessionStatistics coldVerification;
  mapping::SystemMappingImportSessionStatistics incrementalVerification;
};

/// Executes ordinary Tech, Spatial, and System Mapping twice for one exact
/// hardware child: once from a cold plan and once with the typed parent
/// preference. Global impacts intentionally produce a cold-fallback
/// incremental plan. This compiler-side API never runs from runtime and does
/// not create Deployment or resource-time transition identities.
llvm::Expected<JointHardwareMutationRepair> executeJointHardwareMutationRepair(
    const JointDesignExplorationPlan &parentPlan,
    const JointDesignExecution &parentExecution,
    const JointDesignPolicy &policy, const ArtifactRootReference &parentMapping,
    JointHardwareMutationChild child, JointHardwareReopenRequest request,
    const ArtifactStore &artifacts, const BlobStore &blobs);

/// Executes one already-promoted adjacent resource-time state on the same
/// immutable System. It executes one independent cold Mapping and one
/// preserve-first Mapping for the same child partitions. Tech and Spatial
/// frontiers are retained only by the latter, while the typed Dataflow root
/// delta is bound to the existing System preserve-first initializer. This
/// function does not construct a ResourceTimeTransition or claim a safe point,
/// Deployment delta, migration cost, or endpoint class.
llvm::Expected<JointResourceTimeAdjacentRepair>
executeResourceTimeAdjacentMappingRepair(
    const JointDesignExplorationPlan &parentPlan,
    const JointDesignExecution &parentExecution,
    const JointDesignPolicy &policy,
    llvm::ArrayRef<pnr::SystemBindingPartitionIntent> childPartitions,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> reopenedRoots,
    JointHardwareReopenRequest request, const ArtifactStore &artifacts,
    const BlobStore &blobs);

/// Materializes and maps the bounded minimal FIFO child set admitted by one
/// exact runtime witness. A typed negative feedback value returns no child and
/// performs no Mapping work. Every child uses the ordinary hardware decision,
/// impact, preserve/cold-fallback, System PnR, and independent verifier path.
llvm::Expected<JointSpatialFifoHardwareRepair>
executeSpatialFifoHardwareFeedbackReopen(
    const JointDesignExplorationPlan &parentPlan,
    const JointDesignExecution &parentExecution,
    const JointDesignPolicy &policy, const SpatialFifoRuntimeFeedback &feedback,
    JointHardwareReopenRequest request, const ArtifactStore &artifacts,
    const BlobStore &blobs);

/// Projects the canonical global-depth comparison domain for one static
/// shared-pool suggestion: depth one, depth two, the sufficient bound, and
/// one deeper control. The returned typed domain is consumed by the ordinary
/// spatial-microarchitecture generator; this adapter owns no Fabric writer.
llvm::Expected<std::vector<SpatialMicroarchitectureDecisionDomain>>
deriveSpatialCapacityHardwareReopenDomains(
    const pnr::SpatialFifoCapacitySuggestion &feedback);

/// Materializes the bounded Temporal operand-buffer child set admitted by one
/// exact queue-level closed-wait witness. Incomplete, ambiguous, or analytic
/// feedback returns no child and never enters Mapping.
llvm::Expected<JointSpatialOperandBufferHardwareRepair>
executeSpatialOperandBufferHardwareFeedbackReopen(
    const JointDesignExplorationPlan &parentPlan,
    const JointDesignExecution &parentExecution,
    const JointDesignPolicy &policy,
    const SpatialOperandQueueRuntimeFeedback &feedback,
    JointHardwareReopenRequest request, const ArtifactStore &artifacts,
    const BlobStore &blobs);

/// Reopens one exact route at a time on the immutable parent System. Each
/// candidate excludes one Mapping-verified traversal from a closed storage
/// wait and executes the ordinary Spatial/System providers. Finalized Spatial
/// state cannot yet seed the mutable router, so this path reports an explicit
/// constrained cold fallback rather than claiming incremental preservation.
llvm::Expected<JointSpatialTransportMappingRepair>
executeSpatialTransportRuntimeRepair(
    const JointDesignExplorationPlan &parentPlan,
    const JointDesignExecution &parentExecution,
    const JointDesignPolicy &policy,
    const SpatialTransportRuntimeFeedback &feedback,
    JointHardwareReopenRequest request, const ArtifactStore &artifacts,
    const BlobStore &blobs);

/// The typed runtime witnesses one failed application replay derived for its
/// selected parent Mapping. A family whose feedback is absent or not Exact
/// plans no work.
struct JointRuntimeWitnessSet final {
  std::optional<SpatialTransportRuntimeFeedback> transport;
  std::optional<SpatialFifoRuntimeFeedback> fifo;
  std::optional<SpatialOperandQueueRuntimeFeedback> operandQueue;
};

/// Additive candidate ledger of one repair family. Reserved work settles as
/// consumed, rejected, or cancelled, and planned equals reserved.
struct JointRepairWorkLedger final {
  std::uint64_t candidateLimit = 0;
  std::uint64_t planned = 0;
  std::uint64_t reserved = 0;
  std::uint64_t consumed = 0;
  std::uint64_t rejected = 0;
  std::uint64_t cancelled = 0;
};

/// The two repair families one witness set admits, each with its own ledger.
/// Mapping repair rebuilds the Mapping on the immutable parent System;
/// hardware reopen materializes typed System children. `childSystems` is
/// aligned with `executions` across both families.
struct JointRuntimeWitnessRepair final {
  std::optional<JointSpatialTransportMappingRepair> mappingRepair;
  std::optional<JointSpatialFifoHardwareRepair> fifoReopen;
  std::optional<JointSpatialOperandBufferHardwareRepair> operandBufferReopen;
  std::vector<ArtifactRootReference> childSystems;
  std::vector<JointDesignExecution> executions;
  JointRepairWorkLedger mappingRepairLedger;
  JointRepairWorkLedger hardwareReopenLedger;
  /// Wall-clock window withheld from Mapping repair for the admitted hardware
  /// children: the parent's own measured cost per child.
  std::uint64_t hardwareReopenReservedNanoseconds = 0;
};

/// Executes the runtime-witness repair families of one failed replay in
/// order: Mapping repair on the immutable parent System first, then the
/// hardware reopen the witness admits. The families never share a budget.
/// Mapping repair spends at most `request.maximumMappingRepairCandidates` and
/// dispatches only inside the invocation window minus one parent cost
/// (`parentCostNanoseconds`, the parent's measured Mapping and runtime
/// validation wall time) per admitted hardware child. Hardware children spend
/// only `remainingHardwareRepairProbes` (absent means unbounded) and are never
/// materialized under a fixed System frontier.
llvm::Expected<JointRuntimeWitnessRepair> executeJointRuntimeWitnessRepair(
    const JointDesignExplorationPlan &parentPlan,
    const JointDesignExecution &parentExecution,
    const JointDesignPolicy &policy, const JointRuntimeWitnessSet &witnesses,
    std::uint64_t parentCostNanoseconds,
    std::optional<std::uint64_t> remainingHardwareRepairProbes,
    JointHardwareReopenRequest request, const ArtifactStore &artifacts,
    const BlobStore &blobs);

/// Executes bounded software/System pairs before consuming typed Mapping
/// feedback. Builtin hardware growth is rematerialized from its exact recipe;
/// Mapping and the final independent verifiers remain the legality authority.
llvm::Expected<JointDesignExecution> executeJointDesignWithHardwareReopen(
    llvm::ArrayRef<const JointDesignExplorationPlan *> plans,
    const JointDesignPolicy &policy, JointHardwareReopenRequest request,
    const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::dse

#endif
