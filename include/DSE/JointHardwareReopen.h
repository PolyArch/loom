#ifndef LOOM_DSE_JOINTHARDWAREREOPEN_H
#define LOOM_DSE_JOINTHARDWAREREOPEN_H

#include "DSE/HardwareDecision.h"
#include "DSE/JointDesignExploration.h"
#include "DSE/JointMappingMigration.h"
#include "DSE/PreMappingFrontier.h"
#include "DSE/SpatialRuntimeFeedback.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>
#include <utility>
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
  std::vector<ArtifactRootReference> constraintSets;
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
};

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
  ArtifactRootReference parentMapping;
  JointHardwareMutationChild child;
  JointMappingRebaseResult rebase;
  JointSystemMappingReuseDisposition systemDisposition =
      JointSystemMappingReuseDisposition::ColdFallback;
  JointDesignExplorationPlan coldPlan;
  JointDesignExplorationPlan incrementalPlan;
  std::vector<ArtifactRootReference> coldMappings;
  std::vector<ArtifactRootReference> incrementalMappings;
  JointDesignExecution coldExecution;
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

/// Executes bounded software/System pairs before consuming typed Mapping
/// feedback. Builtin hardware growth is rematerialized from its exact recipe;
/// Mapping and the final independent verifiers remain the legality authority.
llvm::Expected<JointDesignExecution> executeJointDesignWithHardwareReopen(
    llvm::ArrayRef<const JointDesignExplorationPlan *> plans,
    const JointDesignPolicy &policy, JointHardwareReopenRequest request,
    const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::dse

#endif
