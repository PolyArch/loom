#ifndef LOOM_DSE_JOINTHARDWAREREOPEN_H
#define LOOM_DSE_JOINTHARDWAREREOPEN_H

#include "DSE/JointDesignExploration.h"
#include "DSE/JointMappingMigration.h"
#include "DSE/PreMappingFrontier.h"
#include "DSE/SpatialRuntimeFeedback.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>
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
};

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
