#ifndef LOOM_DSE_JOINTHARDWAREREOPEN_H
#define LOOM_DSE_JOINTHARDWAREREOPEN_H

#include "DSE/JointDesignExploration.h"
#include "DSE/JointMappingMigration.h"
#include "DSE/SpatialRuntimeFeedback.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <string>
#include <optional>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::dse {

struct JointSpatialFifoHardwareRepair final {
  SpatialFifoRuntimeFeedback feedback;
  std::vector<ArtifactRootReference> childSystems;
  std::vector<JointDesignExecution> executions;
  std::vector<JointMappingReuseDisposition> reuseDispositions;
  bool bypassAlternativeUnsupported = false;
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
};

struct JointResourceTimeAdjacentRepair final {
  ArtifactRootReference parentMapping;
  ArtifactRootReference migrationSeed;
  JointDesignExplorationPlan plan;
  JointDesignExecution execution;
};

/// Executes one already-promoted adjacent resource-time state on the same
/// immutable System. Tech and Spatial frontiers are retained, while the typed
/// Dataflow root delta is bound to the existing System preserve-first
/// initializer. This function does not construct a ResourceTimeTransition or
/// claim a safe point, Deployment delta, migration cost, or endpoint class.
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
    const JointDesignPolicy &policy,
    const SpatialFifoRuntimeFeedback &feedback,
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
