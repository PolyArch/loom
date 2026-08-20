#ifndef LOOM_DSE_JOINTHARDWAREREOPEN_H
#define LOOM_DSE_JOINTHARDWAREREOPEN_H

#include "DSE/JointDesignExploration.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <string>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::dse {

struct JointHardwareReopenRequest final {
  DseProducerSemanticBuildIdentity producer;
  std::string journalRoot;
  std::vector<ArtifactRootReference> evidence;
  SiteCapacity siteCapacity;
  PlanExecutionPolicy executionPolicy;
};

/// Executes bounded software/System pairs before consuming typed Mapping
/// feedback. Builtin hardware growth is rematerialized from its exact recipe;
/// Mapping and the final independent verifiers remain the legality authority.
llvm::Expected<JointDesignExecution> executeJointDesignWithHardwareReopen(
    llvm::ArrayRef<const JointDesignExplorationPlan *> plans,
    const JointDesignPolicy &policy, JointHardwareReopenRequest request,
    const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::dse

#endif
