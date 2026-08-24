#ifndef LOOM_LIB_DSE_JOINTHARDWAREREOPENEXECUTION_H
#define LOOM_LIB_DSE_JOINTHARDWAREREOPENEXECUTION_H

#include "DSE/JointHardwareReopen.h"

namespace loom::dse {

llvm::Expected<JointDesignExecution>
executeJointPlan(const JointDesignExplorationPlan &plan,
                 llvm::ArrayRef<ArtifactRootReference> evidence,
                 const JointHardwareReopenRequest &request,
                 SiteScheduler &scheduler, const ArtifactStore &artifacts,
                 const BlobStore &blobs,
                 const PlanExecutionPolicy *executionPolicy = nullptr);

llvm::Expected<std::vector<ArtifactRootReference>>
normalizedTimingProfiles(const ArtifactRootReference &system,
                         const ArtifactStore &artifacts);

} // namespace loom::dse

#endif // LOOM_LIB_DSE_JOINTHARDWAREREOPENEXECUTION_H
