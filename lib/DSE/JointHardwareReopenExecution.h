#ifndef LOOM_LIB_DSE_JOINTHARDWAREREOPENEXECUTION_H
#define LOOM_LIB_DSE_JOINTHARDWAREREOPENEXECUTION_H

#include "DSE/JointHardwareReopen.h"

namespace loom::dse {

llvm::Expected<JointDesignInvocationManifestReference>
publishJointPlanInvocationManifest(
    DseRunClosure closure, const ResolvedConfig &config,
    const DsePlanGenerateInvocationRecords &generateRecords,
    InvocationControllerOutcome outcome, ExecutionJournal &journal,
    const ArtifactStore &artifacts, const BlobStore &blobs);

llvm::Error bindJointDesignInvocationManifest(
    JointDesignExecution &execution,
    JointDesignInvocationManifestReference invocationManifest);

llvm::Error appendJointDesignSupportingInvocationManifest(
    JointDesignExecution &execution,
    JointDesignInvocationManifestReference invocationManifest);

llvm::Error retainJointDesignInvocationManifest(
    std::vector<JointDesignInvocationManifestReference> &retained,
    const JointDesignInvocationManifestReference &invocationManifest);

llvm::Error retainJointDesignExecutionInvocations(
    std::vector<JointDesignInvocationManifestReference> &retained,
    const JointDesignExecution &execution);

llvm::Error attachJointDesignSupportingInvocationManifests(
    JointDesignExecution &execution,
    llvm::ArrayRef<JointDesignInvocationManifestReference> retained);

llvm::Expected<JointDesignExecution>
executeJointPlan(const JointDesignExplorationPlan &plan,
                 llvm::ArrayRef<ArtifactRootReference> evidence,
                 const JointHardwareReopenRequest &request,
                 SiteScheduler &scheduler, const ArtifactStore &artifacts,
                 const BlobStore &blobs,
                 const PlanExecutionPolicy *executionPolicy = nullptr);

/// Executes one already-bounded repair plan through the same stopping and
/// quality owner as the parent application request. Bounded quality remains
/// fixed to this System frontier; the repair caller owns any further typed
/// hardware feedback domain.
llvm::Expected<JointDesignExecution>
executeJointRepairPlan(const JointDesignExplorationPlan &plan,
                       const JointDesignPolicy &policy,
                       JointHardwareReopenRequest request,
                       const ArtifactStore &artifacts, const BlobStore &blobs);

llvm::Expected<std::vector<ArtifactRootReference>>
normalizedTimingProfiles(const ArtifactRootReference &system,
                         const ArtifactStore &artifacts);

} // namespace loom::dse

#endif // LOOM_LIB_DSE_JOINTHARDWAREREOPENEXECUTION_H
