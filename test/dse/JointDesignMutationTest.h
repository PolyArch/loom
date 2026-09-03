#ifndef LOOM_TEST_DSE_JOINTDESIGNMUTATIONTEST_H
#define LOOM_TEST_DSE_JOINTDESIGNMUTATIONTEST_H

#include "Common/Artifact.h"

#include "llvm/ADT/StringRef.h"

namespace loom {
class ArtifactStore;
class BlobStore;
struct ResolvedConfig;
namespace dse {
class JointDesignPolicy;
struct JointDesignExecution;
struct JointDesignExplorationPlan;
} // namespace dse
} // namespace loom

namespace loom::dse::joint_test {

inline constexpr llvm::StringLiteral allJointDesignTestSections = "*";

void exerciseJointDesignMutationFamilies(
    llvm::StringRef mutationFamily, llvm::StringRef temporaryPath,
    const JointDesignExplorationPlan &plan,
    const JointDesignExecution &parentExecution,
    const JointDesignPolicy &policy, const ArtifactRootReference &parentMapping,
    const ResolvedConfig &config, const ArtifactRootReference &system,
    const ArtifactStore &store, const BlobStore &blobs);

} // namespace loom::dse::joint_test

#endif // LOOM_TEST_DSE_JOINTDESIGNMUTATIONTEST_H
