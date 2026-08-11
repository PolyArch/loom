#ifndef LOOM_DSE_JOINTDESIGNPOLICY_H
#define LOOM_DSE_JOINTDESIGNPOLICY_H

#include "Common/Artifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace loom {
class ArtifactStore;
}

namespace loom::dse {

/// Finite work policy for one explicit software/System frontier join. The
/// selected roots remain ordinary Artifacts and every materialized pair is
/// recorded by the resulting DSE plan.
class JointDesignPolicy final {
public:
  static llvm::Expected<JointDesignPolicy>
  get(std::uint64_t maximumSoftwareFrontier,
      std::uint64_t maximumSystemFrontier,
      std::uint64_t maximumPairEvaluations,
      std::uint64_t maximumSpatialMappingsPerPair);

  std::uint64_t maximumSoftwareFrontier() const {
    return maximumSoftwareFrontier_;
  }
  std::uint64_t maximumSystemFrontier() const {
    return maximumSystemFrontier_;
  }
  std::uint64_t maximumPairEvaluations() const {
    return maximumPairEvaluations_;
  }
  std::uint64_t maximumSpatialMappingsPerPair() const {
    return maximumSpatialMappingsPerPair_;
  }

private:
  JointDesignPolicy(std::uint64_t maximumSoftwareFrontier,
                    std::uint64_t maximumSystemFrontier,
                    std::uint64_t maximumPairEvaluations,
                    std::uint64_t maximumSpatialMappingsPerPair)
      : maximumSoftwareFrontier_(maximumSoftwareFrontier),
        maximumSystemFrontier_(maximumSystemFrontier),
        maximumPairEvaluations_(maximumPairEvaluations),
        maximumSpatialMappingsPerPair_(maximumSpatialMappingsPerPair) {}

  std::uint64_t maximumSoftwareFrontier_ = 0;
  std::uint64_t maximumSystemFrontier_ = 0;
  std::uint64_t maximumPairEvaluations_ = 0;
  std::uint64_t maximumSpatialMappingsPerPair_ = 0;
};

struct JointDesignInputs final {
  std::vector<ArtifactRootReference> softwareFrontier;
  std::vector<ArtifactRootReference> systemFrontier;
};

struct JointDesignPair final {
  ArtifactRootReference software;
  ArtifactRootReference system;

  friend bool operator==(const JointDesignPair &lhs,
                         const JointDesignPair &rhs) {
    return lhs.software == rhs.software && lhs.system == rhs.system;
  }
};

/// Transient declaration of the exact cross-pairs admitted by one bounded
/// join. `eligiblePairCount` records the complete product size before the
/// declared pair bound; a truncated result is never a global optimum claim.
struct BoundedJointFrontier final {
  std::vector<ArtifactRootReference> softwareFrontier;
  std::vector<ArtifactRootReference> systemFrontier;
  std::vector<JointDesignPair> pairs;
  std::uint64_t eligiblePairCount = 0;
  bool truncated = false;
};

llvm::Expected<BoundedJointFrontier>
buildBoundedJointFrontier(JointDesignInputs inputs,
                          const JointDesignPolicy &policy,
                          const ArtifactStore &artifactStore);

} // namespace loom::dse

#endif // LOOM_DSE_JOINTDESIGNPOLICY_H
