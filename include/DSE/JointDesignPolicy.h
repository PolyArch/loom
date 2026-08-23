#ifndef LOOM_DSE_JOINTDESIGNPOLICY_H
#define LOOM_DSE_JOINTDESIGNPOLICY_H

#include "Common/Artifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace loom {
class ArtifactStore;
}

namespace loom::dse {

enum class JointDesignStoppingPolicy : std::uint8_t {
  FirstVerified,
  BoundedQuality,
};

llvm::StringRef
jointDesignStoppingPolicySpelling(JointDesignStoppingPolicy policy);

/// Finite work policy for one explicit software/System frontier join. The
/// selected roots remain ordinary Artifacts and every materialized pair is
/// recorded by the resulting DSE plan.
class JointDesignPolicy final {
public:
  static llvm::Expected<JointDesignPolicy>
  get(std::uint64_t maximumSoftwareFrontier,
      std::uint64_t maximumSystemFrontier,
      std::uint64_t maximumPairEvaluations,
      std::uint64_t maximumTechMappingsPerModule,
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
  std::uint64_t maximumTechMappingsPerModule() const {
    return maximumTechMappingsPerModule_;
  }
  std::uint64_t maximumSpatialMappingsPerPair() const {
    return maximumSpatialMappingsPerPair_;
  }

private:
  JointDesignPolicy(std::uint64_t maximumSoftwareFrontier,
                    std::uint64_t maximumSystemFrontier,
                    std::uint64_t maximumPairEvaluations,
                    std::uint64_t maximumTechMappingsPerModule,
                    std::uint64_t maximumSpatialMappingsPerPair)
      : maximumSoftwareFrontier_(maximumSoftwareFrontier),
        maximumSystemFrontier_(maximumSystemFrontier),
        maximumPairEvaluations_(maximumPairEvaluations),
        maximumTechMappingsPerModule_(maximumTechMappingsPerModule),
        maximumSpatialMappingsPerPair_(maximumSpatialMappingsPerPair) {}

  std::uint64_t maximumSoftwareFrontier_ = 0;
  std::uint64_t maximumSystemFrontier_ = 0;
  std::uint64_t maximumPairEvaluations_ = 0;
  std::uint64_t maximumTechMappingsPerModule_ = 0;
  std::uint64_t maximumSpatialMappingsPerPair_ = 0;
};

struct JointDesignInputs final {
  std::vector<std::vector<ArtifactRootReference>> applicationScopes;
  std::vector<ArtifactRootReference> systemFrontier;
};

struct JointSoftwareScope final {
  ArtifactRootReference dataflow;
  std::vector<ArtifactRootReference> workloads;

  friend bool operator==(const JointSoftwareScope &lhs,
                         const JointSoftwareScope &rhs) {
    return lhs.dataflow == rhs.dataflow && lhs.workloads == rhs.workloads;
  }
};

struct JointDesignPair final {
  JointSoftwareScope software;
  ArtifactRootReference system;

  friend bool operator==(const JointDesignPair &lhs,
                         const JointDesignPair &rhs) {
    return lhs.software == rhs.software && lhs.system == rhs.system;
  }
};

enum class JointPairEstimateConfidence : std::uint8_t {
  LowerBound,
  LowConfidence,
};

/// Invocation-local features used only to rank a bounded pair frontier before
/// any Tech/Spatial/System provider is dispatched. They are not Mapping
/// legality, feasibility, or artifact identity.
struct JointPairAnalyticProjection final {
  std::uint64_t softwareActorCount = 0;
  std::uint64_t softwareGraphCount = 0;
  std::uint64_t softwareMemoryRootCount = 0;
  std::uint64_t systemAccCoreCount = 0;
  std::uint64_t systemTransportResourceCount = 0;
  std::uint64_t minimumExecutionWaves = 0;
  std::uint64_t estimatedWorkUnits = 0;
  JointPairEstimateConfidence confidence =
      JointPairEstimateConfidence::LowConfidence;
};

/// Transient declaration of the exact cross-pairs admitted by one bounded
/// join. `eligiblePairCount` records the complete product size before the
/// declared pair bound; a truncated result is never a global optimum claim.
struct BoundedJointFrontier final {
  std::vector<JointSoftwareScope> softwareFrontier;
  std::vector<ArtifactRootReference> systemFrontier;
  std::vector<JointDesignPair> pairs;
  /// Projections are aligned with `pairs`; they are derived once per retained
  /// pair and are never used as a substitute for Mapping verification.
  std::vector<JointPairAnalyticProjection> pairProjections;
  std::uint64_t eligiblePairCount = 0;
  std::uint64_t analyticEvaluatedPairCount = 0;
  /// Pairs omitted solely because the declared bounded frontier was full.
  /// This is a scheduling decision, never an infeasibility claim.
  std::uint64_t analyticDeferredPairCount = 0;
  bool truncated = false;
};

llvm::StringRef jointPairEstimateConfidenceSpelling(
    JointPairEstimateConfidence confidence);

llvm::Expected<BoundedJointFrontier>
buildBoundedJointFrontier(JointDesignInputs inputs,
                          const JointDesignPolicy &policy,
                          const ArtifactStore &artifactStore);

} // namespace loom::dse

#endif // LOOM_DSE_JOINTDESIGNPOLICY_H
