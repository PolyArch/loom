#include "DSE/JointDesignPolicy.h"

#include "llvm/Support/Error.h"

#include <system_error>

namespace loom::dse {

llvm::Expected<JointDesignPolicy>
JointDesignPolicy::get(std::uint64_t maximumSoftwareFrontier,
                       std::uint64_t maximumSystemFrontier,
                       std::uint64_t maximumPairEvaluations,
                       std::uint64_t maximumSpatialMappingsPerPair) {
  if (maximumSoftwareFrontier == 0 || maximumSystemFrontier == 0 ||
      maximumPairEvaluations == 0 || maximumSpatialMappingsPerPair == 0)
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "joint_design_policy_invalid: every finite bound must be positive");
  return JointDesignPolicy(maximumSoftwareFrontier, maximumSystemFrontier,
                           maximumPairEvaluations,
                           maximumSpatialMappingsPerPair);
}

} // namespace loom::dse
