#ifndef LOOM_FABRIC_ARTIFACT_FABRICMEMORYSERVICECLOSURE_H
#define LOOM_FABRIC_ARTIFACT_FABRICMEMORYSERVICECLOSURE_H

#include "Fabric/Artifact/FabricSystemRootView.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::fabric {

struct FabricMemoryServiceTargetBranch final {
  std::vector<SystemServiceTransformRef> transformPath;
  FabricMemoryServiceRegionRef region;

  friend bool operator==(const FabricMemoryServiceTargetBranch &left,
                         const FabricMemoryServiceTargetBranch &right) {
    return left.transformPath == right.transformPath &&
           left.region == right.region;
  }
};

struct FabricMemoryServiceTargetPlan final {
  std::vector<FabricMemoryServiceTargetBranch> branches;

  friend bool operator==(const FabricMemoryServiceTargetPlan &left,
                         const FabricMemoryServiceTargetPlan &right) {
    return left.branches == right.branches;
  }
};

struct FabricMemoryServiceSourceInterval final {
  std::uint64_t addressBaseBytes = 0;
  std::uint64_t sizeBytes = 0;
};

/// Projects the finite complete terminal-plan domain rooted at one exact
/// System service endpoint. The result is derived only from Fabric-owned
/// memory connections, transform endpoint relations, and service regions.
llvm::Expected<std::vector<FabricMemoryServiceTargetPlan>>
projectFabricMemoryServiceTargetPlans(const FabricSystemRootView &system,
                                      SystemServiceEndpointRef endpoint);

/// Projects only complete plans whose composed Fabric-owned address relation
/// covers the exact source interval once and whose transformed addresses are
/// contained by the selected terminal regions.
llvm::Expected<std::vector<FabricMemoryServiceTargetPlan>>
projectFabricMemoryServiceTargetPlans(
    const FabricSystemRootView &system, SystemServiceEndpointRef endpoint,
    FabricMemoryServiceSourceInterval sourceInterval);

} // namespace loom::fabric

#endif // LOOM_FABRIC_ARTIFACT_FABRICMEMORYSERVICECLOSURE_H
