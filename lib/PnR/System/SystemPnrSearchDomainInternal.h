#ifndef LOOM_LIB_PNR_SYSTEM_SYSTEMPNRSEARCHDOMAININTERNAL_H
#define LOOM_LIB_PNR_SYSTEM_SYSTEMPNRSEARCHDOMAININTERNAL_H

#include "PnR/System/SystemPnrSearchDomain.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::pnr::detail {

struct CanonicalSystemPartitionBinding final {
  SystemSearchBindingKey key;
  std::vector<::loom::mapping::SystemPresburgerCell> cells;
};

llvm::Expected<std::vector<std::uint8_t>>
canonicalBindingKeyBytes(const SystemSearchBindingKey &key,
                         const ArtifactIdentity &dataflowIdentity);

llvm::Expected<std::vector<::dataflow::RootThreadLaunchRef>>
canonicalRootThreadLaunchSet(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots);

llvm::Expected<std::vector<CanonicalSystemPartitionBinding>>
canonicalizeAndValidateSystemPartition(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots,
    const SystemBindingPartitionPlan &plan);

llvm::Expected<bool> systemPresburgerCellsIntersect(
    const ::loom::mapping::SystemPresburgerCell &lhs,
    const ::loom::mapping::SystemPresburgerCell &rhs);

llvm::Expected<std::vector<SystemSearchServiceDomain>>
projectSystemServiceDomains(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SYSTEM_SYSTEMPNRSEARCHDOMAININTERNAL_H
