#ifndef LOOM_LIB_PNR_SYSTEM_RESOURCETIMETRANSITIONINTERNAL_H
#define LOOM_LIB_PNR_SYSTEM_RESOURCETIMETRANSITIONINTERNAL_H

#include "PnR/System/SystemMappingMigration.h"

namespace loom::mapping {
struct SystemExecutionContextProjection;
} // namespace loom::mapping

namespace loom::pnr {

llvm::Expected<std::vector<::loom::fabric::FabricPhysicalOccurrenceOwnerRef>>
projectResourceTimeMappingResources(
    const ::loom::mapping::SystemExecutionContextProjection &contexts,
    ::dataflow::RootThreadLaunchRef root);

llvm::Error verifyResourceTimeTransitionDeltaDigests(
    const ResourceTimeTransition &transition, const ArtifactStore &artifacts,
    const BlobStore &blobs);

} // namespace loom::pnr

#endif // LOOM_LIB_PNR_SYSTEM_RESOURCETIMETRANSITIONINTERNAL_H
