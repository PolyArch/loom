#ifndef LOOM_LIB_PNR_SYSTEM_RESOURCETIMETRANSITIONINTERNAL_H
#define LOOM_LIB_PNR_SYSTEM_RESOURCETIMETRANSITIONINTERNAL_H

#include "PnR/System/SystemMappingMigration.h"

namespace loom::pnr {

llvm::Error verifyResourceTimeTransitionDeltaDigests(
    const ResourceTimeTransition &transition, const ArtifactStore &artifacts,
    const BlobStore &blobs);

} // namespace loom::pnr

#endif // LOOM_LIB_PNR_SYSTEM_RESOURCETIMETRANSITIONINTERNAL_H
