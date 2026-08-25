#ifndef LOOM_LIB_MAPPING_ARTIFACT_SPATIALMAPPINGMODULEREBASE_H
#define LOOM_LIB_MAPPING_ARTIFACT_SPATIALMAPPINGMODULEREBASE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

namespace mapping {
class SpatialOp;
}

namespace loom::fabric {
class FabricArtifactView;
struct FabricModuleEntityCorrespondence;
} // namespace loom::fabric

namespace loom::mapping::detail {

llvm::Error remapSpatialMappingModuleReferences(
    ::mapping::SpatialOp root, const ::loom::fabric::FabricArtifactView &parent,
    const ::loom::fabric::FabricArtifactView &child,
    llvm::ArrayRef<::loom::fabric::FabricModuleEntityCorrespondence>
        correspondence);

} // namespace loom::mapping::detail

#endif // LOOM_LIB_MAPPING_ARTIFACT_SPATIALMAPPINGMODULEREBASE_H
