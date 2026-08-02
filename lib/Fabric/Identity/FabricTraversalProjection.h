#ifndef LOOM_LIB_FABRIC_IDENTITY_FABRICTRAVERSALPROJECTION_H
#define LOOM_LIB_FABRIC_IDENTITY_FABRICTRAVERSALPROJECTION_H

#include "Fabric/Identity/FabricRefImport.h"

namespace loom::fabric::detail {

llvm::Expected<FabricPhysicalTraversalView>
projectFabricTraversal(const FabricArtifactView &view,
                       const FabricPhysicalTraversalRef &reference);

llvm::Expected<std::vector<FabricInventoryOwnerRef>>
projectModuleResourceOwners(const FabricArtifactView &view);

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_IDENTITY_FABRICTRAVERSALPROJECTION_H
