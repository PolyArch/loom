#ifndef LOOM_LIB_FABRIC_ARTIFACT_FABRICCAPABILITYPROJECTION_H
#define LOOM_LIB_FABRIC_ARTIFACT_FABRICCAPABILITYPROJECTION_H

#include "../Identity/FabricArtifactViewInternal.h"
#include "Fabric/IR/FabricOps.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::fabric::detail {

llvm::Expected<std::vector<std::uint8_t>>
projectMemoryEndpointType(::mlir::Type type);

llvm::Error setModuleBoundaryInventory(::fabric::ModuleOp root,
                                       FabricEntityViewData &entity);

llvm::Expected<ResolvedFabricOpCapabilityView>
resolveFabricOpCapability(::fabric::OpOp operation,
                          const FabricFuTemplateNodeRef &reference,
                          FabricFuNodeViewData &node);

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_ARTIFACT_FABRICCAPABILITYPROJECTION_H
