#ifndef LOOM_LIB_HARDWARE_RTL_HIERARCHY_MODULEHIERARCHY_H
#define LOOM_LIB_HARDWARE_RTL_HIERARCHY_MODULEHIERARCHY_H

#include "Hardware/RTL/CommonSkeleton.h"
#include "Hardware/RTL/Transport.h"

#include "llvm/Support/Error.h"

namespace loom::hardware::rtl::hierarchy {

llvm::Expected<ModuleRootCirctSkeleton> buildModuleHierarchySkeleton(
    mlir::MLIRContext &context, fabric::SpatialCoreOccurrenceRef spatialCore,
    const ConfigurationABI &configurationAbi,
    const fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<ModuleBoundaryTransportPortProjection> projections);

} // namespace loom::hardware::rtl::hierarchy

#endif // LOOM_LIB_HARDWARE_RTL_HIERARCHY_MODULEHIERARCHY_H
