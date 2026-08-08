#ifndef LOOM_HARDWARE_RTL_PHYSICALOPERATION_H
#define LOOM_HARDWARE_RTL_PHYSICALOPERATION_H

#include "Fabric/Artifact/FabricSystemRootView.h"

#include "llvm/Support/Error.h"

#include <vector>

namespace loom::hardware::rtl {

/// Transient resolution of one occurrence-qualified operation inside an
/// imported SpatialCore Module. Fabric remains the identity and capability
/// owner; this view only removes the System qualification for local queries.
struct ResolvedFabricPhysicalOperation final {
  fabric::FabricPhysicalOccurrenceOwnerRef physicalOccurrence;
  fabric::FabricArtifactView module;
  fabric::FabricFuOccurrenceNodeRef localOccurrence;
  const fabric::ResolvedFabricOpCapabilityView *capability = nullptr;
};

llvm::Expected<fabric::FabricArtifactView>
resolveFabricSpatialCoreModule(const fabric::FabricSystemRootView &system,
                               fabric::SpatialCoreOccurrenceRef spatialCore);

llvm::Expected<ResolvedFabricPhysicalOperation> resolveFabricPhysicalOperation(
    const fabric::FabricSystemRootView &system,
    const fabric::FabricPhysicalOccurrenceOwnerRef &occurrence);

llvm::Expected<std::vector<ResolvedFabricPhysicalOperation>>
enumerateFabricPhysicalOperations(const fabric::FabricSystemRootView &system);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PHYSICALOPERATION_H
