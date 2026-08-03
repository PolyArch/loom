#ifndef LOOM_LIB_FABRIC_IDENTITY_FABRICPHYSICALTAGPROJECTION_H
#define LOOM_LIB_FABRIC_IDENTITY_FABRICPHYSICALTAGPROJECTION_H

#include "Fabric/Identity/FabricRefImport.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::fabric::detail {

std::optional<FabricPhysicalTagMatchDomainView>
projectPhysicalTagMatchDomain(const FabricArtifactView &view,
                              const FabricTransportEndpointRef &endpoint);

std::optional<FabricPhysicalTagAssignmentPointKind>
classifyPhysicalTagAssignmentPoint(const FabricArtifactView &view,
                                   const FabricTransportEndpointRef &endpoint);

std::vector<std::uint8_t>
ownerWideTagMatchDomainKey(const FabricPhysicalTagMatchDomainView &domain);

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_IDENTITY_FABRICPHYSICALTAGPROJECTION_H
