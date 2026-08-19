#ifndef LOOM_HARDWARE_RTL_SPATIALCOREIMPLEMENTATION_H
#define LOOM_HARDWARE_RTL_SPATIALCOREIMPLEMENTATION_H

#include "Hardware/Implementation/HardwareImplementation.h"
#include "Hardware/RTL/Specialization.h"

#include "llvm/Support/Error.h"

#include <optional>

namespace mlir {
class MLIRContext;
}

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::hardware::rtl {

/// Materializes and publishes one self-contained portable RTL implementation
/// for one exact SpatialCore occurrence in a finalized System.
llvm::Expected<FinalizedHardwareImplementation>
finalizePortableSpatialCoreHardwareImplementation(
    mlir::MLIRContext &context,
    const FinalizedConfigurationABI &configurationAbi,
    fabric::SpatialCoreOccurrenceRef subject,
    std::optional<ArtifactRootReference> implementationPlatform,
    const FabricOperationProviderRegistry &providers,
    const ExternalImplementationContractCatalog &externalContracts,
    const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_SPATIALCOREIMPLEMENTATION_H
