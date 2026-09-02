#ifndef LOOM_HARDWARE_RTL_SPATIALCOREIMPLEMENTATION_H
#define LOOM_HARDWARE_RTL_SPATIALCOREIMPLEMENTATION_H

#include "Hardware/Implementation/HardwareImplementation.h"
#include "Hardware/RTL/RtlModuleGraph.h"
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

/// Uses the canonical portable provider registry and an isolated lowering
/// context to materialize one SpatialCore implementation.
llvm::Expected<FinalizedHardwareImplementation>
finalizePortableSpatialCoreHardwareImplementation(
    const FinalizedConfigurationABI &configurationAbi,
    fabric::SpatialCoreOccurrenceRef subject,
    std::optional<ArtifactRootReference> implementationPlatform,
    const ArtifactStore &artifacts, const BlobStore &blobs);

/// Re-derives the canonical portable operation-leaf specialization and checks
/// one imported implementation without publishing artifacts or payloads.
llvm::Error verifyPortableSpatialCoreHardwareImplementation(
    const FinalizedConfigurationABI &configurationAbi,
    const FinalizedHardwareImplementation &implementation);

/// Rebuilds the generated portable RTL without publication and returns its
/// transient CIRCT-owned module graph only when the exact imported
/// HardwareImplementation is byte-for-byte the canonical result. A valid but
/// different implementation returns std::nullopt.
llvm::Expected<std::optional<RtlModuleGraphProjection>>
projectPortableSpatialCoreRtlModuleGraph(
    const FinalizedConfigurationABI &configurationAbi,
    const FinalizedHardwareImplementation &implementation);

/// Materializes and publishes one self-contained portable RTL implementation
/// for one exact SpatialCore occurrence with an explicit provider catalog.
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
