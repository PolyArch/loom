#ifndef LOOM_HARDWARE_RTL_SYSTEMIMPLEMENTATION_H
#define LOOM_HARDWARE_RTL_SYSTEMIMPLEMENTATION_H

#include "Hardware/Implementation/HardwareImplementation.h"
#include "Hardware/RTL/Specialization.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

namespace mlir {
class MLIRContext;
}

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::hardware::rtl {

/// Materializes and publishes one self-contained portable System-rooted RTL
/// HardwareImplementation from the exact Fabric and ConfigurationABI owners.
llvm::Expected<FinalizedHardwareImplementation>
finalizePortableSystemHardwareImplementation(
    mlir::MLIRContext &context,
    const FinalizedConfigurationABI &configurationAbi,
    const FabricOperationProviderRegistry &providers,
    const ExternalImplementationContractCatalog &externalContracts,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    llvm::ArrayRef<ArtifactRootReference> interconnectImplementations = {});

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_SYSTEMIMPLEMENTATION_H
