#ifndef LOOM_EDA_ADAPTERS_FPGAIMPLEMENTATIONPUBLICATION_H
#define LOOM_EDA_ADAPTERS_FPGAIMPLEMENTATIONPUBLICATION_H

#include "Hardware/Implementation/HardwareImplementation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::eda {

/// Injectively projects an exact vendor ordering code into the one-identifier
/// grammar required by a physical DeviceResource locator.
std::string fpgaDeviceResourceLocatorName(llvm::StringRef deviceOrderingCode);

/// Publishes the exact routed physical state produced by a static FPGA flow.
/// The device ordering code becomes the DeviceResource root; source-rooted
/// locators are projected beneath it and external implementation closure is
/// retained for strict re-import.
llvm::Expected<hardware::FinalizedHardwareImplementation>
publishRoutedFpgaPhysicalImplementation(
    const hardware::FinalizedHardwareImplementation &source,
    const ArtifactRootReference &implementationPlatform,
    llvm::StringRef deviceOrderingCode, llvm::StringRef logicalName,
    llvm::StringRef contents,
    const hardware::ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs);

/// Publishes the exact configured image derived from a routed FPGA physical
/// state. External implementation closure has been incorporated into the
/// image and therefore is not retained as an unresolved dependency.
llvm::Expected<hardware::FinalizedHardwareImplementation>
publishFpgaImageImplementation(
    const hardware::FinalizedHardwareImplementation &routedPhysical,
    const ArtifactRootReference &implementationPlatform,
    llvm::StringRef deviceOrderingCode, llvm::StringRef logicalName,
    llvm::StringRef contents,
    const hardware::ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::eda

#endif // LOOM_EDA_ADAPTERS_FPGAIMPLEMENTATIONPUBLICATION_H
