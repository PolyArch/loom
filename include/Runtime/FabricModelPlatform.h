#ifndef LOOM_RUNTIME_FABRICMODELPLATFORM_H
#define LOOM_RUNTIME_FABRICMODELPLATFORM_H

#include "Hardware/Implementation/HardwareImplementation.h"
#include "Runtime/RuntimePlatformBinding.h"

#include "llvm/Support/Error.h"

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::runtime {

/// Static provider contract for payload-free FabricModel implementations.
const RuntimeProviderDescriptor &fabricModelRuntimeProviderDescriptor();

/// Derives the exact provider endpoints for every runtime-visible interface
/// of one FabricModel HardwareImplementation and publishes the strict binding.
llvm::Expected<FinalizedRuntimePlatformBinding>
finalizeFabricModelRuntimePlatformBinding(
    const hardware::FinalizedHardwareImplementation &implementation,
    const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::runtime

#endif // LOOM_RUNTIME_FABRICMODELPLATFORM_H
