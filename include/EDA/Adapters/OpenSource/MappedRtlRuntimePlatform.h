#ifndef LOOM_EDA_ADAPTERS_OPENSOURCE_MAPPEDRTLRUNTIMEPLATFORM_H
#define LOOM_EDA_ADAPTERS_OPENSOURCE_MAPPEDRTLRUNTIMEPLATFORM_H

#include "Runtime/RuntimePlatformBinding.h"
#include "Runtime/RuntimeProvider.h"

#include "llvm/Support/Error.h"

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::hardware {
class FinalizedHardwareImplementation;
}

namespace loom::eda::open_source {

/// Finalizes the runtime platform binding of one RTL HardwareImplementation
/// under the mapped-RTL runtime provider, the runtime contract selected by a
/// Deployment whose exact implementation is executed through the mapped-RTL
/// external Evaluation provider.
llvm::Expected<runtime::FinalizedRuntimePlatformBinding>
finalizeMappedRtlRuntimePlatformBinding(
    const hardware::FinalizedHardwareImplementation &implementation,
    const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::eda::open_source

#endif // LOOM_EDA_ADAPTERS_OPENSOURCE_MAPPEDRTLRUNTIMEPLATFORM_H
