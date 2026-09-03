#ifndef LOOM_LIB_APPLICATION_RUNTIMEPRODUCTCONTRACT_H
#define LOOM_LIB_APPLICATION_RUNTIMEPRODUCTCONTRACT_H

#include "Application/RuntimeManifest.h"

namespace loom {
class ArtifactStore;
class BlobStore;

namespace deployment {
class FinalizedDeployment;
}

namespace sim {
struct ImportedStructuredProgramSimulationInputs;
struct ImportedSystemSimulationInputs;
}
} // namespace loom

namespace loom::application::detail {

llvm::Error verifyRuntimeProductContract(
    const ProductOracleContract &contract,
    const sim::ImportedStructuredProgramSimulationInputs &sourceInputs,
    const sim::ImportedSystemSimulationInputs &activationInputs,
    const deployment::FinalizedDeployment &deployment,
    const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::application::detail

#endif // LOOM_LIB_APPLICATION_RUNTIMEPRODUCTCONTRACT_H
