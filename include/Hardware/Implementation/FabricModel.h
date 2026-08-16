#ifndef LOOM_HARDWARE_IMPLEMENTATION_FABRICMODEL_H
#define LOOM_HARDWARE_IMPLEMENTATION_FABRICMODEL_H

#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Implementation/HardwareImplementation.h"

#include "llvm/Support/Error.h"

#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::hardware {

/// Derives the complete semantic interface closure of one SpatialCore
/// occurrence from its exact System and ConfigurationABI.
llvm::Expected<std::vector<ImplementationInterfaceSemanticRef>>
deriveSpatialCoreImplementationInterfaceSemantics(
    const FinalizedConfigurationABI &configurationAbi,
    fabric::SpatialCoreOccurrenceRef subject);

/// Publishes the payload-free behavioral implementation relation used by
/// semantic DFG and CGRA runtimes. This operation does not generate RTL.
llvm::Expected<FinalizedHardwareImplementation>
finalizeFabricModelHardwareImplementation(
    const FinalizedConfigurationABI &configurationAbi,
    fabric::SpatialCoreOccurrenceRef subject, const ArtifactStore &artifacts,
    const BlobStore &blobs);

} // namespace loom::hardware

#endif // LOOM_HARDWARE_IMPLEMENTATION_FABRICMODEL_H
