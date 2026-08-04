#ifndef LOOM_LIB_HARDWARE_IMPLEMENTATION_HARDWAREIMPLEMENTATIONINTERNAL_H
#define LOOM_LIB_HARDWARE_IMPLEMENTATION_HARDWAREIMPLEMENTATIONINTERNAL_H

#include "Hardware/Implementation/HardwareImplementation.h"

namespace loom::fabric {
class FabricArtifactView;
}

namespace loom::hardware::detail {

llvm::Error
validateRepresentationLocator(const RepresentationLocator &locator,
                              HardwareRepresentation representation);

llvm::Error canonicalizeMemoryMacroBindings(
    std::vector<MemoryMacroBinding> &bindings,
    llvm::ArrayRef<ExternalImplementationBinding> externalBindings,
    const ExternalImplementationContractCatalog &contracts,
    HardwareRepresentation representation,
    const fabric::FabricArtifactView &fabric);

} // namespace loom::hardware::detail

#endif // LOOM_LIB_HARDWARE_IMPLEMENTATION_HARDWAREIMPLEMENTATIONINTERNAL_H
