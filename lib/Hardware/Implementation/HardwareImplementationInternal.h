#ifndef LOOM_LIB_HARDWARE_IMPLEMENTATION_HARDWAREIMPLEMENTATIONINTERNAL_H
#define LOOM_LIB_HARDWARE_IMPLEMENTATION_HARDWAREIMPLEMENTATIONINTERNAL_H

#include "Hardware/Implementation/HardwareImplementation.h"

namespace loom::fabric {
class FabricArtifactView;
}

namespace loom::hardware::detail {

llvm::Error
validateRepresentationLocator(const RepresentationLocator &locator,
                              const ImplementationRepresentationRoot &root);

llvm::Expected<std::vector<MemoryMacroBinding>> canonicalizeMemoryMacroBindings(
    llvm::ArrayRef<MemoryMacroBindingDraft> bindings,
    llvm::ArrayRef<ExternalImplementationBinding> externalBindings,
    llvm::ArrayRef<std::uint64_t> authoredToCanonicalBinding,
    const ExternalImplementationContractCatalog &contracts,
    const ImplementationRepresentationRoot &representation,
    const fabric::FabricSystemRootView &fabric);

} // namespace loom::hardware::detail

#endif // LOOM_LIB_HARDWARE_IMPLEMENTATION_HARDWAREIMPLEMENTATIONINTERNAL_H
