#ifndef LOOM_HARDWARE_RTL_PROVIDERS_FIXEDVECTORPACKUNPACK_H
#define LOOM_HARDWARE_RTL_PROVIDERS_FIXEDVECTORPACKUNPACK_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableFixedVectorPackProvider(
    FabricOperationProviderRegistry &registry);

llvm::Error registerPortableFixedVectorUnpackProvider(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_FIXEDVECTORPACKUNPACK_H
