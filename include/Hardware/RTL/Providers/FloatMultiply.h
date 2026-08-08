#ifndef LOOM_HARDWARE_RTL_PROVIDERS_FLOATMULTIPLY_H
#define LOOM_HARDWARE_RTL_PROVIDERS_FLOATMULTIPLY_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableFloatMultiplyProviders(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_FLOATMULTIPLY_H
