#ifndef LOOM_HARDWARE_RTL_PROVIDERS_FIXEDVECTORINTEGERMULTIPLY_H
#define LOOM_HARDWARE_RTL_PROVIDERS_FIXEDVECTORINTEGERMULTIPLY_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableFixedVectorIntegerMultiplyProvider(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_FIXEDVECTORINTEGERMULTIPLY_H
