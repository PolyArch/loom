#ifndef LOOM_HARDWARE_RTL_PROVIDERS_SCALARINTEGERMULTIPLY_H
#define LOOM_HARDWARE_RTL_PROVIDERS_SCALARINTEGERMULTIPLY_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableScalarIntegerMultiplyProvider(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_SCALARINTEGERMULTIPLY_H
