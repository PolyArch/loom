#ifndef LOOM_HARDWARE_RTL_PROVIDERS_SCALARMATHTRIGONOMETRIC_H
#define LOOM_HARDWARE_RTL_PROVIDERS_SCALARMATHTRIGONOMETRIC_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableScalarMathTrigonometricProviders(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_SCALARMATHTRIGONOMETRIC_H
