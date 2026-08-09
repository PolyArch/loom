#ifndef LOOM_HARDWARE_RTL_PROVIDERS_SCALARMATHHYPERBOLIC_H
#define LOOM_HARDWARE_RTL_PROVIDERS_SCALARMATHHYPERBOLIC_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableScalarMathHyperbolicProviders(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_SCALARMATHHYPERBOLIC_H
