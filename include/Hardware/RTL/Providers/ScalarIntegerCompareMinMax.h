#ifndef LOOM_HARDWARE_RTL_PROVIDERS_SCALARINTEGERCOMPAREMINMAX_H
#define LOOM_HARDWARE_RTL_PROVIDERS_SCALARINTEGERCOMPAREMINMAX_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableScalarIntegerCompareMinMaxProvider(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_SCALARINTEGERCOMPAREMINMAX_H
