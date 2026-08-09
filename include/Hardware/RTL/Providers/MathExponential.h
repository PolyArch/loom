#ifndef LOOM_HARDWARE_RTL_PROVIDERS_MATHEXPONENTIAL_H
#define LOOM_HARDWARE_RTL_PROVIDERS_MATHEXPONENTIAL_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableMathExponentialProviders(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_MATHEXPONENTIAL_H
