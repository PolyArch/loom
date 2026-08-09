#ifndef LOOM_HARDWARE_RTL_PROVIDERS_MATHLOGARITHM_H
#define LOOM_HARDWARE_RTL_PROVIDERS_MATHLOGARITHM_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableMathLogarithmProviders(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_MATHLOGARITHM_H
