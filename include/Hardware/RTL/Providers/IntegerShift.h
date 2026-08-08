#ifndef LOOM_HARDWARE_RTL_PROVIDERS_INTEGERSHIFT_H
#define LOOM_HARDWARE_RTL_PROVIDERS_INTEGERSHIFT_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableIntegerShiftProviders(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_INTEGERSHIFT_H
