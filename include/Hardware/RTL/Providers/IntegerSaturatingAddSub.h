#ifndef LOOM_HARDWARE_RTL_PROVIDERS_INTEGERSATURATINGADDSUB_H
#define LOOM_HARDWARE_RTL_PROVIDERS_INTEGERSATURATINGADDSUB_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableIntegerSaturatingAddSubProviders(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_INTEGERSATURATINGADDSUB_H
