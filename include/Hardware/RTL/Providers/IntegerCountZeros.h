#ifndef LOOM_HARDWARE_RTL_PROVIDERS_INTEGERCOUNTZEROS_H
#define LOOM_HARDWARE_RTL_PROVIDERS_INTEGERCOUNTZEROS_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableIntegerCountZerosProviders(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_INTEGERCOUNTZEROS_H
