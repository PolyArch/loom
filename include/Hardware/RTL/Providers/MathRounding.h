#ifndef LOOM_HARDWARE_RTL_PROVIDERS_MATHROUNDING_H
#define LOOM_HARDWARE_RTL_PROVIDERS_MATHROUNDING_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableMathRoundingProviders(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_MATHROUNDING_H
