#ifndef LOOM_HARDWARE_RTL_PROVIDERS_FLOATCONVERSIONS_H
#define LOOM_HARDWARE_RTL_PROVIDERS_FLOATCONVERSIONS_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableFloatConversionProviders(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_FLOATCONVERSIONS_H
