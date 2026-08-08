#ifndef LOOM_HARDWARE_RTL_PROVIDERS_FIXEDVECTORFLOATFMA_H
#define LOOM_HARDWARE_RTL_PROVIDERS_FIXEDVECTORFLOATFMA_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableFixedVectorFloatFmaProvider(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_FIXEDVECTORFLOATFMA_H
