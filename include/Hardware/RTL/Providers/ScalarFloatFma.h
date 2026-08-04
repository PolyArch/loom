#ifndef LOOM_HARDWARE_RTL_PROVIDERS_SCALARFLOATFMA_H
#define LOOM_HARDWARE_RTL_PROVIDERS_SCALARFLOATFMA_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableScalarFloatFmaProvider(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_SCALARFLOATFMA_H
