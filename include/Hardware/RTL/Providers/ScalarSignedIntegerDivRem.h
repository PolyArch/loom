#ifndef LOOM_HARDWARE_RTL_PROVIDERS_SCALARSIGNEDINTEGERDIVREM_H
#define LOOM_HARDWARE_RTL_PROVIDERS_SCALARSIGNEDINTEGERDIVREM_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableScalarSignedIntegerDivRemProvider(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_SCALARSIGNEDINTEGERDIVREM_H
