#ifndef LOOM_HARDWARE_RTL_PROVIDERS_SCALARUNSIGNEDINTEGERDIVREM_H
#define LOOM_HARDWARE_RTL_PROVIDERS_SCALARUNSIGNEDINTEGERDIVREM_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableScalarUnsignedIntegerDivRemProvider(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_SCALARUNSIGNEDINTEGERDIVREM_H
