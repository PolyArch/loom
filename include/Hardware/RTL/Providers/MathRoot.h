#ifndef LOOM_HARDWARE_RTL_PROVIDERS_MATHROOT_H
#define LOOM_HARDWARE_RTL_PROVIDERS_MATHROOT_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error
registerPortableMathRootProviders(FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_MATHROOT_H
