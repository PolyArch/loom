#ifndef LOOM_HARDWARE_RTL_PROVIDERS_MATHPOWER_H
#define LOOM_HARDWARE_RTL_PROVIDERS_MATHPOWER_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error
registerPortableMathPowerProvider(FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_MATHPOWER_H
