#ifndef LOOM_HARDWARE_RTL_PROVIDERS_LOOPCARRY_H
#define LOOM_HARDWARE_RTL_PROVIDERS_LOOPCARRY_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error
registerPortableLoopCarryProvider(FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_LOOPCARRY_H
