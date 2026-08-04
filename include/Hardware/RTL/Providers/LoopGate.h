#ifndef LOOM_HARDWARE_RTL_PROVIDERS_LOOPGATE_H
#define LOOM_HARDWARE_RTL_PROVIDERS_LOOPGATE_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error
registerPortableLoopGateProvider(FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_LOOPGATE_H
