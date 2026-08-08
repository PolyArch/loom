#ifndef LOOM_HARDWARE_RTL_PROVIDERS_FLOATSIGN_H
#define LOOM_HARDWARE_RTL_PROVIDERS_FLOATSIGN_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error
registerPortableFloatSignProviders(FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_FLOATSIGN_H
