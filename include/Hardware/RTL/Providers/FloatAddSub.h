#ifndef LOOM_HARDWARE_RTL_PROVIDERS_FLOATADDSUB_H
#define LOOM_HARDWARE_RTL_PROVIDERS_FLOATADDSUB_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error
registerPortableFloatAddSubProviders(FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_FLOATADDSUB_H
