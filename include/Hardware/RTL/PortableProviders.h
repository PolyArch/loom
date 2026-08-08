#ifndef LOOM_HARDWARE_RTL_PORTABLEPROVIDERS_H
#define LOOM_HARDWARE_RTL_PORTABLEPROVIDERS_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error
registerPortableOperationProviders(FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PORTABLEPROVIDERS_H
