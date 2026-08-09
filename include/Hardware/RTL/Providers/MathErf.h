#ifndef LOOM_HARDWARE_RTL_PROVIDERS_MATHERF_H
#define LOOM_HARDWARE_RTL_PROVIDERS_MATHERF_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error
registerPortableMathErfProvider(FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_MATHERF_H
