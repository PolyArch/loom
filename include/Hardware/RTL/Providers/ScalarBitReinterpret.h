#ifndef LOOM_HARDWARE_RTL_PROVIDERS_SCALARBITREINTERPRET_H
#define LOOM_HARDWARE_RTL_PROVIDERS_SCALARBITREINTERPRET_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableScalarBitReinterpretProvider(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_SCALARBITREINTERPRET_H
