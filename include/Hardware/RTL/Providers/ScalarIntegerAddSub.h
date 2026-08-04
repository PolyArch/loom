#ifndef LOOM_HARDWARE_RTL_PROVIDERS_SCALARINTEGERADDSUB_H
#define LOOM_HARDWARE_RTL_PROVIDERS_SCALARINTEGERADDSUB_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableScalarIntegerAddSubProvider(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_SCALARINTEGERADDSUB_H
