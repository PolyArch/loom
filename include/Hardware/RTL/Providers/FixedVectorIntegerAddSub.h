#ifndef LOOM_HARDWARE_RTL_PROVIDERS_FIXEDVECTORINTEGERADDSUB_H
#define LOOM_HARDWARE_RTL_PROVIDERS_FIXEDVECTORINTEGERADDSUB_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableFixedVectorIntegerAddSubProvider(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_FIXEDVECTORINTEGERADDSUB_H
