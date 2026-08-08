#ifndef LOOM_HARDWARE_RTL_PROVIDERS_FLOATDIVIDEREMAINDER_H
#define LOOM_HARDWARE_RTL_PROVIDERS_FLOATDIVIDEREMAINDER_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableFloatDivideRemainderProviders(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_FLOATDIVIDEREMAINDER_H
