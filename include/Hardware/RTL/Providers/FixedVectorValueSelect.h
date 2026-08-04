#ifndef LOOM_HARDWARE_RTL_PROVIDERS_FIXEDVECTORVALUESELECT_H
#define LOOM_HARDWARE_RTL_PROVIDERS_FIXEDVECTORVALUESELECT_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableFixedVectorValueSelectProvider(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_FIXEDVECTORVALUESELECT_H
