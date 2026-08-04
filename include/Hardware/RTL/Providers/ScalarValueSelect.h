#ifndef LOOM_HARDWARE_RTL_PROVIDERS_SCALARVALUESELECT_H
#define LOOM_HARDWARE_RTL_PROVIDERS_SCALARVALUESELECT_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableScalarValueSelectProvider(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_SCALARVALUESELECT_H
