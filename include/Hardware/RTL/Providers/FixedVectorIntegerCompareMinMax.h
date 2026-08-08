#ifndef LOOM_HARDWARE_RTL_PROVIDERS_FIXEDVECTORINTEGERCOMPAREMINMAX_H
#define LOOM_HARDWARE_RTL_PROVIDERS_FIXEDVECTORINTEGERCOMPAREMINMAX_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableFixedVectorIntegerCompareMinMaxProvider(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_FIXEDVECTORINTEGERCOMPAREMINMAX_H
