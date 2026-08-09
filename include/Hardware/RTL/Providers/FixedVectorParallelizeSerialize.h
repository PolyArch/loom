#ifndef LOOM_HARDWARE_RTL_PROVIDERS_FIXEDVECTORPARALLELIZESERIALIZE_H
#define LOOM_HARDWARE_RTL_PROVIDERS_FIXEDVECTORPARALLELIZESERIALIZE_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableFixedVectorParallelizeSerializeProviders(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_FIXEDVECTORPARALLELIZESERIALIZE_H
