#ifndef LOOM_HARDWARE_RTL_PROVIDERS_FIXEDVECTORSHUFFLE_H
#define LOOM_HARDWARE_RTL_PROVIDERS_FIXEDVECTORSHUFFLE_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableFixedVectorShuffleProvider(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_FIXEDVECTORSHUFFLE_H
