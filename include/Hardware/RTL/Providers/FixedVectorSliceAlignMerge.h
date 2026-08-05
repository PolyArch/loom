#ifndef LOOM_HARDWARE_RTL_PROVIDERS_FIXEDVECTORSLICEALIGNMERGE_H
#define LOOM_HARDWARE_RTL_PROVIDERS_FIXEDVECTORSLICEALIGNMERGE_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableFixedVectorSliceAlignMergeProvider(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_FIXEDVECTORSLICEALIGNMERGE_H
