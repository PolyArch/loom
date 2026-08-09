#ifndef LOOM_HARDWARE_RTL_PROVIDERS_LOOPSTREAM_H
#define LOOM_HARDWARE_RTL_PROVIDERS_LOOPSTREAM_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error
registerPortableLoopStreamProvider(FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_LOOPSTREAM_H
