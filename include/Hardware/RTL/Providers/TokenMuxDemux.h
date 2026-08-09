#ifndef LOOM_HARDWARE_RTL_PROVIDERS_TOKENMUXDEMUX_H
#define LOOM_HARDWARE_RTL_PROVIDERS_TOKENMUXDEMUX_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerPortableTokenMuxDemuxProviders(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_TOKENMUXDEMUX_H
