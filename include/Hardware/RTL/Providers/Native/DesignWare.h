#ifndef LOOM_HARDWARE_RTL_PROVIDERS_NATIVE_DESIGNWARE_H
#define LOOM_HARDWARE_RTL_PROVIDERS_NATIVE_DESIGNWARE_H

#include "Hardware/Implementation/SynopsysDesignWareExternalContract.h"
#include "Hardware/RTL/Specialization.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace loom::hardware::rtl {

using hardware::isSynopsysDesignWareDwFpMacComponentInput;
using hardware::registerSynopsysDesignWareExternalContract;
using hardware::synopsysDesignWareBuildIdentity;
using hardware::synopsysDesignWareComponentInputSlot;
using hardware::synopsysDesignWareContractRef;
using hardware::synopsysDesignWareDwFpMacBlackBoxContractBytes;
using hardware::synopsysDesignWareDwFpMacBlackBoxLogicalName;
using hardware::synopsysDesignWareDwFpMacComponentName;
using hardware::synopsysDesignWareDwFpMacResourceKey;

llvm::Error registerSynopsysDesignWareScalarFloatFmaProvider(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_NATIVE_DESIGNWARE_H
