#ifndef LOOM_HARDWARE_RTL_PROVIDERS_NATIVE_DESIGNWARE_H
#define LOOM_HARDWARE_RTL_PROVIDERS_NATIVE_DESIGNWARE_H

#include "Hardware/RTL/Specialization.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace loom::hardware::rtl {

inline constexpr llvm::StringLiteral synopsysDesignWareContractRef =
    "synopsys.designware.component@1";
inline constexpr llvm::StringLiteral synopsysDesignWareComponentInputSlot =
    "component";
inline constexpr llvm::StringLiteral synopsysDesignWareBuildIdentity =
    "synopsys.designware:Y-2026.03-DWBB_202603.2";
inline constexpr llvm::StringLiteral synopsysDesignWareDwFpMacResourceKey =
    "dwbb/DW_fp_mac.DW_fp_dp2";

llvm::Error registerSynopsysDesignWareScalarFloatFmaProvider(
    FabricOperationProviderRegistry &registry);

llvm::Error registerSynopsysDesignWareExternalContract(
    ExternalImplementationContractCatalog &catalog);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_NATIVE_DESIGNWARE_H
