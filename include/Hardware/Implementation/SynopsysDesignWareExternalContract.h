#ifndef LOOM_HARDWARE_IMPLEMENTATION_SYNOPSYSDESIGNWAREEXTERNALCONTRACT_H
#define LOOM_HARDWARE_IMPLEMENTATION_SYNOPSYSDESIGNWAREEXTERNALCONTRACT_H

#include "Hardware/Implementation/HardwareImplementation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom::hardware {

inline constexpr llvm::StringLiteral synopsysDesignWareContractRef =
    "synopsys.designware.component@1";
inline constexpr llvm::StringLiteral synopsysDesignWareComponentInputSlot =
    "component";
inline constexpr llvm::StringLiteral synopsysDesignWareBuildIdentity =
    "synopsys.designware:Y-2026.03-DWBB_202603.2";
inline constexpr llvm::StringLiteral synopsysDesignWareDwFpMacResourceKey =
    "dwbb/DW_fp_mac";
inline constexpr llvm::StringLiteral synopsysDesignWareDwFpMacComponentName =
    "DW_fp_mac";
inline constexpr llvm::StringLiteral
    synopsysDesignWareDwFpMacBlackBoxLogicalName =
        "black-box/synopsys-designware-dw-fp-mac-f32-rne-ieee-v1";

llvm::ArrayRef<std::uint8_t> synopsysDesignWareDwFpMacBlackBoxContractBytes();

bool isSynopsysDesignWareDwFpMacComponentInput(
    llvm::ArrayRef<ExternalInputBinding> inputs);

llvm::Error registerSynopsysDesignWareExternalContract(
    ExternalImplementationContractCatalog &catalog);

} // namespace loom::hardware

#endif // LOOM_HARDWARE_IMPLEMENTATION_SYNOPSYSDESIGNWAREEXTERNALCONTRACT_H
