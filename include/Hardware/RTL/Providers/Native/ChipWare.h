#ifndef LOOM_HARDWARE_RTL_PROVIDERS_NATIVE_CHIPWARE_H
#define LOOM_HARDWARE_RTL_PROVIDERS_NATIVE_CHIPWARE_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

inline constexpr llvm::StringLiteral cadenceChipWareExternalContractRef =
    "cadence.chipware.cw_mult@1";
inline constexpr llvm::StringLiteral cadenceChipWareComponentModelSlotRef =
    "component_model";
inline constexpr llvm::StringLiteral cadenceChipWareCwMultModuleName =
    "CW_mult";
inline constexpr llvm::StringLiteral cadenceChipWareCwMultResourceKey =
    "chipware:CW_mult";
inline constexpr llvm::StringLiteral cadenceChipWareCwMultBlackBoxLogicalName =
    "blackbox/cadence-chipware-cw-mult-i8.txt";

llvm::Error registerCadenceChipWareExternalImplementationContract(
    ExternalImplementationContractCatalog &catalog);

llvm::Error registerCadenceChipWareScalarIntegerMultiplyProvider(
    FabricOperationProviderRegistry &registry);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_NATIVE_CHIPWARE_H
