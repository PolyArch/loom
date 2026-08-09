#ifndef LOOM_HARDWARE_RTL_PROVIDERS_AMDXILINXSCALARINTEGERMULTIPLY_H
#define LOOM_HARDWARE_RTL_PROVIDERS_AMDXILINXSCALARINTEGERMULTIPLY_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerAmdXilinxScalarIntegerMultiplyProvider(
    FabricOperationProviderRegistry &registry);

llvm::Error registerAmdXilinxDsp58ExternalImplementationContract(
    ExternalImplementationContractCatalog &catalog);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_AMDXILINXSCALARINTEGERMULTIPLY_H
