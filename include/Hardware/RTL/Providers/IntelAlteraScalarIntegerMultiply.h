#ifndef LOOM_HARDWARE_RTL_PROVIDERS_INTELALTERASCALARINTEGERMULTIPLY_H
#define LOOM_HARDWARE_RTL_PROVIDERS_INTELALTERASCALARINTEGERMULTIPLY_H

#include "Hardware/RTL/Specialization.h"

namespace loom::hardware::rtl {

llvm::Error registerIntelAlteraScalarIntegerMultiplyProvider(
    FabricOperationProviderRegistry &registry);

llvm::Error registerIntelAlteraLpmMultExternalImplementationContract(
    ExternalImplementationContractCatalog &catalog);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_PROVIDERS_INTELALTERASCALARINTEGERMULTIPLY_H
