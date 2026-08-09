#ifndef LOOM_HARDWARE_IMPLEMENTATION_FPGANATIVEEXTERNALCONTRACTS_H
#define LOOM_HARDWARE_IMPLEMENTATION_FPGANATIVEEXTERNALCONTRACTS_H

#include "Hardware/Implementation/HardwareImplementation.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>

namespace loom::hardware {

struct FpgaNativeExternalModuleContract final {
  platform::FpgaVendor vendor;
  llvm::StringRef contractRef;
  llvm::StringRef providerInputSlotRef;
  llvm::StringRef stableProviderBuildIdentity;
  llvm::StringRef resourceKey;
  llvm::StringRef deviceOrderingCode;
  llvm::StringRef moduleName;
  llvm::StringRef blackBoxPayloadLogicalName;
  llvm::StringRef blackBoxContractBytes;
};

std::string amdVivadoToolBundledResourceProviderIdentity(
    llvm::StringRef stableProviderBuildIdentity);

const FpgaNativeExternalModuleContract &amdXilinxDsp58ExternalModuleContract();

const FpgaNativeExternalModuleContract &
intelAlteraLpmMultExternalModuleContract();

llvm::Expected<ExternalImplementationContractCatalog>
makeFpgaNativeExternalImplementationContractCatalog();

} // namespace loom::hardware

#endif // LOOM_HARDWARE_IMPLEMENTATION_FPGANATIVEEXTERNALCONTRACTS_H
