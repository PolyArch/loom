#ifndef LOOM_TEST_HARDWARE_PORTABLEPROVIDERTESTSUPPORT_H
#define LOOM_TEST_HARDWARE_PORTABLEPROVIDERTESTSUPPORT_H

#include "Hardware/RTL/CommonSkeleton.h"
#include "Hardware/RTL/Specialization.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <filesystem>
#include <string>

namespace loom::hardware::test {

struct PortableProviderArtifact final {
  std::filesystem::path relativePath;
  std::string contents;
};

struct PortableProviderConformance final {
  rtl::FabricOperationProviderOutput providerOutput;
  std::string systemVerilog;
};

llvm::Expected<PortableProviderConformance> specializeAndExportPortableProvider(
    rtl::ModuleRootCirctSkeleton skeleton,
    const FinalizedConfigurationABI &configurationAbi,
    const rtl::FabricOperationProviderRegistry &providers,
    const ExternalImplementationContractCatalog &externalContracts);

llvm::Error writePortableProviderArtifacts(
    const std::filesystem::path &root,
    llvm::ArrayRef<PortableProviderArtifact> artifacts);

} // namespace loom::hardware::test

#endif // LOOM_TEST_HARDWARE_PORTABLEPROVIDERTESTSUPPORT_H
