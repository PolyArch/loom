#include "Hardware/RTL/SystemImplementation.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Hardware/Implementation/ImplementationRepresentationRoot.h"
#include "Hardware/RTL/CommonSkeleton.h"
#include "Hardware/RTL/SystemSkeleton.h"

#include <cstdint>
#include <vector>

namespace loom::hardware::rtl {

llvm::Expected<FinalizedHardwareImplementation>
finalizePortableSystemHardwareImplementation(
    mlir::MLIRContext &context,
    const FinalizedConfigurationABI &configurationAbi,
    const FabricOperationProviderRegistry &providers,
    const ExternalImplementationContractCatalog &externalContracts,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto skeleton = buildPortableSystemRootCirctSkeleton(
      context, configurationAbi, providers, externalContracts);
  if (!skeleton)
    return skeleton.takeError();
  auto systemVerilog =
      lowerAndExportSpecializedSystemVerilog(*skeleton->module);
  if (!systemVerilog)
    return systemVerilog.takeError();
  const std::vector<std::uint8_t> bytes(systemVerilog->begin(),
                                        systemVerilog->end());
  auto digest = blobs.put(bytes);
  if (!digest)
    return digest.takeError();
  auto format = RepresentationFormatDescriptorRef::get(
      RepresentationFormatKind::SystemVerilogRtl);
  if (!format)
    return format.takeError();
  auto representation = createImplementationRepresentationRoot(
      RepresentationRootVariant::Rtl, std::nullopt, *format,
      {RepresentationObjectKind::Module, "loom_system"},
      {{PayloadRole::RtlSource, "rtl/loom_system.sv", *digest}});
  if (!representation)
    return representation.takeError();
  return finalizeHardwareImplementation(
      HardwareImplementationDraft{configurationAbi.abi().fabric(),
                                  configurationAbi.reference(),
                                  {},
                                  std::move(*representation),
                                  std::nullopt,
                                  std::move(skeleton->interfaces),
                                  {},
                                  {},
                                  {}},
      externalContracts, artifacts, blobs);
}

} // namespace loom::hardware::rtl
