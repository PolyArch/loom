#include "Hardware/RTL/SpatialCoreImplementation.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Hardware/Implementation/ImplementationRepresentationRoot.h"
#include "Hardware/RTL/CommonSkeleton.h"
#include "Hardware/RTL/ConfigurationTransport.h"

#include "circt/Dialect/HW/HWOps.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace loom::hardware::rtl {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "rtl_spatial_core_implementation_invalid: " + message);
}

llvm::Expected<fabric::HardwareDomainRef>
findDomain(const fabric::FabricSystemRootView &system,
           fabric::SpatialCoreOccurrenceRef subject,
           fabric::FabricHardwareDomainKind kind) {
  const fabric::FabricInventoryOwnerRef owner =
      fabric::FabricInventoryOwnerRef::of(subject);
  std::optional<fabric::HardwareDomainRef> result;
  for (fabric::HardwareDomainRef domain : system.hardwareDomains()) {
    const auto *contract = system.hardwareDomainContract(domain);
    if (!contract || contract->kind() != kind ||
        !llvm::is_contained(contract->members(), owner))
      continue;
    if (result)
      return invalid("SpatialCore belongs to multiple required domains");
    result = domain;
  }
  if (!result)
    return invalid("SpatialCore has no required Clock or Reset domain");
  return *result;
}

llvm::Expected<fabric::SpatialCoreOccurrenceRef>
attachmentSubject(
    const fabric::FabricSpatialAttachmentEndpointRef &endpoint) {
  if (const auto *transport = endpoint.transport()) {
    if (transport->owner.kind() !=
        fabric::FabricTransportEndpointOwnerKind::SpatialCoreOccurrence)
      return invalid("transport attachment is not SpatialCore-owned");
    return std::get<fabric::SpatialCoreOccurrenceRef>(
        transport->owner.payload);
  }
  const auto *memory = endpoint.memory();
  if (!memory || memory->owner.kind() !=
                     fabric::FabricMemoryEndpointOwnerKind::SpatialCoreOccurrence)
    return invalid("memory attachment is not SpatialCore-owned");
  return std::get<fabric::SpatialCoreOccurrenceRef>(memory->owner.payload);
}

std::string attachmentLocalPort(
    const fabric::FabricSpatialAttachmentRecordView &attachment) {
  const bool input = attachment.moduleEndpoint.target.direction ==
                     fabric::FabricPortDirection::Input;
  const std::string direction = input ? "input_" : "output_";
  const std::string ordinal =
      std::to_string(attachment.moduleEndpoint.target.ordinal);
  if (attachment.spatialEndpoint.transport())
    return direction + ordinal + "_valid";
  return "memory_" + direction + ordinal + "_request_valid";
}

ImplementationInterface
topPortInterface(ImplementationInterfaceSemanticRef semanticRef,
                 llvm::StringRef port) {
  return ImplementationInterface{
      std::move(semanticRef),
      {RepresentationObjectKind::Port, "loom_module." + port.str()},
      std::nullopt};
}

llvm::Expected<std::vector<ImplementationInterface>> deriveInterfaces(
    const FinalizedConfigurationABI &configurationAbi,
    fabric::SpatialCoreOccurrenceRef subject) {
  const fabric::FabricSystemRootView &system =
      configurationAbi.abi().fabricSystem();
  auto clock =
      findDomain(system, subject, fabric::FabricHardwareDomainKind::Clock);
  if (!clock)
    return clock.takeError();
  auto reset =
      findDomain(system, subject, fabric::FabricHardwareDomainKind::Reset);
  if (!reset)
    return reset.takeError();

  std::vector<ImplementationInterface> interfaces;
  interfaces.push_back(topPortInterface(
      ImplementationClockInterfaceRef{*clock}, "clock"));
  interfaces.push_back(topPortInterface(
      ImplementationResetInterfaceRef{*reset}, "reset"));

  auto layout =
      derivePortableConfigurationTransportLayout(configurationAbi, subject);
  if (!layout)
    return layout.takeError();
  for (const ConfigurationTransportUnitLayout &unit : layout->units)
    interfaces.push_back(topPortInterface(
        ImplementationConfigurationInterfaceRef{unit.programmingUnit},
        "cfg_awaddr"));

  for (const auto &attachment : system.spatialAttachments()) {
    auto owner = attachmentSubject(attachment.spatialEndpoint);
    if (!owner)
      return owner.takeError();
    if (*owner != subject)
      continue;
    const auto target = system.spatialCoreTarget(subject.core);
    if (!target ||
        target->dependencyOrdinal != attachment.moduleEndpoint.dependencyOrdinal ||
        target->target != attachment.moduleEndpoint.target.module)
      return invalid("System attachment targets a foreign Module occurrence");
    const std::string port = attachmentLocalPort(attachment);
    if (attachment.spatialEndpoint.transport())
      interfaces.push_back(topPortInterface(
          ImplementationDataInterfaceRef{attachment.spatialEndpoint}, port));
    else
      interfaces.push_back(topPortInterface(
          ImplementationMemoryInterfaceRef{attachment.spatialEndpoint}, port));
  }
  return interfaces;
}

} // namespace

llvm::Expected<FinalizedHardwareImplementation>
finalizePortableSpatialCoreHardwareImplementation(
    mlir::MLIRContext &context,
    const FinalizedConfigurationABI &configurationAbi,
    fabric::SpatialCoreOccurrenceRef subject,
    const FabricOperationProviderRegistry &providers,
    const ExternalImplementationContractCatalog &externalContracts,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto skeleton =
      buildModuleRootCirctSkeleton(context, subject, configurationAbi);
  if (!skeleton)
    return skeleton.takeError();

  std::vector<FabricOperationRecipeBinding> recipes;
  recipes.reserve(skeleton->operationLeaves.size());
  for (const FabricOperationLeafAssociation &association :
       skeleton->operationLeaves)
    recipes.push_back({association.occurrence,
                       BackendRecipeKey::PortableSystemVerilog, {}});
  auto specialization = specializeFabricOperationLeaves(
      *skeleton->module, configurationAbi, skeleton->operationLeaves, recipes,
      providers, externalContracts);
  if (!specialization)
    return specialization.takeError();
  if (!specialization->payloads.empty() ||
      !specialization->activityPoints.empty() ||
      !specialization->externalImplementationBindings.empty())
    return invalid("portable provider returned non-self-contained material");
  if (llvm::Error error = verifySpecializedCirctModule(*skeleton->module))
    return std::move(error);

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
      {RepresentationObjectKind::Module, "loom_module"},
      {{PayloadRole::RtlSource, "rtl/loom_spatial_core.sv", *digest}});
  if (!representation)
    return representation.takeError();
  auto interfaces = deriveInterfaces(configurationAbi, subject);
  if (!interfaces)
    return interfaces.takeError();
  return finalizeHardwareImplementation(
      HardwareImplementationDraft{configurationAbi.abi().fabric(), subject,
                                  configurationAbi.reference(),
                                  std::move(*representation), std::nullopt,
                                  std::move(*interfaces), {}, {}, {}},
      externalContracts, artifacts, blobs);
}

} // namespace loom::hardware::rtl
