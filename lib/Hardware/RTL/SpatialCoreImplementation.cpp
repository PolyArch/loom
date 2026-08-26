#include "Hardware/RTL/SpatialCoreImplementation.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/IR/FabricDialect.h"
#include "Hardware/Implementation/FabricModel.h"
#include "Hardware/Implementation/ImplementationRepresentationRoot.h"
#include "Hardware/RTL/CommonSkeleton.h"
#include "Hardware/RTL/PortableProviders.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/IR/MLIRContext.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace loom::hardware::rtl {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "rtl_spatial_core_implementation_invalid: " +
                                     message);
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

llvm::Expected<std::string>
interfacePort(const ImplementationInterfaceSemanticRef &semantic,
              const fabric::FabricSystemRootView &system) {
  if (std::holds_alternative<ImplementationClockInterfaceRef>(semantic))
    return std::string("clock");
  if (std::holds_alternative<ImplementationResetInterfaceRef>(semantic))
    return std::string("reset");
  if (std::holds_alternative<ImplementationConfigurationInterfaceRef>(semantic))
    return std::string("cfg_awaddr");

  const fabric::FabricSpatialAttachmentEndpointRef *endpoint = nullptr;
  if (const auto *data = std::get_if<ImplementationDataInterfaceRef>(&semantic))
    endpoint = &data->endpoint;
  else if (const auto *memory =
               std::get_if<ImplementationMemoryInterfaceRef>(&semantic))
    endpoint = &memory->endpoint;
  else
    return invalid("SpatialCore RTL received an external protocol interface");
  for (const auto &attachment : system.spatialAttachments())
    if (attachment.spatialEndpoint == *endpoint)
      return attachmentLocalPort(attachment);
  return invalid("SpatialCore interface has no exact System attachment");
}

ImplementationInterface
topPortInterface(ImplementationInterfaceSemanticRef semanticRef,
                 llvm::StringRef port) {
  return ImplementationInterface{
      std::move(semanticRef),
      {RepresentationObjectKind::Port, "loom_module." + port.str()},
      std::nullopt};
}

llvm::Expected<std::vector<ImplementationInterface>>
deriveInterfaces(const FinalizedConfigurationABI &configurationAbi,
                 fabric::SpatialCoreOccurrenceRef subject) {
  const fabric::FabricSystemRootView &system =
      configurationAbi.abi().fabricSystem();
  auto semantics = deriveSpatialCoreImplementationInterfaceSemantics(
      configurationAbi, subject);
  if (!semantics)
    return semantics.takeError();
  std::vector<ImplementationInterface> interfaces;
  interfaces.reserve(semantics->size());
  for (ImplementationInterfaceSemanticRef &semantic : *semantics) {
    auto port = interfacePort(semantic, system);
    if (!port)
      return port.takeError();
    interfaces.push_back(topPortInterface(std::move(semantic), *port));
  }
  return interfaces;
}

std::string fixedDecimal(unsigned __int128 coefficient,
                         std::size_t fractionalDigits) {
  std::string digits;
  do {
    digits.push_back(static_cast<char>('0' + coefficient % 10));
    coefficient /= 10;
  } while (coefficient != 0);
  std::reverse(digits.begin(), digits.end());
  if (fractionalDigits == 0)
    return digits;
  if (digits.size() <= fractionalDigits)
    digits.insert(0, fractionalDigits + 1 - digits.size(), '0');
  digits.insert(digits.size() - fractionalDigits, 1, '.');
  while (digits.back() == '0')
    digits.pop_back();
  if (digits.back() == '.')
    digits.pop_back();
  return digits;
}

llvm::Expected<std::string>
deriveGenerationConstraint(const FinalizedConfigurationABI &configurationAbi,
                           llvm::ArrayRef<ImplementationInterface> interfaces) {
  const ImplementationInterface *clockInterface = nullptr;
  for (const ImplementationInterface &interface : interfaces) {
    if (!std::holds_alternative<ImplementationClockInterfaceRef>(
            interface.semanticRef))
      continue;
    if (clockInterface)
      return invalid("SpatialCore RTL has more than one clock interface");
    clockInterface = &interface;
  }
  if (!clockInterface || clockInterface->representationLocator.kind !=
                             RepresentationObjectKind::Port)
    return invalid("SpatialCore RTL has no exact clock port");
  constexpr llvm::StringLiteral topPrefix = "loom_module.";
  llvm::StringRef clockPort =
      clockInterface->representationLocator.canonicalName;
  if (!clockPort.consume_front(topPrefix) || clockPort.empty())
    return invalid("SpatialCore clock locator is outside the top module");

  const auto &semantic =
      std::get<ImplementationClockInterfaceRef>(clockInterface->semanticRef);
  const fabric::HardwareDomainContractRecord *domain =
      configurationAbi.abi().fabricSystem().hardwareDomainContract(
          semantic.domain);
  const auto *clock =
      domain
          ? std::get_if<fabric::ClockDomainContractRecord>(&domain->contract())
          : nullptr;
  if (!clock)
    return invalid("SpatialCore clock interface has no clock-domain contract");

  const std::string period = fixedDecimal(clock->periodFs(), 6);
  const std::string rising = fixedDecimal(clock->phaseFs(), 6);
  const unsigned __int128 fallingFemtosecondHalves =
      static_cast<unsigned __int128>(clock->phaseFs()) * 2 + clock->periodFs();
  const std::string falling = fixedDecimal(fallingFemtosecondHalves * 5, 7);
  return "create_clock -name loom_clock -period " + period + " -waveform {" +
         rising + " " + falling + "} [get_ports {" + clockPort.str() + "}]\n";
}

struct PendingPortablePayload final {
  BlobDigest digest;
  std::vector<std::uint8_t> bytes;
};

struct PortableSpatialCoreMaterialization final {
  HardwareImplementationDraft draft;
  std::vector<PendingPortablePayload> payloads;
};

llvm::Expected<PortableSpatialCoreMaterialization>
derivePortableSpatialCoreMaterialization(
    mlir::MLIRContext &context,
    const FinalizedConfigurationABI &configurationAbi,
    fabric::SpatialCoreOccurrenceRef subject,
    std::optional<ArtifactRootReference> implementationPlatform,
    const FabricOperationProviderRegistry &providers,
    const ExternalImplementationContractCatalog &externalContracts) {
  auto skeleton =
      buildModuleRootCirctSkeleton(context, subject, configurationAbi);
  if (!skeleton)
    return skeleton.takeError();

  std::vector<FabricOperationRecipeBinding> recipes;
  recipes.reserve(skeleton->operationLeaves.size());
  for (const FabricOperationLeafAssociation &association :
       skeleton->operationLeaves)
    recipes.push_back(
        {association.occurrence, BackendRecipeKey::PortableSystemVerilog, {}});
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
  std::vector<std::uint8_t> rtlBytes(systemVerilog->begin(),
                                     systemVerilog->end());
  const BlobDigest rtlDigest = computeBlobDigest(rtlBytes);
  auto interfaces = deriveInterfaces(configurationAbi, subject);
  if (!interfaces)
    return interfaces.takeError();
  canonicalizeHardwareImplementationInterfaceOrder(*interfaces);
  auto constraint = deriveGenerationConstraint(configurationAbi, *interfaces);
  if (!constraint)
    return constraint.takeError();
  std::vector<std::uint8_t> constraintBytes(
      reinterpret_cast<const std::uint8_t *>(constraint->data()),
      reinterpret_cast<const std::uint8_t *>(constraint->data()) +
          constraint->size());
  const BlobDigest constraintDigest = computeBlobDigest(constraintBytes);
  auto format = RepresentationFormatDescriptorRef::get(
      RepresentationFormatKind::SystemVerilogRtl);
  if (!format)
    return format.takeError();
  auto representation = createImplementationRepresentationRoot(
      RepresentationRootVariant::Rtl, std::nullopt, *format,
      {RepresentationObjectKind::Module, "loom_module"},
      {{PayloadRole::RtlSource, "rtl/loom_spatial_core.sv", rtlDigest},
       {PayloadRole::GenerationConstraint, "constraints/loom_spatial_core.sdc",
        constraintDigest}});
  if (!representation)
    return representation.takeError();
  std::vector<PendingPortablePayload> payloads;
  payloads.push_back({rtlDigest, std::move(rtlBytes)});
  payloads.push_back({constraintDigest, std::move(constraintBytes)});
  return PortableSpatialCoreMaterialization{
      HardwareImplementationDraft{configurationAbi.abi().fabric(),
                                  subject,
                                  configurationAbi.reference(),
                                  std::move(*representation),
                                  std::move(implementationPlatform),
                                  std::move(*interfaces),
                                  {},
                                  {},
                                  {}},
      std::move(payloads)};
}

llvm::Error verifyPortableSpatialCoreMaterialization(
    const HardwareImplementationDraft &expected,
    const HardwareImplementation &actual) {
  if (actual.fabric() != expected.fabric)
    return invalid("portable implementation selects another System");
  if (actual.subject() != expected.subject)
    return invalid("portable implementation selects another SpatialCore");
  if (actual.configurationAbi() != expected.configurationAbi)
    return invalid("portable implementation selects another ConfigurationABI");
  if (!(actual.representationRoot() == expected.representationRoot))
    return invalid("portable implementation has another representation root");
  if (actual.implementationPlatform() != expected.implementationPlatform)
    return invalid("portable implementation has another platform binding");
  if (actual.interfaces() != llvm::ArrayRef(expected.interfaces))
    return invalid("portable implementation has another interface mapping");
  if (actual.activityPoints() != llvm::ArrayRef(expected.activityPoints))
    return invalid("portable implementation has another activity mapping");
  if (!actual.memoryMacroBindings().empty())
    return invalid("portable implementation has memory macro bindings");
  if (!actual.externalImplementationBindings().empty())
    return invalid("portable implementation has external bindings");
  return llvm::Error::success();
}

} // namespace

llvm::Expected<FinalizedHardwareImplementation>
finalizePortableSpatialCoreHardwareImplementation(
    const FinalizedConfigurationABI &configurationAbi,
    fabric::SpatialCoreOccurrenceRef subject,
    std::optional<ArtifactRootReference> implementationPlatform,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  mlir::MLIRContext context;
  context.loadDialect<::dataflow::DataflowDialect, ::fabric::FabricDialect,
                      circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  FabricOperationProviderRegistry providers;
  if (llvm::Error error = registerPortableOperationProviders(providers))
    return std::move(error);
  ExternalImplementationContractCatalog externalContracts;
  return finalizePortableSpatialCoreHardwareImplementation(
      context, configurationAbi, subject, std::move(implementationPlatform),
      providers, externalContracts, artifacts, blobs);
}

llvm::Error verifyPortableSpatialCoreHardwareImplementation(
    const FinalizedConfigurationABI &configurationAbi,
    const FinalizedHardwareImplementation &implementation) {
  mlir::MLIRContext context;
  context.loadDialect<::dataflow::DataflowDialect, ::fabric::FabricDialect,
                      circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  FabricOperationProviderRegistry providers;
  if (llvm::Error error = registerPortableOperationProviders(providers))
    return error;
  ExternalImplementationContractCatalog externalContracts;
  const HardwareImplementation &actual = implementation.implementation();
  auto expected = derivePortableSpatialCoreMaterialization(
      context, configurationAbi, actual.subject(),
      actual.implementationPlatform(), providers, externalContracts);
  if (!expected)
    return expected.takeError();
  return verifyPortableSpatialCoreMaterialization(expected->draft, actual);
}

llvm::Expected<FinalizedHardwareImplementation>
finalizePortableSpatialCoreHardwareImplementation(
    mlir::MLIRContext &context,
    const FinalizedConfigurationABI &configurationAbi,
    fabric::SpatialCoreOccurrenceRef subject,
    std::optional<ArtifactRootReference> implementationPlatform,
    const FabricOperationProviderRegistry &providers,
    const ExternalImplementationContractCatalog &externalContracts,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto materialization = derivePortableSpatialCoreMaterialization(
      context, configurationAbi, subject, std::move(implementationPlatform),
      providers, externalContracts);
  if (!materialization)
    return materialization.takeError();
  for (const PendingPortablePayload &payload : materialization->payloads) {
    auto published = blobs.put(payload.bytes);
    if (!published)
      return published.takeError();
    if (*published != payload.digest)
      return invalid("portable payload publication changed its digest");
  }
  return finalizeHardwareImplementation(std::move(materialization->draft),
                                        externalContracts, artifacts, blobs);
}

} // namespace loom::hardware::rtl
