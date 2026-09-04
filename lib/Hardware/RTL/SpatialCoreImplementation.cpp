#include "Hardware/RTL/SpatialCoreImplementation.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Common/DiagnosticVerbosity.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Hardware/Implementation/FabricModel.h"
#include "Hardware/Implementation/ImplementationRepresentationRoot.h"
#include "Hardware/RTL/CommonSkeleton.h"
#include "Hardware/RTL/MaterializationDiagnostics.h"
#include "Hardware/RTL/PortableProviders.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::hardware::rtl {
namespace {

inline constexpr llvm::StringLiteral portableSpatialCoreTop = "loom_module";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "rtl_spatial_core_implementation_invalid: " +
                                     message);
}

std::string
materializationKey(const FinalizedConfigurationABI &configurationAbi,
                   fabric::SpatialCoreOccurrenceRef subject) {
  return formatArtifactIdentityHex(configurationAbi.reference().artifact) +
         ":" + llvm::toHex(fabric::canonicalFabricBytes(subject), true);
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

/// The top port named by one interface locator, or an error when the locator
/// is not a port of the portable top module.
llvm::Expected<std::string>
topPortName(const ImplementationInterface &interface, llvm::StringRef role) {
  constexpr llvm::StringLiteral topPrefix = "loom_module.";
  if (interface.representationLocator.kind != RepresentationObjectKind::Port)
    return invalid("SpatialCore RTL has no exact " + role + " port");
  llvm::StringRef port = interface.representationLocator.canonicalName;
  if (!port.consume_front(topPrefix) || port.empty())
    return invalid("SpatialCore " + role +
                   " locator is outside the top module");
  return port.str();
}

llvm::Expected<std::string>
deriveGenerationConstraint(const FinalizedConfigurationABI &configurationAbi,
                           llvm::ArrayRef<ImplementationInterface> interfaces) {
  auto binding = deriveSpatialCoreClockBinding(configurationAbi, interfaces);
  if (!binding)
    return binding.takeError();
  return renderCreateClockConstraint(binding->clock, binding->clockPort);
}

struct PendingPortablePayload final {
  BlobDigest digest;
  std::vector<std::uint8_t> bytes;
};

struct PortableSpatialCoreMaterialization final {
  HardwareImplementationDraft draft;
  std::vector<PendingPortablePayload> payloads;
  RtlModuleGraphProjection moduleGraph;
};

class Sha256Ostream final : public llvm::raw_ostream {
public:
  Sha256Ostream() {
    auto digest = BlobDigestBuilder::create();
    if (!digest) {
      error_ = llvm::toString(digest.takeError());
      return;
    }
    digest_.emplace(std::move(*digest));
  }

  llvm::Expected<BlobDigest> finish() {
    flush();
    if (!error_.empty())
      return invalid(error_);
    if (!digest_)
      return invalid("RTL digest stream was not initialized");
    return digest_->finish();
  }

  std::uint64_t byteCount() const { return position_; }

private:
  void write_impl(const char *data, std::size_t size) override {
    if (error_.empty() && digest_)
      if (llvm::Error error = digest_->update(llvm::ArrayRef<std::uint8_t>(
              reinterpret_cast<const std::uint8_t *>(data), size)))
        error_ = llvm::toString(std::move(error));
    position_ += size;
  }

  std::uint64_t current_pos() const override { return position_; }

  std::optional<BlobDigestBuilder> digest_;
  std::string error_;
  std::uint64_t position_ = 0;
};

llvm::Expected<PortableSpatialCoreMaterialization>
derivePortableSpatialCoreMaterialization(
    mlir::MLIRContext &context,
    const FinalizedConfigurationABI &configurationAbi,
    fabric::SpatialCoreOccurrenceRef subject,
    std::optional<ArtifactRootReference> implementationPlatform,
    const FabricOperationProviderRegistry &providers,
    const ExternalImplementationContractCatalog &externalContracts,
    const BlobStore *publishBlobs) {
  const std::string diagnosticKey =
      materializationKey(configurationAbi, subject);
  auto skeleton = [&]() -> llvm::Expected<ModuleRootCirctSkeleton> {
    RtlMaterializationStageTracker stage("skeleton", diagnosticKey);
    auto result = buildModuleRootCirctSkeleton(context, subject,
                                               configurationAbi, diagnosticKey);
    if (result)
      stage.finish(*result->module);
    return result;
  }();
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
      providers, externalContracts, nullptr, diagnosticKey);
  if (!specialization)
    return specialization.takeError();
  if (!specialization->payloads.empty() ||
      !specialization->activityPoints.empty() ||
      !specialization->externalImplementationBindings.empty())
    return invalid("portable provider returned non-self-contained material");
  if (llvm::Error error = verifySpecializedCirctModule(*skeleton->module))
    return std::move(error);

  std::optional<BlobDigest> rtlDigest;
  RtlModuleGraphProjection moduleGraph;
  const RtlModuleGraphCapture graphCapture{portableSpatialCoreTop,
                                           &moduleGraph};
  if (publishBlobs) {
    RtlMaterializationStageTracker publication(
        "rtl_blob_publication", diagnosticKey, *skeleton->module);
    auto published = publishBlobs->putGenerated([&](llvm::raw_ostream &output) {
      return lowerAndExportSpecializedSystemVerilog(
          *skeleton->module, output, diagnosticKey, graphCapture);
    });
    if (!published)
      return published.takeError();
    rtlDigest = published->digest;
    publication.finish(*skeleton->module, published->logicalByteCount);
  } else {
    RtlMaterializationStageTracker publication(
        "rtl_digest_replay", diagnosticKey, *skeleton->module);
    Sha256Ostream output;
    if (llvm::Error error = lowerAndExportSpecializedSystemVerilog(
            *skeleton->module, output, diagnosticKey, graphCapture))
      return std::move(error);
    auto digest = output.finish();
    if (!digest)
      return digest.takeError();
    rtlDigest = *digest;
    publication.finish(*skeleton->module, output.byteCount());
  }
  if (!moduleGraph.sourceDigest || !rtlDigest ||
      *moduleGraph.sourceDigest != *rtlDigest)
    return invalid("RTL module graph changed the source content identity");
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
      {{PayloadRole::RtlSource, "rtl/loom_spatial_core.sv", *rtlDigest},
       {PayloadRole::GenerationConstraint, "constraints/loom_spatial_core.sdc",
        constraintDigest}});
  if (!representation)
    return representation.takeError();
  std::vector<PendingPortablePayload> payloads;
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
      std::move(payloads), std::move(moduleGraph)};
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

llvm::Expected<SpatialCoreClockBinding> deriveSpatialCoreClockBinding(
    const FinalizedConfigurationABI &configurationAbi,
    llvm::ArrayRef<ImplementationInterface> interfaces) {
  const ImplementationInterface *clockInterface = nullptr;
  const ImplementationInterface *resetInterface = nullptr;
  for (const ImplementationInterface &interface : interfaces) {
    if (std::holds_alternative<ImplementationClockInterfaceRef>(
            interface.semanticRef)) {
      if (clockInterface)
        return invalid("SpatialCore RTL has more than one clock interface");
      clockInterface = &interface;
    } else if (std::holds_alternative<ImplementationResetInterfaceRef>(
                   interface.semanticRef)) {
      if (resetInterface)
        return invalid("SpatialCore RTL has more than one reset interface");
      resetInterface = &interface;
    }
  }
  if (!clockInterface)
    return invalid("SpatialCore RTL has no exact clock port");
  auto clockPort = topPortName(*clockInterface, "clock");
  if (!clockPort)
    return clockPort.takeError();
  std::optional<std::string> resetPort;
  if (resetInterface) {
    auto port = topPortName(*resetInterface, "reset");
    if (!port)
      return port.takeError();
    resetPort = std::move(*port);
  }

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
  return SpatialCoreClockBinding{std::move(*clockPort), std::move(resetPort),
                                 *clock};
}

std::string
renderCreateClockConstraint(const fabric::ClockDomainContractRecord &clock,
                            llvm::StringRef clockPort) {
  const std::string period = fixedDecimal(clock.periodFs(), 6);
  const std::string rising = fixedDecimal(clock.phaseFs(), 6);
  const unsigned __int128 fallingFemtosecondHalves =
      static_cast<unsigned __int128>(clock.phaseFs()) * 2 + clock.periodFs();
  const std::string falling = fixedDecimal(fallingFemtosecondHalves * 5, 7);
  return "create_clock -name loom_clock -period " + period + " -waveform {" +
         rising + " " + falling + "} [get_ports {" + clockPort.str() + "}]\n";
}

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
      actual.implementationPlatform(), providers, externalContracts, nullptr);
  if (!expected)
    return expected.takeError();
  return verifyPortableSpatialCoreMaterialization(expected->draft, actual);
}

llvm::Expected<std::optional<RtlModuleGraphProjection>>
projectPortableSpatialCoreRtlModuleGraph(
    const FinalizedConfigurationABI &configurationAbi,
    const FinalizedHardwareImplementation &implementation) {
  const HardwareImplementation &actual = implementation.implementation();
  if (actual.configurationAbi() != configurationAbi.reference() ||
      actual.fabric() != configurationAbi.abi().fabric())
    return std::nullopt;

  mlir::MLIRContext context;
  context.loadDialect<::dataflow::DataflowDialect, ::fabric::FabricDialect,
                      circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  FabricOperationProviderRegistry providers;
  if (llvm::Error error = registerPortableOperationProviders(providers))
    return std::move(error);
  ExternalImplementationContractCatalog externalContracts;
  auto expected = derivePortableSpatialCoreMaterialization(
      context, configurationAbi, actual.subject(),
      actual.implementationPlatform(), providers, externalContracts, nullptr);
  if (!expected)
    return expected.takeError();
  if (llvm::Error mismatch =
          verifyPortableSpatialCoreMaterialization(expected->draft, actual)) {
    if (diagnosticVerbosityEnabled(DiagnosticVerbosity::Decision))
      llvm::errs() << "rtl_spatial_core_implementation: canonical "
                      "materialization mismatch: "
                   << llvm::toString(std::move(mismatch)) << '\n';
    else
      llvm::consumeError(std::move(mismatch));
    return std::nullopt;
  }
  return std::optional<RtlModuleGraphProjection>(
      std::move(expected->moduleGraph));
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
      providers, externalContracts, &blobs);
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
