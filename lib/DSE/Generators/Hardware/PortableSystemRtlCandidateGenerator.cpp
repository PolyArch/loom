#include "DSE/PortableSystemRtlCandidateGenerator.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "Hardware/RTL/CommonSkeleton.h"
#include "Hardware/RTL/PortableProviders.h"
#include "Hardware/RTL/Specialization.h"
#include "Hardware/RTL/SystemImplementation.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/IR/MLIRContext.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

constexpr llvm::StringLiteral configDescriptor =
    "loom.portable_system_rtl_generator.config.1.0";

enum InputSlot : std::uint32_t {
  SystemInput = 0,
  ConfigurationAbiInput = 1,
  InterconnectImplementationInput = 2,
};

constexpr std::array<CandidateGeneratorInputSlotDescriptor, 3> inputSlots = {{
    {CandidateGeneratorInputSlotRef(SystemInput), "fabric_system",
     PlanValueRole::CandidateSet, &loom::fabric::fabricArtifactSchema,
     PlanValueCardinality::ExactlyOne},
    {CandidateGeneratorInputSlotRef(ConfigurationAbiInput), "configuration_abi",
     PlanValueRole::CandidateSet, &loom::hardware::configurationAbiSchema,
     PlanValueCardinality::ExactlyOne},
    {CandidateGeneratorInputSlotRef(InterconnectImplementationInput),
     "interconnect_implementation", PlanValueRole::CandidateSet,
     &loom::fabric::fabricArtifactSchema, PlanValueCardinality::FiniteSet},
}};

constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputSlots = {{
    {CandidateGeneratorOutputSlotRef(0), "portable_system_rtl",
     PlanValueRole::CandidateSet, &loom::hardware::hardwareImplementationSchema,
     PlanValueCardinality::ExactlyOne},
}};

constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 1> workUnits = {{
    {CandidateGeneratorWorkUnitRef(0), "portable_system_rtl_derivation"},
}};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_system_rtl_generator_invalid: " +
                                     message);
}

llvm::ArrayRef<std::uint8_t> descriptorBytes() {
  return {reinterpret_cast<const std::uint8_t *>(configDescriptor.data()),
          configDescriptor.size()};
}

llvm::Expected<CandidateGeneratorProviderResult> unsupportedResult() {
  return CandidateGeneratorProviderResult{
      IncompleteCandidateGeneratorResult{
          CandidateGeneratorIncompleteReason::Unsupported,
          {{CandidateGeneratorOutputSlotRef(0), {}}},
          {}},
      {{CandidateGeneratorWorkUnitRef(0), 1, 1}}};
}

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted = adoptResolvedPortableSystemRtlConfigView(descriptorBytes(),
                                                          bytes, digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

const CandidateGeneratorDescriptor descriptor{
    portableSystemRtlCandidateGeneratorKind,
    "portable_system_rtl",
    "loom.portable_system_rtl.generator.v1",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{descriptorBytes(), validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
    nullptr,
    ProviderForm::InProcess,
};

llvm::Expected<CandidateGeneratorProviderResult>
invokeProvider(llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
               const ResolvedCandidateGeneratorBinding &binding,
               const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto config = adoptResolvedPortableSystemRtlConfigView(
      descriptorBytes(), binding.canonicalConfigBytes(),
      binding.configDigest());
  if (!config)
    return config.takeError();

  auto system = loom::fabric::importEntireFabricRoot(
      inputs[SystemInput].artifacts.front(), artifacts);
  if (!system)
    return system.takeError();
  if (system->view().rootKind() != loom::fabric::FabricRootKind::System)
    return invalid("fabric input is not a complete System root");

  auto configurationAbi = loom::hardware::importConfigurationABI(
      inputs[ConfigurationAbiInput].artifacts.front(), artifacts);
  if (!configurationAbi)
    return configurationAbi.takeError();
  if (configurationAbi->abi().fabric() != system->reference())
    return invalid("ConfigurationABI describes another Fabric System");

  for (const ArtifactRootReference &reference :
       inputs[InterconnectImplementationInput].artifacts) {
    auto interconnect =
        loom::fabric::importEntireFabricRoot(reference, artifacts);
    if (!interconnect)
      return interconnect.takeError();
    if (interconnect->view().rootKind() !=
        loom::fabric::FabricRootKind::InterconnectImplementation)
      return invalid("interconnect input has the wrong Fabric root kind");
  }

  mlir::MLIRContext context;
  context.loadDialect<::dataflow::DataflowDialect, ::fabric::FabricDialect,
                      circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  loom::hardware::rtl::FabricOperationProviderRegistry providers;
  if (llvm::Error error =
          loom::hardware::rtl::registerPortableOperationProviders(providers))
    return std::move(error);
  loom::hardware::ExternalImplementationContractCatalog externalContracts;
  auto implementation =
      loom::hardware::rtl::finalizePortableSystemHardwareImplementation(
          context, *configurationAbi, providers, externalContracts, artifacts,
          blobs, inputs[InterconnectImplementationInput].artifacts);
  if (!implementation) {
    bool unsupported = false;
    llvm::Error remainder = llvm::handleErrors(
        implementation.takeError(),
        [&](const loom::hardware::rtl::FabricStructuralLoweringUnsupportedError
                &) { unsupported = true; },
        [&](const loom::hardware::rtl::FabricOperationProviderUnsupportedError
                &) { unsupported = true; });
    if (remainder)
      return std::move(remainder);
    if (unsupported)
      return unsupportedResult();
    return invalid("portable RTL generation failed without a typed error");
  }

  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{
          {{CandidateGeneratorOutputSlotRef(0), {implementation->reference()}}},
          {{CandidateGeneratorLineageEdgeKind::MechanicalDerivation,
            CandidateGeneratorOutputSlotRef(0),
            implementation->reference(),
            {},
            {}}}},
      {{CandidateGeneratorWorkUnitRef(0), 1, 1}}};
}

const CandidateGeneratorProvider provider{
    descriptor.reference(),
    CandidateGeneratorInProcessProvider{invokeProvider}};

} // namespace

llvm::ArrayRef<std::uint8_t> resolvedPortableSystemRtlConfigSchemaBytes() {
  return descriptorBytes();
}

llvm::Expected<ResolvedPortableSystemRtlConfigView>
resolvePortableSystemRtlConfig() {
  std::vector<std::uint8_t> bytes;
  auto digest = computeComponentViewDigest(descriptorBytes(), bytes);
  if (!digest)
    return digest.takeError();
  return ResolvedPortableSystemRtlConfigView(std::move(bytes), *digest);
}

llvm::Expected<ResolvedPortableSystemRtlConfigView>
adoptResolvedPortableSystemRtlConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest) {
  if (schemaDescriptorBytes != descriptorBytes())
    return invalid("config descriptor does not match the exact owner");
  if (!canonicalViewBytes.empty())
    return invalid("portable System RTL config must be empty");
  if (llvm::Error error = validateComponentViewDigest(
          schemaDescriptorBytes, canonicalViewBytes, digest))
    return std::move(error);
  return ResolvedPortableSystemRtlConfigView({}, digest);
}

const CandidateGeneratorDescriptor &
portableSystemRtlCandidateGeneratorDescriptor() {
  return descriptor;
}

llvm::Error registerPortableSystemRtlCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindPortableSystemRtlCandidateGeneratorInputs(
    const ArtifactRootReference &system,
    const ArtifactRootReference &configurationAbi,
    llvm::ArrayRef<ArtifactRootReference> interconnectImplementations) {
  if (llvm::Error error = registerPortableSystemRtlCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings = {
      {CandidateGeneratorInputSlotRef(SystemInput), {system}},
      {CandidateGeneratorInputSlotRef(ConfigurationAbiInput),
       {configurationAbi}},
      {CandidateGeneratorInputSlotRef(InterconnectImplementationInput),
       interconnectImplementations.vec()},
  };
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolvePortableSystemRtlCandidateGeneratorBinding(
    const ResolvedPortableSystemRtlConfigView &config) {
  if (llvm::Error error = registerPortableSystemRtlCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

} // namespace loom::dse
