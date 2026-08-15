#include "DSE/PortableSpatialCoreRtlCandidateGenerator.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "Hardware/RTL/CommonSkeleton.h"
#include "Hardware/RTL/PortableProviders.h"
#include "Hardware/RTL/SpatialCoreImplementation.h"
#include "Hardware/RTL/Specialization.h"

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
    "loom.portable_spatial_core_rtl_generator.config.1.0";

enum InputSlot : std::uint32_t {
  SystemInput = 0,
  ConfigurationAbiInput = 1,
};

constexpr std::array<CandidateGeneratorInputSlotDescriptor, 2> inputSlots = {{
    {CandidateGeneratorInputSlotRef(SystemInput), "fabric_system",
     PlanValueRole::CandidateSet, &loom::fabric::fabricArtifactSchema,
     PlanValueCardinality::ExactlyOne},
    {CandidateGeneratorInputSlotRef(ConfigurationAbiInput), "configuration_abi",
     PlanValueRole::CandidateSet, &loom::hardware::configurationAbiSchema,
     PlanValueCardinality::ExactlyOne},
}};

constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputSlots = {{
    {CandidateGeneratorOutputSlotRef(0), "portable_spatial_core_rtl",
     PlanValueRole::CandidateSet, &loom::hardware::hardwareImplementationSchema,
     PlanValueCardinality::FiniteSet},
}};

constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 1> workUnits = {{
    {CandidateGeneratorWorkUnitRef(0), "spatial_core_rtl_derivation"},
}};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "portable_spatial_core_rtl_generator_invalid: " + message);
}

llvm::ArrayRef<std::uint8_t> descriptorBytes() {
  return {reinterpret_cast<const std::uint8_t *>(configDescriptor.data()),
          configDescriptor.size()};
}

llvm::Expected<CandidateGeneratorProviderResult>
unsupportedResult(std::vector<ArtifactRootReference> outputs,
                  std::vector<CandidateGeneratorLineageEdge> lineage,
                  std::uint64_t completed, std::uint64_t total) {
  return CandidateGeneratorProviderResult{
      IncompleteCandidateGeneratorResult{
          CandidateGeneratorIncompleteReason::Unsupported,
          {{CandidateGeneratorOutputSlotRef(0), std::move(outputs)}},
          std::move(lineage)},
      {{CandidateGeneratorWorkUnitRef(0), completed, total}}};
}

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted = adoptResolvedPortableSpatialCoreRtlConfigView(
      descriptorBytes(), bytes, digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

const CandidateGeneratorDescriptor descriptor{
    portableSpatialCoreRtlCandidateGeneratorKind,
    "portable_spatial_core_rtl",
    "loom.portable_spatial_core_rtl.generator.v1",
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
               const ArtifactStore &artifacts, const BlobStore &blobs,
               const CandidateGeneratorInvocationView &) {
  auto config = adoptResolvedPortableSpatialCoreRtlConfigView(
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

  mlir::MLIRContext context;
  context.loadDialect<::dataflow::DataflowDialect, ::fabric::FabricDialect,
                      circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  loom::hardware::rtl::FabricOperationProviderRegistry providers;
  if (llvm::Error error =
          loom::hardware::rtl::registerPortableOperationProviders(providers))
    return std::move(error);
  loom::hardware::ExternalImplementationContractCatalog externalContracts;
  std::vector<ArtifactRootReference> outputs;
  std::vector<CandidateGeneratorLineageEdge> lineage;
  const auto accCores = system->view().accCoreOccurrences();
  outputs.reserve(accCores.size());
  lineage.reserve(accCores.size());
  for (loom::fabric::AccCoreOccurrenceRef accCore : accCores) {
    auto implementation =
        loom::hardware::rtl::finalizePortableSpatialCoreHardwareImplementation(
            context, *configurationAbi,
            loom::fabric::SpatialCoreOccurrenceRef{accCore}, providers,
            externalContracts, artifacts, blobs);
    if (!implementation) {
      bool unsupported = false;
      llvm::Error remainder = llvm::handleErrors(
          implementation.takeError(),
          [&](const loom::hardware::rtl::
                  FabricStructuralLoweringUnsupportedError &) {
            unsupported = true;
          },
          [&](const loom::hardware::rtl::FabricOperationProviderUnsupportedError
                  &) { unsupported = true; });
      if (remainder)
        return std::move(remainder);
      if (unsupported) {
        const std::uint64_t completed = outputs.size();
        return unsupportedResult(std::move(outputs), std::move(lineage),
                                 completed, accCores.size());
      }
      return invalid("portable RTL generation failed without a typed error");
    }
    outputs.push_back(implementation->reference());
    lineage.push_back({CandidateGeneratorLineageEdgeKind::MechanicalDerivation,
                       CandidateGeneratorOutputSlotRef(0),
                       implementation->reference(),
                       {},
                       {}});
  }

  const std::uint64_t work = outputs.size();
  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{
          {{CandidateGeneratorOutputSlotRef(0), std::move(outputs)}},
          std::move(lineage)},
      {{CandidateGeneratorWorkUnitRef(0), work, work}}};
}

const CandidateGeneratorProvider provider{
    descriptor.reference(),
    CandidateGeneratorInProcessProvider{invokeProvider}};

} // namespace

llvm::ArrayRef<std::uint8_t> resolvedPortableSpatialCoreRtlConfigSchemaBytes() {
  return descriptorBytes();
}

llvm::Expected<ResolvedPortableSpatialCoreRtlConfigView>
resolvePortableSpatialCoreRtlConfig() {
  std::vector<std::uint8_t> bytes;
  auto digest = computeComponentViewDigest(descriptorBytes(), bytes);
  if (!digest)
    return digest.takeError();
  return ResolvedPortableSpatialCoreRtlConfigView(std::move(bytes), *digest);
}

llvm::Expected<ResolvedPortableSpatialCoreRtlConfigView>
adoptResolvedPortableSpatialCoreRtlConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest) {
  if (schemaDescriptorBytes != descriptorBytes())
    return invalid("config descriptor does not match the exact owner");
  if (!canonicalViewBytes.empty())
    return invalid("portable SpatialCore RTL config must be empty");
  if (llvm::Error error = validateComponentViewDigest(
          schemaDescriptorBytes, canonicalViewBytes, digest))
    return std::move(error);
  return ResolvedPortableSpatialCoreRtlConfigView({}, digest);
}

const CandidateGeneratorDescriptor &
portableSpatialCoreRtlCandidateGeneratorDescriptor() {
  return descriptor;
}

llvm::Error registerPortableSpatialCoreRtlCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindPortableSpatialCoreRtlCandidateGeneratorInputs(
    const ArtifactRootReference &system,
    const ArtifactRootReference &configurationAbi) {
  if (llvm::Error error = registerPortableSpatialCoreRtlCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings = {
      {CandidateGeneratorInputSlotRef(SystemInput), {system}},
      {CandidateGeneratorInputSlotRef(ConfigurationAbiInput),
       {configurationAbi}},
  };
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolvePortableSpatialCoreRtlCandidateGeneratorBinding(
    const ResolvedPortableSpatialCoreRtlConfigView &config) {
  if (llvm::Error error = registerPortableSpatialCoreRtlCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

} // namespace loom::dse
