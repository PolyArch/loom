#include "DSE/PortableSpatialCoreRtlCandidateGenerator.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "Hardware/RTL/SpatialCoreImplementation.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

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
  ImplementationPlatformInput = 2,
};

constexpr std::array<CandidateGeneratorInputSlotDescriptor, 3> inputSlots = {{
    {CandidateGeneratorInputSlotRef(SystemInput), "fabric_system",
     PlanValueRole::CandidateSet, &loom::fabric::fabricArtifactSchema,
     PlanValueCardinality::ExactlyOne},
    {CandidateGeneratorInputSlotRef(ConfigurationAbiInput), "configuration_abi",
     PlanValueRole::CandidateSet, &loom::hardware::configurationAbiSchema,
     PlanValueCardinality::ExactlyOne},
    {CandidateGeneratorInputSlotRef(ImplementationPlatformInput),
     "implementation_platform", PlanValueRole::CandidateSet,
     &loom::platform::implementationPlatformSchema,
     PlanValueCardinality::ZeroOrOne},
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
incompleteResult(CandidateGeneratorIncompleteReason reason,
                 std::vector<ArtifactRootReference> outputs,
                 std::vector<CandidateGeneratorLineageEdge> lineage,
                 std::uint64_t planned, std::uint64_t consumed) {
  return CandidateGeneratorProviderResult{
      IncompleteCandidateGeneratorResult{
          reason,
          {{CandidateGeneratorOutputSlotRef(0), std::move(outputs)}},
          std::move(lineage)},
      {{CandidateGeneratorWorkUnitRef(0), planned, consumed}}};
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
    "loom.portable_spatial_core_rtl.generator.v4",
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
               const CandidateGeneratorInvocationView &invocation) {
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

  std::optional<ArtifactRootReference> implementationPlatform;
  if (!inputs[ImplementationPlatformInput].artifacts.empty()) {
    auto imported = loom::platform::importImplementationPlatform(
        inputs[ImplementationPlatformInput].artifacts.front(), artifacts);
    if (!imported)
      return imported.takeError();
    implementationPlatform = imported->reference();
  }

  std::vector<ArtifactRootReference> outputs;
  std::vector<CandidateGeneratorLineageEdge> lineage;
  const auto accCores = system->view().accCoreOccurrences();
  const std::uint64_t planned = accCores.size();
  const std::optional<std::uint64_t> maximumOutputs =
      invocation.maximumOutputArtifacts(CandidateGeneratorOutputSlotRef(0));
  outputs.reserve(accCores.size());
  lineage.reserve(accCores.size());
  for (loom::fabric::AccCoreOccurrenceRef accCore : accCores) {
    if (invocation.stopRequested()) {
      const std::uint64_t consumed = outputs.size();
      return incompleteResult(
          CandidateGeneratorIncompleteReason::CancelledOrTimeout,
          std::move(outputs), std::move(lineage), planned, consumed);
    }
    if (maximumOutputs && outputs.size() >= *maximumOutputs) {
      const std::uint64_t consumed = outputs.size();
      return incompleteResult(
          CandidateGeneratorIncompleteReason::SemanticLimitReached,
          std::move(outputs), std::move(lineage), planned, consumed);
    }
    auto implementation =
        loom::hardware::rtl::finalizePortableSpatialCoreHardwareImplementation(
            *configurationAbi, loom::fabric::SpatialCoreOccurrenceRef{accCore},
            implementationPlatform, artifacts, blobs);
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
        return incompleteResult(CandidateGeneratorIncompleteReason::Unsupported,
                                std::move(outputs), std::move(lineage), planned,
                                completed + 1);
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
    const ArtifactRootReference &configurationAbi,
    std::optional<ArtifactRootReference> implementationPlatform) {
  if (llvm::Error error = registerPortableSpatialCoreRtlCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings = {
      {CandidateGeneratorInputSlotRef(SystemInput), {system}},
      {CandidateGeneratorInputSlotRef(ConfigurationAbiInput),
       {configurationAbi}},
      {CandidateGeneratorInputSlotRef(ImplementationPlatformInput),
       implementationPlatform
           ? std::vector<ArtifactRootReference>{*implementationPlatform}
           : std::vector<ArtifactRootReference>{}},
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
