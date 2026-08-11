#include "DSE/FabricTemplateCandidateGenerator.h"

#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

constexpr llvm::StringLiteral configDescriptor =
    "loom.fabric_template_generator.config.1.0";

constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputSlots = {{
    {CandidateGeneratorOutputSlotRef(0), "fabric", PlanValueRole::CandidateSet,
     &loom::fabric::fabricArtifactSchema, PlanValueCardinality::FiniteSet},
}};

constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 1> workUnits = {{
    {CandidateGeneratorWorkUnitRef(0), "template_expansion"},
}};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_template_generator_invalid: " +
                                     message);
}

llvm::ArrayRef<std::uint8_t> descriptorBytes() {
  return {reinterpret_cast<const std::uint8_t *>(configDescriptor.data()),
          configDescriptor.size()};
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::vector<std::uint8_t> encodeConfig(loom::adg::BuiltinTargetPreset preset) {
  const loom::adg::BuiltinTargetDescriptor &descriptor =
      loom::adg::getBuiltinTargetDescriptor(preset);
  std::vector<std::uint8_t> bytes;
  appendU32(bytes, descriptor.templateIdentity.size());
  bytes.insert(bytes.end(), descriptor.templateIdentity.begin(),
               descriptor.templateIdentity.end());
  appendU32(bytes, descriptor.schemaMajor);
  appendU32(bytes, descriptor.schemaMinor);
  return bytes;
}

llvm::Expected<loom::adg::BuiltinTargetPreset>
decodeConfig(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() < 4)
    return invalid("truncated template descriptor identity length");
  std::uint32_t size = 0;
  for (std::uint8_t byte : bytes.take_front(4))
    size = (size << 8) | byte;
  bytes = bytes.drop_front(4);
  if (size > bytes.size())
    return invalid("truncated template descriptor identity");
  llvm::StringRef identity(reinterpret_cast<const char *>(bytes.data()), size);
  bytes = bytes.drop_front(size);
  if (bytes.size() != 8)
    return invalid("template descriptor version is not canonical");
  std::uint32_t major = 0;
  std::uint32_t minor = 0;
  for (std::uint8_t byte : bytes.take_front(4))
    major = (major << 8) | byte;
  for (std::uint8_t byte : bytes.drop_front(4))
    minor = (minor << 8) | byte;
  for (loom::adg::BuiltinTargetPreset preset :
       {loom::adg::BuiltinTargetPreset::Small,
        loom::adg::BuiltinTargetPreset::Default,
        loom::adg::BuiltinTargetPreset::Large}) {
    const auto &descriptor = loom::adg::getBuiltinTargetDescriptor(preset);
    if (identity == descriptor.templateIdentity &&
        major == descriptor.schemaMajor && minor == descriptor.schemaMinor)
      return preset;
  }
  return invalid(
      "template descriptor is not a registered public Builder template");
}

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted =
      adoptResolvedFabricTemplateConfigView(descriptorBytes(), bytes, digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

const CandidateGeneratorDescriptor descriptor{
    fabricTemplateCandidateGeneratorKind,
    "fabric_template",
    "loom.fabric_template.generator.v1",
    {},
    outputSlots,
    ResolvedDseConfigViewContract{descriptorBytes(), validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
    nullptr,
    ProviderForm::InProcess,
};

llvm::Expected<CandidateGeneratorProviderResult>
invokeProvider(llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
               const ResolvedCandidateGeneratorBinding &binding,
               const ArtifactStore &store, const BlobStore &) {
  if (!inputBindings.empty())
    return invalid("fabric template generator received an input binding");
  auto config = adoptResolvedFabricTemplateConfigView(
      descriptorBytes(), binding.canonicalConfigBytes(),
      binding.configDigest());
  if (!config)
    return config.takeError();
  auto result = loom::adg::buildBuiltinTarget(store, config->preset());
  if (!result)
    return result.takeError();
  std::vector<ArtifactRootReference> outputs;
  std::vector<CandidateGeneratorLineageEdge> lineage;
  outputs.reserve(result->roots().size());
  lineage.reserve(result->roots().size());
  for (const loom::fabric::FinalizedFabricRoot &root : result->roots()) {
    outputs.push_back(root.reference());
    lineage.push_back(CandidateGeneratorLineageEdge{
        CandidateGeneratorLineageEdgeKind::MechanicalDerivation,
        CandidateGeneratorOutputSlotRef(0),
        root.reference(),
        {},
        {}});
  }
  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{
          {{CandidateGeneratorOutputSlotRef(0), std::move(outputs)}},
          std::move(lineage)},
      {{CandidateGeneratorWorkUnitRef(0), 1, 1}}};
}

const CandidateGeneratorProvider provider{
    descriptor.reference(),
    CandidateGeneratorInProcessProvider{invokeProvider}};

} // namespace

llvm::ArrayRef<std::uint8_t> resolvedFabricTemplateConfigSchemaBytes() {
  return descriptorBytes();
}

llvm::Expected<ResolvedFabricTemplateConfigView>
resolveFabricTemplateConfig(loom::adg::BuiltinTargetPreset preset) {
  std::vector<std::uint8_t> bytes = encodeConfig(preset);
  auto digest = computeComponentViewDigest(descriptorBytes(), bytes);
  if (!digest)
    return digest.takeError();
  return ResolvedFabricTemplateConfigView(preset, std::move(bytes), *digest);
}

llvm::Expected<ResolvedFabricTemplateConfigView>
adoptResolvedFabricTemplateConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest) {
  if (schemaDescriptorBytes != descriptorBytes())
    return invalid("config descriptor does not match the exact owner");
  if (llvm::Error error = validateComponentViewDigest(
          schemaDescriptorBytes, canonicalViewBytes, digest))
    return std::move(error);
  auto preset = decodeConfig(canonicalViewBytes);
  if (!preset)
    return preset.takeError();
  std::vector<std::uint8_t> reencoded = encodeConfig(*preset);
  if (llvm::ArrayRef<std::uint8_t>(reencoded) != canonicalViewBytes)
    return invalid("template config does not re-encode to the source bytes");
  return ResolvedFabricTemplateConfigView(*preset, std::move(reencoded),
                                          digest);
}

const CandidateGeneratorDescriptor &
fabricTemplateCandidateGeneratorDescriptor() {
  return descriptor;
}

llvm::Error registerFabricTemplateCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindFabricTemplateCandidateGeneratorInputs() {
  if (llvm::Error error = registerFabricTemplateCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings;
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveFabricTemplateCandidateGeneratorBinding(
    const ResolvedFabricTemplateConfigView &config) {
  if (llvm::Error error = registerFabricTemplateCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

} // namespace loom::dse
