#include "DSE/FabricTemplateCandidateGenerator.h"

#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/CompositeFuMining.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "HardwareTopologyQuality.h"

#include "llvm/ADT/STLExtras.h"

#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

constexpr llvm::StringLiteral configDescriptor =
    "loom.fabric_template_generator.config.7.4";

// The Dataflow a mined composite FU selection names. A template built from
// the catalog alone binds nothing here, so the slot admits zero or one.
constexpr std::array<CandidateGeneratorInputSlotDescriptor, 1> inputSlots = {{
    {CandidateGeneratorInputSlotRef(0), "dataflow", PlanValueRole::CandidateSet,
     &::dataflow::canonicalDataflowSchema, PlanValueCardinality::ZeroOrOne},
}};

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

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::uint32_t specialMathCapabilityProfileWireTag(
    loom::adg::BuiltinSpecialMathCapabilityProfile profile) {
  switch (profile) {
  case loom::adg::BuiltinSpecialMathCapabilityProfile::FullCatalog:
    return 0;
  case loom::adg::BuiltinSpecialMathCapabilityProfile::PortableProviderClosed:
    return 1;
  }
  llvm_unreachable("invalid builtin special-math capability profile");
}

std::optional<loom::adg::BuiltinSpecialMathCapabilityProfile>
specialMathCapabilityProfileFromWireTag(std::uint32_t tag) {
  switch (tag) {
  case 0:
    return loom::adg::BuiltinSpecialMathCapabilityProfile::FullCatalog;
  case 1:
    return loom::adg::BuiltinSpecialMathCapabilityProfile::
        PortableProviderClosed;
  default:
    return std::nullopt;
  }
}

std::vector<std::uint8_t>
encodeConfig(const loom::adg::BuiltinTargetScale &scale,
             const std::optional<MinedCompositeFuSelection> &mined) {
  const loom::adg::BuiltinTargetDescriptor &descriptor =
      loom::adg::builtinCoverageTarget;
  std::vector<std::uint8_t> bytes;
  appendU32(bytes, descriptor.templateIdentity.size());
  bytes.insert(bytes.end(), descriptor.templateIdentity.begin(),
               descriptor.templateIdentity.end());
  appendU32(bytes, descriptor.schemaMajor);
  appendU32(bytes, descriptor.schemaMinor);
  appendU32(bytes, scale.accCoreCount);
  appendU32(bytes, scale.meshDimension);
  appendU32(bytes, scale.spatialMeshLanesPerDirection);
  appendU32(bytes, scale.temporalMeshLanesPerDirection);
  appendU32(bytes, scale.spatialPeCount);
  appendU32(bytes, scale.temporalPeCount);
  const auto appendOccurrences = [&](const auto &occurrences) {
    appendU32(bytes, occurrences.dedicatedScalarAdd);
    appendU32(bytes, occurrences.mac);
    appendU32(bytes, occurrences.vectorCompute);
    appendU32(bytes, occurrences.loopControl);
    appendU32(bytes, occurrences.tokenControl);
    appendU32(bytes, occurrences.vectorAdapter);
    appendU32(bytes, occurrences.vectorStructural);
    appendU32(bytes, occurrences.specialMath);
  };
  appendOccurrences(scale.spatialFuOccurrences);
  appendOccurrences(scale.temporalFuOccurrences);
  appendU32(bytes, scale.spatialMemoryCount);
  appendU32(bytes, scale.temporalMemoryCount);
  appendU32(bytes, scale.temporalResidentContexts);
  appendU32(bytes, static_cast<std::uint32_t>(scale.localMemoryPortVariant));
  appendU32(bytes, scale.crossScheduleBoundaryLanesPerTemporalPe);
  appendU32(bytes, scale.gatewayCount);
  appendU64(bytes, scale.memoryCapacityBytes);
  appendU32(bytes, scale.interconnectFifoDepth);
  appendU32(bytes,
            static_cast<std::uint32_t>(scale.interconnectFifoQueueDiscipline));
  appendU32(bytes, scale.interconnectFifoReservedChannels);
  appendU32(bytes, specialMathCapabilityProfileWireTag(
                       scale.specialMathCapabilityProfile));
  appendU64(bytes, scale.privateCaches.instructionCoreCacheBytes);
  appendU64(bytes, scale.privateCaches.spatialMemoryCacheBytes);
  appendU32(bytes, scale.privateCaches.lineBytes);
  appendU32(bytes, scale.privateCaches.associativity);
  appendU32(bytes, scale.privateCaches.hitLatencyCycles);
  appendU32(bytes, scale.privateCaches.inOrderMissStatusEntries);
  appendU32(bytes, scale.privateCaches.outOfOrderMissStatusEntries);
  appendU32(bytes, scale.memoryOperationIssueDepth);
  appendU32(bytes, mined ? mined->templates.size() : 0);
  if (mined) {
    bytes.insert(bytes.end(), mined->dataflow.bytes().begin(),
                 mined->dataflow.bytes().end());
    for (const MinedCompositeFuSelectionEntry &entry : mined->templates) {
      appendU32(bytes, entry.shapeKey.size());
      bytes.insert(bytes.end(), entry.shapeKey.begin(), entry.shapeKey.end());
      appendU32(bytes, entry.occurrences);
    }
  }
  return bytes;
}

struct DecodedConfig final {
  loom::adg::BuiltinTargetScale scale;
  std::optional<MinedCompositeFuSelection> mined;
};

llvm::Expected<DecodedConfig> decodeConfig(llvm::ArrayRef<std::uint8_t> bytes) {
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
  std::uint32_t major = 0;
  std::uint32_t minor = 0;
  for (std::uint8_t byte : bytes.take_front(4))
    major = (major << 8) | byte;
  for (std::uint8_t byte : bytes.slice(4, 4))
    minor = (minor << 8) | byte;
  bytes = bytes.drop_front(8);
  const auto readU32 = [&bytes]() {
    std::uint32_t value = 0;
    for (std::uint8_t byte : bytes.take_front(4))
      value = (value << 8) | byte;
    bytes = bytes.drop_front(4);
    return value;
  };
  const auto readOccurrences = [&]() {
    return loom::adg::BuiltinFuOccurrenceCounts{readU32(), readU32(), readU32(),
                                                readU32(), readU32(), readU32(),
                                                readU32(), readU32()};
  };
  loom::adg::BuiltinTargetScale scale{};
  scale.accCoreCount = readU32();
  scale.meshDimension = readU32();
  scale.spatialMeshLanesPerDirection = readU32();
  scale.temporalMeshLanesPerDirection = readU32();
  scale.spatialPeCount = readU32();
  scale.temporalPeCount = readU32();
  scale.spatialFuOccurrences = readOccurrences();
  scale.temporalFuOccurrences = readOccurrences();
  scale.spatialMemoryCount = readU32();
  scale.temporalMemoryCount = readU32();
  scale.temporalResidentContexts = readU32();
  scale.localMemoryPortVariant =
      static_cast<loom::adg::LocalMemoryPortVariant>(readU32());
  scale.crossScheduleBoundaryLanesPerTemporalPe = readU32();
  scale.gatewayCount = readU32();
  for (std::uint8_t byte : bytes.take_front(8))
    scale.memoryCapacityBytes = (scale.memoryCapacityBytes << 8) | byte;
  bytes = bytes.drop_front(8);
  scale.interconnectFifoDepth = readU32();
  scale.interconnectFifoQueueDiscipline =
      static_cast<::fabric::FifoQueueDiscipline>(readU32());
  scale.interconnectFifoReservedChannels = readU32();
  auto specialMathProfile =
      specialMathCapabilityProfileFromWireTag(readU32());
  if (!specialMathProfile)
    return invalid("template special-math capability profile is invalid");
  scale.specialMathCapabilityProfile = *specialMathProfile;
  const auto readU64 = [&bytes]() {
    std::uint64_t value = 0;
    for (std::uint8_t byte : bytes.take_front(8))
      value = (value << 8) | byte;
    bytes = bytes.drop_front(8);
    return value;
  };
  scale.privateCaches.instructionCoreCacheBytes = readU64();
  scale.privateCaches.spatialMemoryCacheBytes = readU64();
  scale.privateCaches.lineBytes = readU32();
  scale.privateCaches.associativity = readU32();
  scale.privateCaches.hitLatencyCycles = readU32();
  scale.privateCaches.inOrderMissStatusEntries = readU32();
  scale.privateCaches.outOfOrderMissStatusEntries = readU32();
  scale.memoryOperationIssueDepth = readU32();
  if (!loom::adg::isValidBuiltinTargetScale(scale))
    return invalid("template base scale is invalid or an FU occurrence count "
                   "exceeds its schedule-local PE count");
  if (bytes.size() < 4)
    return invalid("truncated mined composite FU selection");
  const std::uint32_t minedCount = readU32();
  std::optional<MinedCompositeFuSelection> mined;
  if (minedCount != 0) {
    if (bytes.size() < ArtifactIdentity::byteSize)
      return invalid("truncated mined composite FU Dataflow identity");
    auto dataflow =
        ArtifactIdentity::fromBytes(bytes.take_front(ArtifactIdentity::byteSize));
    if (!dataflow)
      return dataflow.takeError();
    bytes = bytes.drop_front(ArtifactIdentity::byteSize);
    MinedCompositeFuSelection selection{*dataflow, {}};
    selection.templates.reserve(minedCount);
    for (std::uint32_t entry = 0; entry != minedCount; ++entry) {
      if (bytes.size() < 4)
        return invalid("truncated mined composite FU shape key");
      const std::uint32_t keySize = readU32();
      if (keySize == 0 || keySize > bytes.size())
        return invalid("mined composite FU shape key is empty or truncated");
      std::vector<std::uint8_t> key(bytes.take_front(keySize).begin(),
                                    bytes.take_front(keySize).end());
      bytes = bytes.drop_front(keySize);
      if (bytes.size() < 4)
        return invalid("truncated mined composite FU occurrence count");
      const std::uint32_t occurrences = readU32();
      if (occurrences == 0 || occurrences > scale.spatialPeCount)
        return invalid("a mined composite FU occurrence count is zero or "
                       "exceeds the Spatial PE count");
      if (!selection.templates.empty() &&
          !(selection.templates.back().shapeKey < key))
        return invalid("mined composite FU shape keys are not canonical");
      selection.templates.push_back({std::move(key), occurrences});
    }
    mined = std::move(selection);
  }
  if (!bytes.empty())
    return invalid("template descriptor and scale are not canonical");
  const auto *descriptor =
      loom::adg::findBuiltinTargetDescriptor(identity, major, minor);
  if (descriptor)
    return DecodedConfig{scale, std::move(mined)};
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
    "loom.fabric_template.generator.v7",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{descriptorBytes(), validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
    nullptr,
    ProviderForm::InProcess,
};

/// Re-derives the FU structure of every selected mined template from the exact
/// Dataflow the selection names. The config carries only the selection, so the
/// miner and the canonical capability derivation remain the sole owners of
/// that structure and a config can never describe an FU they would not.
llvm::Expected<std::vector<loom::adg::BuiltinCompositeFuPlacement>>
resolveMinedCompositeFus(
    const std::optional<MinedCompositeFuSelection> &selection,
    const CandidateGeneratorInputBinding &dataflowBinding,
    const ArtifactStore &store) {
  std::vector<loom::adg::BuiltinCompositeFuPlacement> placements;
  if (!selection) {
    if (!dataflowBinding.artifacts.empty())
      return invalid("fabric template generator bound a Dataflow without a "
                     "mined composite FU selection");
    return placements;
  }
  if (dataflowBinding.artifacts.size() != 1)
    return invalid("a mined composite FU selection requires its exact "
                   "Dataflow input");
  if (dataflowBinding.artifacts.front().artifact != selection->dataflow)
    return invalid("the bound Dataflow is not the one the mined composite FU "
                   "selection names");
  auto program = ::dataflow::importCanonicalDataflow(
      dataflowBinding.artifacts.front(), store);
  if (!program)
    return program.takeError();
  std::vector<::dataflow::GraphRef> graphs;
  for (const ::dataflow::CanonicalGraphView &graph : program->view().graphs())
    graphs.push_back(graph.ref);
  auto candidates = mineCompositeFuCandidates(program->view(), graphs);
  if (!candidates)
    return candidates.takeError();
  placements.reserve(selection->templates.size());
  for (const MinedCompositeFuSelectionEntry &entry : selection->templates) {
    const auto found = llvm::find_if(
        *candidates, [&](const CompositeFuCandidate &candidate) {
          return candidate.canonicalKey == entry.shapeKey;
        });
    if (found == candidates->end())
      return invalid("a selected mined composite FU shape is not one this "
                     "Dataflow mines");
    auto spec = deriveCompositeFuTemplate(program->view(), *found);
    if (!spec)
      return spec.takeError();
    placements.push_back({std::move(*spec), entry.occurrences});
  }
  return placements;
}

llvm::Expected<CandidateGeneratorProviderResult>
invokeProvider(llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
               const ResolvedCandidateGeneratorBinding &binding,
               const ArtifactStore &store, const BlobStore &,
               const CandidateGeneratorInvocationView &) {
  if (inputBindings.size() != 1)
    return invalid("fabric template generator input bindings are not dense");
  auto config = adoptResolvedFabricTemplateConfigView(
      descriptorBytes(), binding.canonicalConfigBytes(),
      binding.configDigest());
  if (!config)
    return config.takeError();
  auto placements = resolveMinedCompositeFus(config->minedCompositeFus(),
                                             inputBindings.front(), store);
  if (!placements)
    return placements.takeError();
  auto result =
      loom::adg::buildBuiltinTarget(store, config->scale(), *placements);
  if (!result)
    return result.takeError();
  std::vector<ArtifactRootReference> outputs;
  std::vector<CandidateGeneratorLineageEdge> lineage;
  outputs.reserve(result->roots().size());
  lineage.reserve(result->roots().size());
  for (const loom::fabric::FinalizedFabricRoot &root : result->roots()) {
    if (llvm::Error error = validateHardwareTopologyQuality(root.view()))
      return std::move(error);
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

llvm::Expected<ResolvedFabricTemplateConfigView> resolveFabricTemplateConfig(
    llvm::StringRef templateIdentity, std::uint32_t schemaMajor,
    std::uint32_t schemaMinor, const loom::adg::BuiltinTargetScale &scale,
    const std::optional<MinedCompositeFuSelection> &minedCompositeFus) {
  if (!loom::adg::isValidBuiltinTargetScale(scale))
    return invalid("template base scale is invalid or an FU occurrence count "
                   "exceeds its schedule-local PE count");
  const auto *descriptor = loom::adg::findBuiltinTargetDescriptor(
      templateIdentity, schemaMajor, schemaMinor);
  if (!descriptor)
    return invalid(
        "template descriptor is not a registered public Builder template");
  if (minedCompositeFus) {
    if (minedCompositeFus->templates.empty())
      return invalid("a mined composite FU selection names no template");
    for (const auto &indexed :
         llvm::enumerate(minedCompositeFus->templates)) {
      const MinedCompositeFuSelectionEntry &entry = indexed.value();
      if (entry.shapeKey.empty())
        return invalid("a mined composite FU selection has no shape key");
      if (entry.occurrences == 0 || entry.occurrences > scale.spatialPeCount)
        return invalid("a mined composite FU occurrence count is zero or "
                       "exceeds the Spatial PE count");
      if (indexed.index() != 0 &&
          !(minedCompositeFus->templates[indexed.index() - 1].shapeKey <
            entry.shapeKey))
        return invalid("mined composite FU shape keys are not canonical");
    }
  }
  std::vector<std::uint8_t> bytes = encodeConfig(scale, minedCompositeFus);
  auto digest = computeComponentViewDigest(descriptorBytes(), bytes);
  if (!digest)
    return digest.takeError();
  return ResolvedFabricTemplateConfigView(scale, minedCompositeFus,
                                          std::move(bytes), *digest);
}

llvm::Expected<ResolvedFabricTemplateConfigView>
projectResolvedFabricTemplateConfigView(const ResolvedConfig &config) {
  return resolveFabricTemplateConfig(config.hardwareTarget.templateIdentity,
                                     config.hardwareTarget.schemaVersion.major,
                                     config.hardwareTarget.schemaVersion.minor,
                                     config.hardwareTarget.parameters);
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
  auto decoded = decodeConfig(canonicalViewBytes);
  if (!decoded)
    return decoded.takeError();
  std::vector<std::uint8_t> reencoded =
      encodeConfig(decoded->scale, decoded->mined);
  if (llvm::ArrayRef<std::uint8_t>(reencoded) != canonicalViewBytes)
    return invalid("template config does not re-encode to the source bytes");
  return ResolvedFabricTemplateConfigView(decoded->scale,
                                          std::move(decoded->mined),
                                          std::move(reencoded), digest);
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
bindFabricTemplateCandidateGeneratorInputs(
    const std::optional<ArtifactRootReference> &dataflow) {
  if (llvm::Error error = registerFabricTemplateCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings;
  std::vector<ArtifactRootReference> artifacts;
  if (dataflow)
    artifacts.push_back(*dataflow);
  bindings.push_back(
      CandidateGeneratorInputBinding{CandidateGeneratorInputSlotRef(0),
                                     std::move(artifacts)});
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
