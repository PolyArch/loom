#include "EDA/Adapters/Cadence/Genus.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobDigest.h"
#include "Common/BlobStore.h"
#include "ExternalTool/ExternalFile.h"
#include "ExternalTool/Provider.h"
#include "ExternalTool/RuntimeBinding.h"
#include "ExternalTool/ShellProbe.h"
#include "Hardware/Implementation/RepresentationIndex.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace loom::eda::cadence {
namespace {

using namespace dse;
using namespace external_tool;
using namespace hardware;

constexpr char configSchema[] =
    "loom.eda.cadence.genus_gate_netlist_config.1.0";
constexpr llvm::StringLiteral providerIdentity = "cadence.genus.gate_netlist@1";
constexpr llvm::StringLiteral importerIdentity =
    "loom.eda.cadence.genus.gate_netlist.importer@1";
constexpr llvm::StringLiteral librarySlot = "standard_cell_liberty";
constexpr llvm::StringLiteral outputPath = "outputs/genus-gate-netlist.v";
constexpr llvm::StringLiteral externalContractRef =
    "cadence.genus.standard_cell_library";
constexpr llvm::StringLiteral externalContractPayload =
    "contracts/genus-standard-cells.txt";

enum InputSlot : std::uint32_t {
  RtlImplementationInput,
  AsicTargetInput,
  InputSlotCount,
};

constexpr std::array<CandidateGeneratorInputSlotDescriptor, InputSlotCount>
    inputSlots{{
        {CandidateGeneratorInputSlotRef(RtlImplementationInput),
         "finalized_rtl_with_generation_constraints",
         PlanValueRole::CandidateSet, &hardwareImplementationSchema,
         PlanValueCardinality::ExactlyOne},
        {CandidateGeneratorInputSlotRef(AsicTargetInput), "asic_target",
         PlanValueRole::CandidateSet, &platform::implementationPlatformSchema,
         PlanValueCardinality::ExactlyOne},
    }};

constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputSlots{{
    {CandidateGeneratorOutputSlotRef(0), "gate_netlist",
     PlanValueRole::CandidateSet, &hardwareImplementationSchema,
     PlanValueCardinality::ExactlyOne},
}};

constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 1> workUnits{{
    {CandidateGeneratorWorkUnitRef(0), "synthesis_attempt"},
}};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "cadence_genus_invalid: " + message);
}

llvm::ArrayRef<std::uint8_t> schemaBytes() {
  return llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(configSchema),
      sizeof(configSchema) - 1);
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

std::uint64_t readU64(llvm::ArrayRef<std::uint8_t> bytes) {
  std::uint64_t value = 0;
  for (std::uint8_t byte : bytes)
    value = (value << 8) | byte;
  return value;
}

std::uint32_t readU32(llvm::ArrayRef<std::uint8_t> bytes) {
  std::uint32_t value = 0;
  for (std::uint8_t byte : bytes)
    value = (value << 8) | byte;
  return value;
}

llvm::Error validateProviderBuildIdentity(llvm::StringRef identity) {
  if (identity.empty() ||
      identity.size() > std::numeric_limits<std::uint32_t>::max() ||
      identity.trim() != identity || identity.contains('\0') ||
      identity.contains('\n') || identity.contains('\r'))
    return invalid("provider build identity is not one normalized line");
  return llvm::Error::success();
}

std::vector<std::uint8_t>
encodeConfig(llvm::StringRef stableProviderBuildIdentity,
             const platform::TechnologyCornerRef &corner,
             const ExternalFileFingerprint &library) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(sizeof(std::uint32_t) + stableProviderBuildIdentity.size() +
                ArtifactIdentity::byteSize + sizeof(std::uint64_t) +
                ExternalFileFingerprint::byteSize);
  appendU32(bytes,
            static_cast<std::uint32_t>(stableProviderBuildIdentity.size()));
  bytes.insert(bytes.end(), stableProviderBuildIdentity.bytes_begin(),
               stableProviderBuildIdentity.bytes_end());
  bytes.insert(bytes.end(), corner.artifact.bytes().begin(),
               corner.artifact.bytes().end());
  appendU64(bytes, corner.entity.value());
  bytes.insert(bytes.end(), library.bytes().begin(), library.bytes().end());
  return bytes;
}

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted =
      adoptResolvedGenusGateNetlistConfigView(schemaBytes(), bytes, digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

llvm::Expected<PreparedExternalToolInvocation>
prepareProvider(llvm::ArrayRef<CandidateGeneratorInputBinding>,
                const ResolvedCandidateGeneratorBinding &,
                const ArtifactStore &, const BlobStore &,
                const ExternalToolPreparationContext &);

llvm::Expected<CandidateGeneratorProviderResult>
importProvider(llvm::ArrayRef<CandidateGeneratorInputBinding>,
               const ResolvedCandidateGeneratorBinding &,
               const PreparedExternalToolInvocation &, const ArtifactStore &,
               const BlobStore &);

const CandidateGeneratorDescriptor descriptor{
    genusGateNetlistCandidateGeneratorKind,
    "cadence.genus.gate_netlist",
    "loom.eda.cadence.genus.gate_netlist.generator.v1",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{schemaBytes(), validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
    nullptr,
    ProviderForm::ExternalPrepareImport,
};

const CandidateGeneratorProvider provider{
    descriptor.reference(), CandidateGeneratorExternalPrepareImportProvider{
                                prepareProvider, importProvider}};

bool isPortableIdentifier(llvm::StringRef value) {
  const auto first = [](char character) {
    return (character >= 'A' && character <= 'Z') ||
           (character >= 'a' && character <= 'z') || character == '_';
  };
  const auto rest = [&](char character) {
    return first(character) || (character >= '0' && character <= '9') ||
           character == '$';
  };
  return !value.empty() && first(value.front()) &&
         llvm::all_of(value.drop_front(), rest);
}

llvm::Error validateInputPath(llvm::StringRef path) {
  if (path.empty() || path.contains('\0'))
    return invalid("bundle input path is empty or contains NUL");
  const std::filesystem::path candidate(path.str());
  if (candidate.is_absolute() || candidate.lexically_normal() != candidate ||
      !llvm::StringRef(candidate.generic_string()).starts_with("inputs/"))
    return invalid("bundle input path must be normalized beneath inputs");
  return llvm::Error::success();
}

llvm::Expected<std::string> tclWord(llvm::StringRef value) {
  if (value.empty() || value.contains('\0') || value.contains('{') ||
      value.contains('}') || value.contains('\n') || value.contains('\r'))
    return invalid("value cannot be represented as one literal Tcl word");
  return "{" + value.str() + "}";
}

llvm::Error validateCanonicalPaths(llvm::ArrayRef<std::string> paths,
                                   llvm::StringRef inventory) {
  if (paths.empty())
    return invalid(inventory + " inventory is empty");
  if (!llvm::is_sorted(paths) ||
      std::adjacent_find(paths.begin(), paths.end()) != paths.end())
    return invalid(inventory + " inventory is not canonical");
  for (const std::string &path : paths)
    if (llvm::Error error = validateInputPath(path))
      return error;
  return llvm::Error::success();
}

bool containsTopModule(llvm::StringRef text, llvm::StringRef top) {
  std::size_t offset = 0;
  while (true) {
    const std::size_t found = text.find("module", offset);
    if (found == llvm::StringRef::npos)
      return false;
    const bool leftBoundary =
        found == 0 || !isPortableIdentifier(text.slice(found - 1, found + 1));
    llvm::StringRef rest = text.drop_front(found + 6).ltrim();
    if (leftBoundary && rest.starts_with(top)) {
      const llvm::StringRef suffix = rest.drop_front(top.size());
      if (suffix.empty() || suffix.front() == '(' || suffix.front() == '#' ||
          suffix.front() == ';' || suffix.front() == ' ' ||
          suffix.front() == '\t' || suffix.front() == '\n')
        return true;
    }
    offset = found + 6;
  }
}

std::vector<std::uint8_t> canonicalInputBindings(
    llvm::ArrayRef<CandidateGeneratorInputBinding> bindings) {
  std::vector<std::uint8_t> bytes;
  appendU64(bytes, bindings.size());
  for (const CandidateGeneratorInputBinding &binding : bindings) {
    appendU32(bytes, binding.slot.ordinal());
    appendU64(bytes, binding.artifacts.size());
    for (const ArtifactRootReference &artifact : binding.artifacts) {
      const std::vector<std::uint8_t> encoded =
          encodeArtifactRootReference(artifact);
      bytes.insert(bytes.end(), encoded.begin(), encoded.end());
    }
  }
  return bytes;
}

std::vector<std::uint8_t>
canonicalResolvedBinding(const ResolvedCandidateGeneratorBinding &binding) {
  const std::vector<std::uint8_t> reference =
      canonicalCandidateGeneratorDescriptorReferenceBytes(
          binding.descriptorRef());
  std::vector<std::uint8_t> bytes(reference.begin(), reference.end());
  appendU64(bytes, binding.canonicalConfigBytes().size());
  bytes.insert(bytes.end(), binding.canonicalConfigBytes().begin(),
               binding.canonicalConfigBytes().end());
  bytes.insert(bytes.end(), binding.configDigest().bytes().begin(),
               binding.configDigest().bytes().end());
  return bytes;
}

SemanticInvocationClosure
makeClosure(llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
            const ResolvedCandidateGeneratorBinding &binding) {
  const BlobDigest identity = deriveCandidateGeneratorBindingIdentity(
      binding.descriptorRef(), binding.canonicalConfigBytes());
  return CandidateGeneratorInvocationClosure{canonicalInputBindings(inputs),
                                             canonicalResolvedBinding(binding),
                                             identity.bytes()};
}

struct InvocationFacts final {
  FinalizedHardwareImplementation rtl;
  platform::FinalizedImplementationPlatform platform;
  ResolvedGenusGateNetlistConfigView config;
  SemanticInvocationClosure closure;
  std::vector<MaterializedBundleFile> semanticInputs;
  std::vector<std::string> rtlPaths;
  std::vector<std::string> constraintPaths;
  std::string top;
};

llvm::Expected<std::string> blobText(const BlobStore &blobs,
                                     const BlobDigest &digest) {
  auto bytes = blobs.get(digest);
  if (!bytes)
    return bytes.takeError();
  return std::string(reinterpret_cast<const char *>(bytes->data()),
                     bytes->size());
}

llvm::Expected<InvocationFacts>
invocationFacts(llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
                const ResolvedCandidateGeneratorBinding &binding,
                const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (inputs.size() != InputSlotCount ||
      inputs[RtlImplementationInput].artifacts.size() != 1 ||
      inputs[AsicTargetInput].artifacts.size() != 1)
    return invalid("typed input closure is incomplete");
  auto config = adoptResolvedGenusGateNetlistConfigView(
      schemaBytes(), binding.canonicalConfigBytes(), binding.configDigest());
  if (!config)
    return config.takeError();
  auto rtl = importHardwareImplementation(
      inputs[RtlImplementationInput].artifacts.front(), artifacts, blobs);
  if (!rtl)
    return rtl.takeError();
  auto target = platform::importImplementationPlatform(
      inputs[AsicTargetInput].artifacts.front(), artifacts);
  if (!target)
    return target.takeError();

  const HardwareImplementation &implementation = rtl->implementation();
  const ImplementationRepresentationRoot &representation =
      implementation.representationRoot();
  if (representation.variant != RepresentationRootVariant::Rtl ||
      representation.stage ||
      representation.formatRef.kind() !=
          RepresentationFormatKind::SystemVerilogRtl)
    return invalid("input HardwareImplementation is not finalized RTL");
  if (!implementation.implementationPlatform() ||
      *implementation.implementationPlatform() != target->reference())
    return invalid("RTL and target input slots do not name one exact platform");
  if (!std::holds_alternative<platform::AsicTarget>(
          target->platform().target()))
    return invalid("bound implementation platform is not an ASIC target");
  if (config->technologyCorner().artifact != target->reference().artifact ||
      !target->platform().findTechnologyCorner(
          config->technologyCorner().entity))
    return invalid("resolved technology corner is outside the exact target");
  if (!implementation.externalImplementationBindings().empty() ||
      !implementation.memoryMacroBindings().empty())
    return invalid("RTL with external implementation bindings is unsupported");
  if (representation.top.kind != RepresentationObjectKind::Module ||
      !isPortableIdentifier(representation.top.canonicalName))
    return invalid("RTL top is not a portable module identifier");

  std::vector<MaterializedBundleFile> materialized;
  std::vector<std::string> rtlPaths;
  std::vector<std::string> constraintPaths;
  for (const ImplementationPayload &payload : representation.payloads) {
    if (payload.role != PayloadRole::RtlSource &&
        payload.role != PayloadRole::GenerationConstraint)
      continue;
    auto contents = blobText(blobs, payload.blobDigest);
    if (!contents)
      return contents.takeError();
    const std::string path =
        "inputs/implementation/" + payload.canonicalLogicalName;
    if (llvm::Error error = validateInputPath(path))
      return std::move(error);
    materialized.push_back(MaterializedBundleFile{path, std::move(*contents),
                                                  rtl->reference(), false});
    if (payload.role == PayloadRole::RtlSource)
      rtlPaths.push_back(path);
    else
      constraintPaths.push_back(path);
  }
  if (llvm::Error error = validateCanonicalPaths(rtlPaths, "RTL source"))
    return std::move(error);
  if (llvm::Error error = validateCanonicalPaths(constraintPaths, "constraint"))
    return std::move(error);

  const llvm::ArrayRef<std::uint8_t> platformBytes =
      target->canonicalBytes().bytes();
  materialized.push_back(MaterializedBundleFile{
      "inputs/target/implementation-platform.json",
      std::string(reinterpret_cast<const char *>(platformBytes.data()),
                  platformBytes.size()),
      target->reference(), false});
  const std::string top = representation.top.canonicalName;
  return InvocationFacts{std::move(*rtl),
                         std::move(*target),
                         std::move(*config),
                         makeClosure(inputs, binding),
                         std::move(materialized),
                         std::move(rtlPaths),
                         std::move(constraintPaths),
                         top};
}

ExternalToolInvocationImportExpectation
expectation(const InvocationFacts &facts) {
  ExternalToolInvocationImportExpectation result;
  result.providerIdentity = providerIdentity.str();
  result.semanticClosure = facts.closure;
  result.resultImporterIdentity = importerIdentity.str();
  for (const MaterializedBundleFile &file : facts.semanticInputs) {
    result.semanticInputs.push_back(ExternalToolInvocationSemanticInput{
        file.relativePath, *file.sourceArtifact,
        computeBlobDigest(llvm::ArrayRef<std::uint8_t>(
            reinterpret_cast<const std::uint8_t *>(file.contents.data()),
            file.contents.size()))});
  }
  result.externalInputs.push_back(ExternalToolInvocationExternalInput{
      librarySlot.str(), facts.config.standardCellLiberty()});
  result.declaredOutputs.push_back(outputPath.str());
  return result;
}

llvm::Expected<ExternalImplementationContractCatalog> makeExternalCatalog() {
  ExternalImplementationContractCatalog catalog;
  if (llvm::Error error = catalog.add(ExternalImplementationContract{
          externalContractRef.str(),
          {{librarySlot.str(), {ExternalDependencyKind::ExplicitFile}}},
          {RepresentationRootVariant::GateNetlist},
          true,
          false,
          nullptr}))
    return std::move(error);
  return catalog;
}

} // namespace

llvm::ArrayRef<std::uint8_t>
resolvedGenusGateNetlistConfigSchemaDescriptorBytes() {
  return schemaBytes();
}

llvm::Expected<ResolvedGenusGateNetlistConfigView>
createResolvedGenusGateNetlistConfigView(
    llvm::StringRef stableProviderBuildIdentity,
    platform::TechnologyCornerRef technologyCorner,
    ExternalFileFingerprint standardCellLiberty) {
  if (llvm::Error error =
          validateProviderBuildIdentity(stableProviderBuildIdentity))
    return std::move(error);
  std::vector<std::uint8_t> bytes = encodeConfig(
      stableProviderBuildIdentity, technologyCorner, standardCellLiberty);
  auto digest = computeComponentViewDigest(schemaBytes(), bytes);
  if (!digest)
    return digest.takeError();
  return ResolvedGenusGateNetlistConfigView(
      stableProviderBuildIdentity.str(), std::move(technologyCorner),
      std::move(standardCellLiberty), std::move(bytes), std::move(*digest));
}

llvm::Expected<ResolvedGenusGateNetlistConfigView>
adoptResolvedGenusGateNetlistConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest) {
  if (schemaDescriptorBytes != schemaBytes())
    return invalid("config descriptor does not match the exact owner");
  if (llvm::Error error = validateComponentViewDigest(
          schemaDescriptorBytes, canonicalViewBytes, digest))
    return std::move(error);
  constexpr std::size_t fixedConfigSize = ArtifactIdentity::byteSize +
                                          sizeof(std::uint64_t) +
                                          ExternalFileFingerprint::byteSize;
  if (canonicalViewBytes.size() < sizeof(std::uint32_t) + fixedConfigSize)
    return invalid("config bytes have the wrong size");
  const std::uint32_t buildIdentitySize =
      readU32(canonicalViewBytes.take_front(sizeof(std::uint32_t)));
  const std::size_t buildIdentityOffset = sizeof(std::uint32_t);
  if (canonicalViewBytes.size() !=
      buildIdentityOffset + buildIdentitySize + fixedConfigSize)
    return invalid("config bytes have the wrong size");
  const llvm::ArrayRef<std::uint8_t> buildIdentityBytes =
      canonicalViewBytes.slice(buildIdentityOffset, buildIdentitySize);
  const llvm::StringRef buildIdentity(
      reinterpret_cast<const char *>(buildIdentityBytes.data()),
      buildIdentityBytes.size());
  if (llvm::Error error = validateProviderBuildIdentity(buildIdentity))
    return std::move(error);
  const std::size_t platformOffset = buildIdentityOffset + buildIdentitySize;
  auto platformIdentity = ArtifactIdentity::fromBytes(
      canonicalViewBytes.slice(platformOffset, ArtifactIdentity::byteSize));
  if (!platformIdentity)
    return platformIdentity.takeError();
  const std::size_t cornerOffset = platformOffset + ArtifactIdentity::byteSize;
  const std::uint64_t corner =
      readU64(canonicalViewBytes.slice(cornerOffset, sizeof(std::uint64_t)));
  auto liberty = ExternalFileFingerprint::fromBytes(
      canonicalViewBytes.take_back(ExternalFileFingerprint::byteSize));
  if (!liberty)
    return liberty.takeError();
  platform::TechnologyCornerRef cornerRef{std::move(*platformIdentity),
                                          platform::TechnologyCornerId(corner)};
  std::vector<std::uint8_t> canonical =
      encodeConfig(buildIdentity, cornerRef, *liberty);
  if (llvm::ArrayRef<std::uint8_t>(canonical) != canonicalViewBytes)
    return invalid("decoded config does not re-encode canonically");
  return ResolvedGenusGateNetlistConfigView(
      buildIdentity.str(), std::move(cornerRef), std::move(*liberty),
      std::move(canonical), digest);
}

const dse::CandidateGeneratorDescriptor &
genusGateNetlistCandidateGeneratorDescriptor() {
  return descriptor;
}

llvm::Error registerGenusGateNetlistCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<dse::CandidateGeneratorInputBinding>>
bindGenusGateNetlistInputs(
    const ArtifactRootReference &rtlImplementation,
    const ArtifactRootReference &implementationPlatform) {
  if (llvm::Error error = registerGenusGateNetlistCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> inputs{
      {CandidateGeneratorInputSlotRef(RtlImplementationInput),
       {rtlImplementation}},
      {CandidateGeneratorInputSlotRef(AsicTargetInput),
       {implementationPlatform}},
  };
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), inputs))
    return std::move(error);
  return inputs;
}

llvm::Expected<dse::ResolvedCandidateGeneratorBinding>
resolveGenusGateNetlistBinding(
    const ResolvedGenusGateNetlistConfigView &config) {
  if (llvm::Error error = registerGenusGateNetlistCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

llvm::Expected<std::string>
renderGenusGateNetlistDriver(llvm::StringRef top,
                             llvm::ArrayRef<std::string> rtlSources,
                             llvm::ArrayRef<std::string> generationConstraints,
                             llvm::StringRef standardCellLiberty) {
  if (!isPortableIdentifier(top))
    return invalid("top is not a portable HDL identifier");
  if (llvm::Error error = validateCanonicalPaths(rtlSources, "RTL source"))
    return std::move(error);
  if (llvm::Error error =
          validateCanonicalPaths(generationConstraints, "constraint"))
    return std::move(error);
  auto topWord = tclWord(top);
  auto libraryWord = tclWord(standardCellLiberty);
  if (!topWord)
    return topWord.takeError();
  if (!libraryWord)
    return libraryWord.takeError();

  std::string sources;
  for (const std::string &path : rtlSources) {
    auto word = tclWord(path);
    if (!word)
      return word.takeError();
    sources += (sources.empty() ? "" : " ") + *word;
  }
  std::string script = "read_libs " + *libraryWord + "\n";
  script += "read_hdl -sv [list " + sources + "]\n";
  script += "elaborate " + *topWord + "\n";
  for (const std::string &path : generationConstraints) {
    auto word = tclWord(path);
    if (!word)
      return word.takeError();
    script += "read_sdc " + *word + "\n";
  }
  script += "syn_generic\n"
            "syn_map\n"
            "syn_opt\n"
            "write_hdl > {outputs/genus-gate-netlist.v}\n"
            "exit\n";
  return script;
}

llvm::Expected<GenusGateNetlist> parseGenusGateNetlist(llvm::StringRef contents,
                                                       llvm::StringRef top) {
  if (!isPortableIdentifier(top))
    return invalid("expected top is not a portable HDL identifier");
  if (contents.empty() || contents.contains('\0') || contents.contains('\r') ||
      !contents.ends_with("\n"))
    return invalid("gate netlist is empty or violates the LF text contract");
  if (!containsTopModule(contents, top))
    return invalid("gate netlist does not define exact top '" + top + "'");
  return GenusGateNetlist{contents.str()};
}

llvm::Expected<hardware::FinalizedHardwareImplementation>
importGenusGateNetlistImplementation(const ArtifactRootReference &reference,
                                     const ArtifactStore &artifacts,
                                     const BlobStore &blobs) {
  auto catalog = makeExternalCatalog();
  if (!catalog)
    return catalog.takeError();
  return importHardwareImplementation(reference, *catalog, artifacts, blobs);
}

namespace {

llvm::Expected<PreparedExternalToolInvocation>
prepareProvider(llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
                const ResolvedCandidateGeneratorBinding &binding,
                const ArtifactStore &artifacts, const BlobStore &blobs,
                const ExternalToolPreparationContext &context) {
  auto facts = invocationFacts(inputs, binding, artifacts, blobs);
  if (!facts)
    return facts.takeError();
  auto externalFiles = resolveExternalFiles(
      {{librarySlot.str(), facts->config.standardCellLiberty()}},
      context.localConfig);
  if (!externalFiles)
    return externalFiles.takeError();

  const ExternalToolProviderDescriptor &toolProvider = genusProvider();
  const std::filesystem::path destination(context.bundleDestination);
  const std::filesystem::path probeRoot = destination.parent_path();
  ShellToolBindingProbe toolProbe(probeRoot.string(),
                                  toolProvider.versionProbe);
  const ToolEnvironment toolEnvironment =
      captureToolEnvironment(toolProvider.binding);
  auto tool = resolveToolBinding(toolProvider.binding, context.localConfig,
                                 toolEnvironment, toolProbe);
  if (!tool)
    return tool.takeError();
  if (tool->version != facts->config.stableProviderBuildIdentity())
    return invalid(llvm::Twine("resolved Genus build '") + tool->version +
                   "' does not match semantic build '" +
                   facts->config.stableProviderBuildIdentity() + "'");

  std::vector<std::string> inheritEnvironment;
  const auto configured =
      context.localConfig.tools.find(toolProvider.binding.key);
  if (configured != context.localConfig.tools.end())
    inheritEnvironment = configured->second.inheritEnvironment;

  const ExternalToolProviderDescriptor &containerProvider =
      polyArchContainerProvider();
  ShellToolBindingProbe containerProbe(probeRoot.string(),
                                       containerProvider.versionProbe);
  const ToolEnvironment containerEnvironment =
      captureToolEnvironment(containerProvider.binding);
  auto runtime = resolveInvocationRuntime(
      *tool, context.localConfig, containerProvider.binding,
      containerEnvironment, containerProbe, toolProvider.runtimeCompatibility,
      [&](const ResolvedToolBinding &resolvedTool,
          const ResolvedToolBinding &container,
          llvm::StringRef os) -> llvm::Expected<std::optional<std::string>> {
        return probeContainerToolComposition(probeRoot.string(), resolvedTool,
                                             toolProvider.versionProbe,
                                             container, os, inheritEnvironment);
      });
  if (!runtime)
    return runtime.takeError();

  auto driver = renderGenusGateNetlistDriver(
      facts->top, facts->rtlPaths, facts->constraintPaths,
      externalFiles->front().absolutePath);
  if (!driver)
    return driver.takeError();
  std::vector<MaterializedBundleFile> files{
      {"drivers/genus.tcl", std::move(*driver), std::nullopt, false}};
  files.insert(files.end(), facts->semanticInputs.begin(),
               facts->semanticInputs.end());
  const std::string executable = tool->executable;
  ExternalToolInvocationBundleSpec specification{
      providerIdentity.str(),
      facts->closure,
      importerIdentity.str(),
      std::move(*tool),
      toolProvider.versionProbe,
      std::move(*runtime),
      containerProvider.versionProbe,
      {{executable, "-batch", "-files", "drivers/genus.tcl"}},
      std::move(inheritEnvironment),
      {outputPath.str()},
      std::move(files),
      std::move(*externalFiles)};
  return finalizeExternalToolInvocationBundle(context.bundleDestination,
                                              specification);
}

llvm::Error rejectUndeclaredOutputs(llvm::StringRef bundleRoot) {
  const std::filesystem::path outputs =
      std::filesystem::path(bundleRoot.str()) / "outputs";
  const std::set<std::string> allowed{"completion.json", "genus-gate-netlist.v",
                                      "stderr.log", "stdout.log"};
  std::set<std::string> found;
  std::error_code error;
  const std::filesystem::file_status rootStatus =
      std::filesystem::symlink_status(outputs, error);
  if (error || !std::filesystem::is_directory(rootStatus) ||
      std::filesystem::is_symlink(rootStatus))
    return invalid("outputs directory is missing or not an ordinary directory");
  for (std::filesystem::directory_iterator iterator(outputs, error), end;
       !error && iterator != end; iterator.increment(error)) {
    const std::filesystem::path path = iterator->path();
    const std::filesystem::file_status status =
        std::filesystem::symlink_status(path, error);
    if (error)
      break;
    const std::string name = path.filename().string();
    if (!std::filesystem::is_regular_file(status) ||
        std::filesystem::is_symlink(status) || !allowed.count(name))
      return invalid("outputs directory contains undeclared output '" + name +
                     "'");
    found.insert(name);
  }
  if (error)
    return invalid("could not enumerate outputs directory: " + error.message());
  if (found != allowed)
    return invalid("outputs directory omits a lifecycle or declared output");
  return llvm::Error::success();
}

llvm::Error validatePreservedInterfaces(const HardwareImplementation &source,
                                        const RepresentationIndex &output,
                                        const BlobStore &blobs) {
  auto input = indexRepresentation(source.representationRoot().formatRef,
                                   source.representationRoot().top,
                                   source.representationRoot().payloads, blobs);
  if (!input)
    return input.takeError();
  for (const ImplementationInterface &interface : source.interfaces()) {
    auto before = input->lookup(interface.representationLocator);
    if (!before)
      return before.takeError();
    auto after = output.lookup(interface.representationLocator);
    if (!after)
      return after.takeError();
    if (!*before || !*after || !(**before == **after))
      return invalid(
          "synthesis changed an exact interface locator or geometry");
  }
  return llvm::Error::success();
}

llvm::Expected<ArtifactRootReference>
publishGateNetlist(const InvocationFacts &facts,
                   const GenusGateNetlist &netlist,
                   const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto netlistDigest = blobs.put(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(netlist.verilog.data()),
      netlist.verilog.size()));
  if (!netlistDigest)
    return netlistDigest.takeError();
  auto gateFormat = RepresentationFormatDescriptorRef::get(
      RepresentationFormatKind::StructuralVerilogGateNetlist);
  if (!gateFormat)
    return gateFormat.takeError();

  std::vector<ImplementationPayload> payloads{
      {PayloadRole::Netlist, "netlist/" + facts.top + ".v", *netlistDigest}};
  for (const ImplementationPayload &payload :
       facts.rtl.implementation().representationRoot().payloads)
    if (payload.role == PayloadRole::GenerationConstraint)
      payloads.push_back(payload);
  auto canonicalPayloads = canonicalizeImplementationPayloadCatalog(payloads);
  if (!canonicalPayloads)
    return canonicalPayloads.takeError();
  auto index = indexRepresentation(
      *gateFormat, {RepresentationObjectKind::Module, facts.top},
      *canonicalPayloads, blobs);
  if (!index)
    return index.takeError();
  const HardwareImplementation &source = facts.rtl.implementation();
  if (llvm::Error error = validatePreservedInterfaces(source, *index, blobs))
    return std::move(error);

  std::string contract =
      "loom.cadence.genus.standard_cell_contract.1.0\nliberty_sha256=" +
      formatExternalFileFingerprint(facts.config.standardCellLiberty()) + "\n";
  for (const RepresentationLocator &locator :
       index->unresolvedExternalDefinitions())
    contract += "module=" + locator.canonicalName + "\n";
  auto contractDigest = blobs.put(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(contract.data()),
      contract.size()));
  if (!contractDigest)
    return contractDigest.takeError();
  payloads.push_back({PayloadRole::BlackBoxContract,
                      externalContractPayload.str(), *contractDigest});
  auto representation = createImplementationRepresentationRoot(
      RepresentationRootVariant::GateNetlist, std::nullopt, *gateFormat,
      {RepresentationObjectKind::Module, facts.top}, std::move(payloads));
  if (!representation)
    return representation.takeError();

  auto catalog = makeExternalCatalog();
  if (!catalog)
    return catalog.takeError();
  HardwareImplementationDraft draft{
      source.fabric(),
      source.configurationAbi(),
      source.interconnectImplementations().vec(),
      std::move(*representation),
      source.implementationPlatform(),
      source.interfaces().vec(),
      {},
      {},
      {ExternalImplementationBindingDraft{
          externalContractRef.str(),
          {{librarySlot.str(),
            ExplicitFileDependency{facts.config.standardCellLiberty()}}},
          {},
          index->unresolvedExternalDefinitions().vec(),
          ImplementationPayloadKey{PayloadRole::BlackBoxContract,
                                   externalContractPayload.str()}}}};
  auto finalized = finalizeHardwareImplementation(std::move(draft), *catalog,
                                                  artifacts, blobs);
  if (!finalized)
    return finalized.takeError();
  auto strict = importHardwareImplementation(finalized->reference(), *catalog,
                                             artifacts, blobs);
  if (!strict)
    return strict.takeError();
  return strict->reference();
}

llvm::Expected<CandidateGeneratorProviderResult>
importProvider(llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
               const ResolvedCandidateGeneratorBinding &binding,
               const PreparedExternalToolInvocation &prepared,
               const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto facts = invocationFacts(inputs, binding, artifacts, blobs);
  if (!facts)
    return facts.takeError();
  auto imported =
      importExternalToolInvocationBundle(prepared, expectation(*facts));
  if (!imported)
    return imported.takeError();
  if (llvm::Error error = rejectUndeclaredOutputs(prepared.bundleRoot))
    return std::move(error);
  auto output = readExternalToolInvocationDeclaredOutput(*imported, outputPath);
  if (!output)
    return output.takeError();
  auto netlist = parseGenusGateNetlist(*output, facts->top);
  if (!netlist)
    return netlist.takeError();
  auto published = publishGateNetlist(*facts, *netlist, artifacts, blobs);
  if (!published)
    return published.takeError();
  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{
          {{CandidateGeneratorOutputSlotRef(0), {*published}}},
          {{CandidateGeneratorLineageEdgeKind::MechanicalDerivation,
            CandidateGeneratorOutputSlotRef(0),
            *published,
            {},
            {}}}},
      {{CandidateGeneratorWorkUnitRef(0), 1, 1}}};
}

} // namespace
} // namespace loom::eda::cadence
