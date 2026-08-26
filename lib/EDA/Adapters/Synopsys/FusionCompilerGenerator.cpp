#include "EDA/Adapters/Synopsys/FusionCompiler.h"

#include "EDA/Adapters/Synopsys/DesignCompiler.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobDigest.h"
#include "Common/BlobStore.h"
#include "ExternalTool/Provider.h"
#include "ExternalTool/RuntimeBinding.h"
#include "ExternalTool/ShellProbe.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::eda::synopsys {
namespace {

using namespace dse;
using namespace external_tool;
using namespace hardware;

constexpr char configSchema[] =
    "loom.eda.synopsys.fusion_compiler_routed_config.1.0";
constexpr llvm::StringLiteral referenceLibrarySlot = "reference_library";
constexpr llvm::StringLiteral earlyParasiticSlot = "early_parasitic_tech";
constexpr llvm::StringLiteral lateParasiticSlot = "late_parasitic_tech";
constexpr llvm::StringLiteral layerMapSlot = "parasitic_layer_map";
constexpr llvm::StringLiteral floorplanPath = "drivers/fusion-floorplan.def";

enum InputSlot : std::uint32_t {
  GateNetlistInput,
  AsicTargetInput,
  InputSlotCount,
};

constexpr std::array<CandidateGeneratorInputSlotDescriptor, InputSlotCount>
    inputSlots{{
        {CandidateGeneratorInputSlotRef(GateNetlistInput),
         "finalized_gate_netlist_with_generation_constraints",
         PlanValueRole::CandidateSet, &hardwareImplementationSchema,
         PlanValueCardinality::ExactlyOne},
        {CandidateGeneratorInputSlotRef(AsicTargetInput), "asic_target",
         PlanValueRole::CandidateSet, &platform::implementationPlatformSchema,
         PlanValueCardinality::ExactlyOne},
    }};

constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputSlots{{
    {CandidateGeneratorOutputSlotRef(0), "routed_asic_physical",
     PlanValueRole::CandidateSet, &hardwareImplementationSchema,
     PlanValueCardinality::ExactlyOne},
}};

constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 1> workUnits{{
    {CandidateGeneratorWorkUnitRef(0), "physical_implementation_attempt"},
}};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "synopsys_fusion_compiler_invalid: " +
                                     message);
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

void appendString32(std::vector<std::uint8_t> &bytes, llvm::StringRef value) {
  appendU32(bytes, static_cast<std::uint32_t>(value.size()));
  bytes.insert(bytes.end(), value.bytes_begin(), value.bytes_end());
}

class Reader final {
public:
  explicit Reader(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<llvm::ArrayRef<std::uint8_t>> take(std::size_t size) {
    if (size > bytes_.size() - offset_)
      return invalid("config bytes are truncated");
    llvm::ArrayRef<std::uint8_t> result = bytes_.slice(offset_, size);
    offset_ += size;
    return result;
  }

  llvm::Expected<std::uint32_t> u32() {
    auto bytes = take(sizeof(std::uint32_t));
    if (!bytes)
      return bytes.takeError();
    std::uint32_t value = 0;
    for (std::uint8_t byte : *bytes)
      value = (value << 8) | byte;
    return value;
  }

  llvm::Expected<std::uint64_t> u64() {
    auto bytes = take(sizeof(std::uint64_t));
    if (!bytes)
      return bytes.takeError();
    std::uint64_t value = 0;
    for (std::uint8_t byte : *bytes)
      value = (value << 8) | byte;
    return value;
  }

  llvm::Expected<std::string> string32() {
    auto size = u32();
    if (!size)
      return size.takeError();
    auto bytes = take(*size);
    if (!bytes)
      return bytes.takeError();
    return std::string(reinterpret_cast<const char *>(bytes->data()),
                       bytes->size());
  }

  std::size_t remaining() const { return bytes_.size() - offset_; }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

llvm::Error validateProviderBuildIdentity(llvm::StringRef identity) {
  if (identity.empty() ||
      identity.size() > std::numeric_limits<std::uint32_t>::max() ||
      identity.trim() != identity || identity.contains('\0') ||
      identity.contains('\n') || identity.contains('\r'))
    return invalid("provider build identity is not one normalized line");
  return llvm::Error::success();
}

llvm::Error validateFloorplan(llvm::StringRef floorplan) {
  if (floorplan.empty() || floorplan.contains('\0') || floorplan.contains('\r'))
    return invalid("floorplan DEF violates the nonempty LF text contract");
  return llvm::Error::success();
}

llvm::Error
validateConfigValues(llvm::StringRef providerBuild,
                     llvm::ArrayRef<ExternalFileTreeMember> referenceLibrary,
                     llvm::StringRef floorplan) {
  if (llvm::Error error = validateProviderBuildIdentity(providerBuild))
    return error;
  if (referenceLibrary.size() > std::numeric_limits<std::uint32_t>::max())
    return invalid("reference library member count is not encodable");
  if (llvm::Error error = validateExternalFileTreeRequirement(
          {referenceLibrarySlot.str(), referenceLibrary.vec()}))
    return error;
  return validateFloorplan(floorplan);
}

std::vector<std::uint8_t> encodeConfig(
    llvm::StringRef providerBuild, const platform::TechnologyCornerRef &corner,
    llvm::ArrayRef<ExternalFileTreeMember> referenceLibrary,
    const ExternalFileFingerprint &earlyParasitic,
    const ExternalFileFingerprint &lateParasitic,
    const ExternalFileFingerprint &layerMap, llvm::StringRef floorplan) {
  std::vector<std::uint8_t> bytes;
  appendString32(bytes, providerBuild);
  bytes.insert(bytes.end(), corner.artifact.bytes().begin(),
               corner.artifact.bytes().end());
  appendU64(bytes, corner.entity.value());
  appendU32(bytes, static_cast<std::uint32_t>(referenceLibrary.size()));
  for (const ExternalFileTreeMember &member : referenceLibrary) {
    appendString32(bytes, member.relativePath);
    bytes.insert(bytes.end(), member.fingerprint.bytes().begin(),
                 member.fingerprint.bytes().end());
  }
  for (const ExternalFileFingerprint *fingerprint :
       {&earlyParasitic, &lateParasitic, &layerMap})
    bytes.insert(bytes.end(), fingerprint->bytes().begin(),
                 fingerprint->bytes().end());
  appendU64(bytes, floorplan.size());
  bytes.insert(bytes.end(), floorplan.bytes_begin(), floorplan.bytes_end());
  return bytes;
}

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted =
      adoptResolvedFusionCompilerRoutedConfigView(schemaBytes(), bytes, digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

llvm::Expected<PreparedExternalToolInvocation>
prepareProviderWithContracts(llvm::ArrayRef<CandidateGeneratorInputBinding>,
                             const ResolvedCandidateGeneratorBinding &,
                             const ExternalImplementationContractCatalog &,
                             const ArtifactStore &, const BlobStore &,
                             const ExternalToolPreparationContext &);

llvm::Expected<CandidateGeneratorProviderResult> importProviderWithContracts(
    llvm::ArrayRef<CandidateGeneratorInputBinding>,
    const ResolvedCandidateGeneratorBinding &,
    const PreparedExternalToolInvocation &,
    const ExternalImplementationContractCatalog &, const ArtifactStore &,
    const BlobStore &,
    const ExternalToolInvocationExecutionObservation * = nullptr);

llvm::Expected<PreparedExternalToolInvocation>
prepareProvider(llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
                const ResolvedCandidateGeneratorBinding &binding,
                const ArtifactStore &artifacts, const BlobStore &blobs,
                const ExternalToolPreparationContext &context) {
  auto contracts = makeSynopsysStandardCellContractCatalog();
  if (!contracts)
    return contracts.takeError();
  return prepareProviderWithContracts(inputs, binding, *contracts, artifacts,
                                      blobs, context);
}

llvm::Expected<CandidateGeneratorProviderResult>
importProvider(llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
               const ResolvedCandidateGeneratorBinding &binding,
               const PreparedExternalToolInvocation &prepared,
               const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto contracts = makeSynopsysStandardCellContractCatalog();
  if (!contracts)
    return contracts.takeError();
  return importProviderWithContracts(inputs, binding, prepared, *contracts,
                                     artifacts, blobs);
}

llvm::Expected<CandidateGeneratorProviderResult> importProviderWithExecution(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    const ResolvedCandidateGeneratorBinding &binding,
    const PreparedExternalToolInvocation &prepared,
    const ExternalToolInvocationExecutionObservation &execution,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto contracts = makeSynopsysStandardCellContractCatalog();
  if (!contracts)
    return contracts.takeError();
  return importProviderWithContracts(inputs, binding, prepared, *contracts,
                                     artifacts, blobs, &execution);
}

const CandidateGeneratorDescriptor &generatorDescriptor() {
  static const CandidateGeneratorDescriptor descriptor{
      fusionCompilerRoutedCandidateGeneratorKind,
      "synopsys.fusion_compiler.routed_asic_physical",
      fusionCompilerDescriptor().implementationSemanticIdentity,
      inputSlots,
      outputSlots,
      ResolvedDseConfigViewContract{schemaBytes(), validateConfig},
      CandidateGeneratorDeterminism::IndependentReplicates,
      workUnits,
      nullptr,
      ProviderForm::ExternalPrepareImport,
  };
  return descriptor;
}

const CandidateGeneratorProvider &generatorProvider() {
  static const CandidateGeneratorProvider provider{
      generatorDescriptor().reference(),
      CandidateGeneratorExternalPrepareImportProvider{
          prepareProvider, importProvider, importProviderWithExecution}};
  return provider;
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

struct InvocationFacts final {
  FinalizedHardwareImplementation gate;
  platform::FinalizedImplementationPlatform platform;
  ResolvedFusionCompilerRoutedConfigView config;
  ExternalToolSemanticContract semanticContract;
  std::vector<MaterializedBundleFile> semanticInputs;
  std::string netlistPath;
  std::string constraintPath;
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
                const ExternalImplementationContractCatalog &contracts,
                const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (inputs.size() != InputSlotCount ||
      inputs[GateNetlistInput].artifacts.size() != 1 ||
      inputs[AsicTargetInput].artifacts.size() != 1)
    return invalid("typed input closure is incomplete");
  auto config = adoptResolvedFusionCompilerRoutedConfigView(
      schemaBytes(), binding.canonicalConfigBytes(), binding.configDigest());
  if (!config)
    return config.takeError();
  auto gate = importHardwareImplementation(
      inputs[GateNetlistInput].artifacts.front(), contracts, artifacts, blobs);
  if (!gate)
    return gate.takeError();
  auto target = platform::importImplementationPlatform(
      inputs[AsicTargetInput].artifacts.front(), artifacts);
  if (!target)
    return target.takeError();

  const HardwareImplementation &implementation = gate->implementation();
  const ImplementationRepresentationRoot &representation =
      implementation.representationRoot();
  if (representation.variant != RepresentationRootVariant::GateNetlist ||
      representation.stage ||
      representation.formatRef.kind() !=
          RepresentationFormatKind::StructuralVerilogGateNetlist)
    return invalid("input HardwareImplementation is not a GateNetlist");
  if (!implementation.implementationPlatform() ||
      *implementation.implementationPlatform() != target->reference())
    return invalid(
        "GateNetlist and target inputs do not name one exact platform");
  if (!std::holds_alternative<platform::AsicTarget>(
          target->platform().target()))
    return invalid("bound implementation platform is not an ASIC target");
  if (config->technologyCorner().artifact != target->reference().artifact ||
      !target->platform().findTechnologyCorner(
          config->technologyCorner().entity))
    return invalid("resolved technology corner is outside the exact target");
  if (representation.top.kind != RepresentationObjectKind::Module ||
      !isPortableHdlIdentifier(representation.top.canonicalName))
    return invalid("GateNetlist top is not a portable module identifier");

  std::vector<MaterializedBundleFile> materialized;
  std::vector<std::string> netlists;
  std::vector<std::string> constraints;
  for (const ImplementationPayload &payload : representation.payloads) {
    if (payload.role != PayloadRole::Netlist &&
        payload.role != PayloadRole::GenerationConstraint)
      continue;
    auto contents = blobText(blobs, payload.blobDigest);
    if (!contents)
      return contents.takeError();
    const std::string path =
        "inputs/implementation/" + payload.canonicalLogicalName;
    if (llvm::Error error = validateInputPath(path))
      return error;
    materialized.push_back(MaterializedBundleFile{path, std::move(*contents),
                                                  gate->reference(), false});
    (payload.role == PayloadRole::Netlist ? netlists : constraints)
        .push_back(path);
  }
  if (netlists.size() != 1 || constraints.size() != 1)
    return invalid(
        "Fusion Compiler requires one exact netlist and one constraint");

  const llvm::ArrayRef<std::uint8_t> platformBytes =
      target->canonicalBytes().bytes();
  materialized.push_back(MaterializedBundleFile{
      "inputs/target/implementation-platform.json",
      std::string(reinterpret_cast<const char *>(platformBytes.data()),
                  platformBytes.size()),
      target->reference(), false});
  auto semanticContract = deriveExternalToolSemanticContract(inputs, binding);
  if (!semanticContract)
    return semanticContract.takeError();
  const std::string top = representation.top.canonicalName;
  return InvocationFacts{std::move(*gate),
                         std::move(*target),
                         std::move(*config),
                         std::move(*semanticContract),
                         std::move(materialized),
                         std::move(netlists.front()),
                         std::move(constraints.front()),
                         top};
}

ExternalToolInvocationImportExpectation
expectation(const InvocationFacts &facts) {
  ExternalToolInvocationImportExpectation result;
  result.semanticContract = facts.semanticContract;
  for (const MaterializedBundleFile &file : facts.semanticInputs) {
    result.semanticInputs.push_back(ExternalToolInvocationSemanticInput{
        file.relativePath, *file.sourceArtifact,
        computeBlobDigest(llvm::ArrayRef<std::uint8_t>(
            reinterpret_cast<const std::uint8_t *>(file.contents.data()),
            file.contents.size()))});
  }
  result.externalInputs = {
      {earlyParasiticSlot.str(), facts.config.earlyParasiticTech()},
      {lateParasiticSlot.str(), facts.config.lateParasiticTech()},
      {layerMapSlot.str(), facts.config.parasiticLayerMap()},
  };
  result.externalFileTrees.push_back(
      {referenceLibrarySlot.str(),
       facts.config.referenceLibraryMembers().vec()});
  for (llvm::StringLiteral output : fusionCompilerDescriptor().declaredOutputs)
    result.declaredOutputs.push_back(output.str());
  return result;
}

CandidateGeneratorProviderResult
incompleteResult(CandidateGeneratorIncompleteReason reason) {
  return CandidateGeneratorProviderResult{
      IncompleteCandidateGeneratorResult{
          reason, {{CandidateGeneratorOutputSlotRef(0), {}}}, {}},
      {{CandidateGeneratorWorkUnitRef(0), 1, 1}}};
}

llvm::Expected<PreparedExternalToolInvocation> prepareProviderWithContracts(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    const ResolvedCandidateGeneratorBinding &binding,
    const ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const ExternalToolPreparationContext &context) {
  auto facts = invocationFacts(inputs, binding, contracts, artifacts, blobs);
  if (!facts)
    return facts.takeError();
  auto externalFiles = resolveExternalFiles(
      {{earlyParasiticSlot.str(), facts->config.earlyParasiticTech()},
       {lateParasiticSlot.str(), facts->config.lateParasiticTech()},
       {layerMapSlot.str(), facts->config.parasiticLayerMap()}},
      context.localConfig);
  if (!externalFiles)
    return externalFiles.takeError();
  auto externalTrees = resolveExternalFileTrees(
      {{referenceLibrarySlot.str(),
        facts->config.referenceLibraryMembers().vec()}},
      context.localConfig);
  if (!externalTrees)
    return externalTrees.takeError();
  if (externalFiles->size() != 3 || externalTrees->size() != 1)
    return invalid("resolved provider input closure has the wrong size");

  const ExternalToolProviderDescriptor &toolProvider = fusionCompilerProvider();
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
    return invalid(llvm::Twine("resolved Fusion Compiler build '") +
                   tool->version + "' does not match semantic build '" +
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

  SynopsysFrozenInvocation frozen{
      std::move(*tool),
      toolProvider.versionProbe,
      std::move(*runtime),
      containerProvider.versionProbe,
      std::move(inheritEnvironment),
      std::move(*externalFiles),
      std::move(*externalTrees),
  };
  SynopsysBundleInputs bundleInputs{
      facts->semanticContract,
      &facts->gate.implementation().representationRoot(),
      facts->gate.implementation().implementationPlatform(),
      &facts->platform,
      platform::encodeTechnologyCornerRef(facts->config.technologyCorner()),
      std::move(frozen),
      facts->semanticInputs,
  };
  if (llvm::Error error = validateSynopsysSemanticInputs(
          fusionCompilerDescriptor(), bundleInputs,
          {facts->netlistPath, facts->constraintPath}))
    return error;

  const ResolvedExternalFileTree &library =
      bundleInputs.frozen.externalFileTrees.front();
  const auto findFile =
      [&](llvm::StringRef slot) -> const ResolvedExternalFile * {
    const auto found = llvm::find_if(bundleInputs.frozen.externalFiles,
                                     [&](const ResolvedExternalFile &file) {
                                       return file.providerInputSlot == slot;
                                     });
    return found == bundleInputs.frozen.externalFiles.end() ? nullptr : &*found;
  };
  const ResolvedExternalFile *early = findFile(earlyParasiticSlot);
  const ResolvedExternalFile *late = findFile(lateParasiticSlot);
  const ResolvedExternalFile *layerMap = findFile(layerMapSlot);
  if (!early || !late || !layerMap)
    return invalid("resolved provider file closure changed slot identity");
  auto driver = renderFusionCompilerDriver(
      facts->top, facts->netlistPath, facts->constraintPath, floorplanPath,
      library.absolutePath, early->absolutePath, late->absolutePath,
      layerMap->absolutePath);
  if (!driver)
    return driver.takeError();
  const std::string executable = bundleInputs.frozen.tool.executable;
  auto specification = makeSynopsysInvocationBundleSpec(
      fusionCompilerDescriptor(), bundleInputs,
      {{executable, "-f", "drivers/fusion-compiler.tcl"}},
      {{"drivers/fusion-compiler.tcl", std::move(*driver), std::nullopt, false},
       {floorplanPath.str(), facts->config.floorplanDef().str(), std::nullopt,
        false}});
  if (!specification)
    return specification.takeError();
  return finalizeExternalToolInvocationBundle(context.bundleDestination,
                                              *specification);
}

llvm::Expected<CandidateGeneratorProviderResult> importProviderWithContracts(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    const ResolvedCandidateGeneratorBinding &binding,
    const PreparedExternalToolInvocation &prepared,
    const ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const ExternalToolInvocationExecutionObservation *execution) {
  auto facts = invocationFacts(inputs, binding, contracts, artifacts, blobs);
  if (!facts)
    return facts.takeError();
  ExternalToolInvocationImportExpectation importExpectation =
      expectation(*facts);
  auto attempt =
      execution
          ? importExternalToolInvocationAttempt(prepared, importExpectation,
                                                *execution)
          : importExternalToolInvocationAttempt(prepared, importExpectation);
  if (!attempt)
    return attempt.takeError();
  if (std::holds_alternative<IncompleteExternalToolInvocationAttempt>(*attempt))
    return llvm::make_error<IncompleteExternalToolInvocationError>();
  if (const auto *failed =
          std::get_if<FailedExternalToolInvocationAttempt>(&*attempt)) {
    switch (failed->status) {
    case InvocationCompletionStatus::Success:
      return invalid("failed invocation outcome carries success status");
    case InvocationCompletionStatus::MissingEnvironment:
    case InvocationCompletionStatus::ModuleActivationFailed:
    case InvocationCompletionStatus::VersionMismatch:
      return incompleteResult(
          CandidateGeneratorIncompleteReason::ProviderUnavailable);
    case InvocationCompletionStatus::BundleContentMismatch:
      return invalid("invocation bundle content changed before execution");
    case InvocationCompletionStatus::ToolExit:
    case InvocationCompletionStatus::MissingOutput:
      return incompleteResult(
          CandidateGeneratorIncompleteReason::ExecutionFailed);
    }
  }
  ImportedExternalToolInvocationBundle imported =
      std::get<ImportedExternalToolInvocationBundle>(std::move(*attempt));
  if (llvm::Error error = validateSynopsysOutputInventory(
          fusionCompilerDescriptor(), prepared.bundleRoot))
    return error;
  auto netlist = readSynopsysDeclaredOutput(
      fusionCompilerDescriptor(), imported, "outputs/fusion-compiler-routed.v");
  if (!netlist)
    return netlist.takeError();
  auto def = readSynopsysDeclaredOutput(fusionCompilerDescriptor(), imported,
                                        "outputs/fusion-compiler-routed.def");
  if (!def)
    return def.takeError();
  auto constraints =
      readSynopsysDeclaredOutput(fusionCompilerDescriptor(), imported,
                                 "outputs/fusion-compiler-routed.sdc");
  if (!constraints)
    return constraints.takeError();
  auto snapshot = parseFusionCompilerPhysicalSnapshot(
      *netlist, *def, *constraints, facts->top,
      RepresentationPhysicalStage::Routed);
  if (!snapshot)
    return snapshot.takeError();
  auto published = publishFusionCompilerPhysicalImplementation(
      facts->gate, *snapshot, contracts, artifacts, blobs);
  if (!published)
    return published.takeError();
  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{
          {{CandidateGeneratorOutputSlotRef(0), {published->reference()}}},
          {{CandidateGeneratorLineageEdgeKind::MechanicalDerivation,
            CandidateGeneratorOutputSlotRef(0),
            published->reference(),
            {},
            {}}}},
      {{CandidateGeneratorWorkUnitRef(0), 1, 1}}};
}

} // namespace

llvm::ArrayRef<std::uint8_t>
resolvedFusionCompilerRoutedConfigSchemaDescriptorBytes() {
  return schemaBytes();
}

llvm::Expected<ResolvedFusionCompilerRoutedConfigView>
createResolvedFusionCompilerRoutedConfigView(
    llvm::StringRef stableProviderBuildIdentity,
    platform::TechnologyCornerRef technologyCorner,
    std::vector<ExternalFileTreeMember> referenceLibraryMembers,
    ExternalFileFingerprint earlyParasiticTech,
    ExternalFileFingerprint lateParasiticTech,
    ExternalFileFingerprint parasiticLayerMap, llvm::StringRef floorplanDef) {
  if (llvm::Error error = validateConfigValues(
          stableProviderBuildIdentity, referenceLibraryMembers, floorplanDef))
    return error;
  std::vector<std::uint8_t> bytes = encodeConfig(
      stableProviderBuildIdentity, technologyCorner, referenceLibraryMembers,
      earlyParasiticTech, lateParasiticTech, parasiticLayerMap, floorplanDef);
  auto digest = computeComponentViewDigest(schemaBytes(), bytes);
  if (!digest)
    return digest.takeError();
  return ResolvedFusionCompilerRoutedConfigView(
      stableProviderBuildIdentity.str(), std::move(technologyCorner),
      std::move(referenceLibraryMembers), std::move(earlyParasiticTech),
      std::move(lateParasiticTech), std::move(parasiticLayerMap),
      floorplanDef.str(), std::move(bytes), std::move(*digest));
}

llvm::Expected<ResolvedFusionCompilerRoutedConfigView>
adoptResolvedFusionCompilerRoutedConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest) {
  if (schemaDescriptorBytes != schemaBytes())
    return invalid("config descriptor does not match the exact owner");
  if (llvm::Error error = validateComponentViewDigest(
          schemaDescriptorBytes, canonicalViewBytes, digest))
    return error;
  Reader reader(canonicalViewBytes);
  auto providerBuild = reader.string32();
  if (!providerBuild)
    return providerBuild.takeError();
  auto artifactBytes = reader.take(ArtifactIdentity::byteSize);
  if (!artifactBytes)
    return artifactBytes.takeError();
  auto artifact = ArtifactIdentity::fromBytes(*artifactBytes);
  if (!artifact)
    return artifact.takeError();
  auto cornerId = reader.u64();
  if (!cornerId)
    return cornerId.takeError();
  auto memberCount = reader.u32();
  if (!memberCount)
    return memberCount.takeError();
  constexpr std::size_t minimumMemberSize =
      sizeof(std::uint32_t) + 1 + ExternalFileFingerprint::byteSize;
  if (*memberCount > reader.remaining() / minimumMemberSize)
    return invalid("reference library member count exceeds config bytes");
  std::vector<ExternalFileTreeMember> members;
  members.reserve(*memberCount);
  for (std::uint32_t index = 0; index < *memberCount; ++index) {
    auto path = reader.string32();
    if (!path)
      return path.takeError();
    auto fingerprintBytes = reader.take(ExternalFileFingerprint::byteSize);
    if (!fingerprintBytes)
      return fingerprintBytes.takeError();
    auto fingerprint = ExternalFileFingerprint::fromBytes(*fingerprintBytes);
    if (!fingerprint)
      return fingerprint.takeError();
    members.push_back({std::move(*path), std::move(*fingerprint)});
  }
  std::vector<ExternalFileFingerprint> fingerprints;
  fingerprints.reserve(3);
  for (unsigned index = 0; index != 3; ++index) {
    auto bytes = reader.take(ExternalFileFingerprint::byteSize);
    if (!bytes)
      return bytes.takeError();
    auto parsed = ExternalFileFingerprint::fromBytes(*bytes);
    if (!parsed)
      return parsed.takeError();
    fingerprints.push_back(std::move(*parsed));
  }
  auto floorplanSize = reader.u64();
  if (!floorplanSize)
    return floorplanSize.takeError();
  if (*floorplanSize > std::numeric_limits<std::size_t>::max())
    return invalid("floorplan size is not representable");
  auto floorplanBytes = reader.take(static_cast<std::size_t>(*floorplanSize));
  if (!floorplanBytes)
    return floorplanBytes.takeError();
  if (reader.remaining() != 0)
    return invalid("config bytes have trailing data");
  const std::string floorplan(
      reinterpret_cast<const char *>(floorplanBytes->data()),
      floorplanBytes->size());
  if (llvm::Error error =
          validateConfigValues(*providerBuild, members, floorplan))
    return error;
  const platform::TechnologyCornerRef corner{
      std::move(*artifact), platform::TechnologyCornerId(*cornerId)};
  std::vector<std::uint8_t> canonical =
      encodeConfig(*providerBuild, corner, members, fingerprints[0],
                   fingerprints[1], fingerprints[2], floorplan);
  if (llvm::ArrayRef<std::uint8_t>(canonical) != canonicalViewBytes)
    return invalid("decoded config does not re-encode canonically");
  return ResolvedFusionCompilerRoutedConfigView(
      std::move(*providerBuild), corner, std::move(members),
      std::move(fingerprints[0]), std::move(fingerprints[1]),
      std::move(fingerprints[2]), floorplan, std::move(canonical), digest);
}

const CandidateGeneratorDescriptor &
fusionCompilerRoutedCandidateGeneratorDescriptor() {
  return generatorDescriptor();
}

llvm::Error registerFusionCompilerRoutedCandidateGenerator() {
  const CandidateGeneratorDescriptor &descriptor = generatorDescriptor();
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(generatorProvider());
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindFusionCompilerRoutedInputs(
    const ArtifactRootReference &gateNetlistImplementation,
    const ArtifactRootReference &implementationPlatform) {
  if (llvm::Error error = registerFusionCompilerRoutedCandidateGenerator())
    return error;
  std::vector<CandidateGeneratorInputBinding> inputs{
      {CandidateGeneratorInputSlotRef(GateNetlistInput),
       {gateNetlistImplementation}},
      {CandidateGeneratorInputSlotRef(AsicTargetInput),
       {implementationPlatform}},
  };
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          generatorDescriptor().reference(), inputs))
    return error;
  return inputs;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveFusionCompilerRoutedBinding(
    const ResolvedFusionCompilerRoutedConfigView &config) {
  if (llvm::Error error = registerFusionCompilerRoutedCandidateGenerator())
    return error;
  return ResolvedCandidateGeneratorBinding::get(
      generatorDescriptor().reference(), config.canonicalViewBytes(),
      config.digest());
}

llvm::Expected<PreparedExternalToolInvocation>
prepareFusionCompilerRoutedInvocation(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    const ResolvedCandidateGeneratorBinding &binding,
    const ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const ExternalToolPreparationContext &context) {
  return prepareProviderWithContracts(inputs, binding, contracts, artifacts,
                                      blobs, context);
}

llvm::Expected<CandidateGeneratorProviderResult>
importFusionCompilerRoutedInvocation(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    const ResolvedCandidateGeneratorBinding &binding,
    const PreparedExternalToolInvocation &prepared,
    const ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  return importProviderWithContracts(inputs, binding, prepared, contracts,
                                     artifacts, blobs);
}

llvm::Expected<FinalizedHardwareImplementation>
importFusionCompilerRoutedImplementation(const ArtifactRootReference &reference,
                                         const ArtifactStore &artifacts,
                                         const BlobStore &blobs) {
  auto catalog = makeSynopsysStandardCellContractCatalog();
  if (!catalog)
    return catalog.takeError();
  return importHardwareImplementation(reference, *catalog, artifacts, blobs);
}

} // namespace loom::eda::synopsys
