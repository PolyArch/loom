#include "EDA/Adapters/OpenSource/YosysBlock.h"

#include "YosysSynthesisInvocation.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "EDA/Adapters/AsicStandardCellContracts.h"
#include "Hardware/Implementation/RepresentationIndex.h"
#include "Hardware/RTL/BlockGateNetlistComposition.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include <array>
#include <map>

namespace loom::eda::open_source {
namespace {

using namespace dse;
using namespace external_tool;
using namespace hardware;
using namespace hardware::rtl;

constexpr CandidateGeneratorInputSlotRef blockInput(0);
constexpr CandidateGeneratorInputSlotRef targetInput(1);
constexpr CandidateGeneratorInputSlotRef childrenInput(2);
constexpr CandidateGeneratorOutputSlotRef netlistOutput(0);
constexpr CandidateGeneratorWorkUnitRef synthesisWork(0);
constexpr llvm::StringLiteral rtlPath = "inputs/block/source.sv";
constexpr llvm::StringLiteral constraintPath = "inputs/block/clock.sdc";
constexpr llvm::StringLiteral contractPayloadPath =
    "contracts/standard-cells.txt";
constexpr std::array<CandidateGeneratorInputSlotDescriptor, 2> inputSlots{
    {{blockInput, "reusable_block_source", PlanValueRole::CandidateSet,
      &rtlBlockSourceSchema, PlanValueCardinality::ExactlyOne},
     {targetInput, "asic_target", PlanValueRole::CandidateSet,
      &platform::implementationPlatformSchema,
      PlanValueCardinality::ExactlyOne}}};
constexpr std::array<CandidateGeneratorInputSlotDescriptor, 3> parentInputSlots{
    {inputSlots[0],
     inputSlots[1],
     {childrenInput, "compiled_children", PlanValueRole::CandidateSet,
      &blockGateNetlistSchema, PlanValueCardinality::NonEmptySet}}};
constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputSlots{
    {{netlistOutput, "block_gate_netlist", PlanValueRole::CandidateSet,
      &blockGateNetlistSchema, PlanValueCardinality::ExactlyOne}}};
constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 1> work{
    {{synthesisWork, "block_synthesis_attempt"}}};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "yosys_block_invalid: " + message);
}

llvm::ArrayRef<std::uint8_t> bytesOf(llvm::StringRef text) {
  return {reinterpret_cast<const std::uint8_t *>(text.data()), text.size()};
}

std::string textOf(llvm::ArrayRef<std::uint8_t> bytes) {
  return {reinterpret_cast<const char *>(bytes.data()), bytes.size()};
}

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted = adoptResolvedYosysGateNetlistConfigView(
      resolvedYosysGateNetlistConfigSchemaDescriptorBytes(), bytes, digest);
  return adopted ? llvm::Error::success() : adopted.takeError();
}

const CandidateGeneratorDescriptor descriptor{
    yosysBlockGateNetlistCandidateGeneratorKind,
    "open_source.yosys.block_gate_netlist",
    "loom.eda.open_source.yosys.block_gate_netlist.generator.v1",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{
        resolvedYosysGateNetlistConfigSchemaDescriptorBytes(), validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    work,
    nullptr,
    ProviderForm::ExternalPrepareImport};

const CandidateGeneratorDescriptor parentDescriptor{
    yosysHierarchicalBlockGateNetlistCandidateGeneratorKind,
    "open_source.yosys.hierarchical_block_gate_netlist",
    "loom.eda.open_source.yosys.hierarchical_block_gate_netlist."
    "generator.v1",
    parentInputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{
        resolvedYosysGateNetlistConfigSchemaDescriptorBytes(), validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    work,
    nullptr,
    ProviderForm::ExternalPrepareImport};

struct ParentComposition final {
  YosysMappedChildren mapped;
  std::vector<ImplementationPayload> payloads;
  std::map<std::string, std::uint64_t> multiplicities;
  std::vector<BlockGateNetlistChildBoundary> boundaries;
};

llvm::Expected<ParentComposition>
deriveParentComposition(const FinalizedRtlBlockSource &source,
                        const platform::FinalizedImplementationPlatform &target,
                        const ResolvedYosysGateNetlistConfigView &config,
                        llvm::ArrayRef<ArtifactRootReference> children,
                        const ArtifactStore &artifacts, const BlobStore &blobs,
                        std::vector<MaterializedBundleFile> &files) {
  std::vector<FinalizedBlockGateNetlist> products;
  for (const auto &reference : children) {
    auto product = importYosysBlockGateNetlist(reference, artifacts, blobs);
    if (!product)
      return product.takeError();
    products.push_back(std::move(*product));
  }
  auto composition =
      composeBlockGateNetlistChildren(source, products, artifacts, blobs);
  if (!composition)
    return composition.takeError();
  const auto &technology = products.front().netlist();
  if (technology.implementationPlatform != target.reference() ||
      technology.corner != config.technologyCorner() ||
      technology.standardCellLibrary != config.standardCellLiberty())
    return invalid("compiled child has another platform, corner or library");
  ParentComposition result;
  for (const auto &child : composition->children) {
    result.multiplicities.emplace(child.definition, child.multiplicity);
    result.mapped.directModuleNames.push_back(child.definition);
  }
  result.boundaries = std::move(composition->children);
  for (const auto &unit : composition->units) {
    result.payloads.push_back(unit.payload);
    auto bytes = blobs.get(unit.payload.blobDigest);
    if (!bytes)
      return bytes.takeError();
    const std::string path = "inputs/children/" +
                             formatBlobDigestHex(unit.payload.blobDigest) +
                             ".v";
    result.mapped.netlistPaths.push_back(path);
    files.push_back({path, textOf(*bytes), unit.contributor, false});
  }
  llvm::sort(result.mapped.netlistPaths);

  return result;
}

struct InvocationFacts final {
  FinalizedRtlBlockSource source;
  platform::FinalizedImplementationPlatform target;
  ResolvedYosysGateNetlistConfigView config;
  ExternalToolSemanticContract contract;
  std::vector<MaterializedBundleFile> files;
  std::optional<ParentComposition> parent;
};

llvm::Expected<InvocationFacts>
facts(llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
      const ResolvedCandidateGeneratorBinding &binding,
      const ArtifactStore &artifacts, const BlobStore &blobs) {
  const bool hierarchical =
      binding.descriptorRef() == parentDescriptor.reference();
  if (!hierarchical && binding.descriptorRef() != descriptor.reference())
    return invalid("binding names another generator");
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          binding.descriptorRef(), inputs))
    return std::move(error);
  auto config = adoptResolvedYosysGateNetlistConfigView(
      resolvedYosysGateNetlistConfigSchemaDescriptorBytes(),
      binding.canonicalConfigBytes(), binding.configDigest());
  if (!config)
    return config.takeError();
  auto source = importRtlBlockSource(
      inputs[blockInput.ordinal()].artifacts.front(), artifacts, blobs);
  if (!source)
    return source.takeError();
  auto target = platform::importImplementationPlatform(
      inputs[targetInput.ordinal()].artifacts.front(), artifacts);
  if (!target)
    return target.takeError();
  if (!std::holds_alternative<platform::AsicTarget>(
          target->platform().target()) ||
      config->technologyCorner().artifact != target->reference().artifact ||
      !target->platform().findTechnologyCorner(
          config->technologyCorner().entity))
    return invalid(
        "configured corner does not belong to the exact ASIC target");
  auto sourceArtifact = artifacts.get(source->reference());
  if (!sourceArtifact)
    return sourceArtifact.takeError();
  std::vector<MaterializedBundleFile> files{
      {rtlPath.str(), source->projection().source, source->reference(), false},
      {"inputs/block/source-artifact.json", textOf(sourceArtifact->bytes()),
       source->reference(), false},
      {"inputs/target/implementation-platform.json",
       textOf(target->canonicalBytes().bytes()), target->reference(), false}};
  const std::string constraint = source->generationConstraint();
  if (!constraint.empty()) {
    files.push_back(
        {constraintPath.str(), constraint, source->reference(), false});
  }
  std::optional<ParentComposition> parent;
  if (hierarchical) {
    auto composition = deriveParentComposition(
        *source, *target, *config, inputs[childrenInput.ordinal()].artifacts,
        artifacts, blobs, files);
    if (!composition)
      return composition.takeError();
    parent = std::move(*composition);
    auto bound = bindRtlModuleGraphSource(source->projection().graph,
                                          source->projection().source);
    if (!bound)
      return bound.takeError();
    files.front().contents =
        bound->preamble().str() +
        bound->moduleBytes()[source->projection().graph.topModule].str();
  }
  if (!hierarchical && !source->projection()
                            .graph.modules[source->projection().graph.topModule]
                            .dependencies.empty())
    return invalid("a non-leaf Source requires the hierarchical provider");
  auto semantic = deriveExternalToolSemanticContract(inputs, binding);
  if (!semantic)
    return semantic.takeError();
  return InvocationFacts{std::move(*source), std::move(*target),
                         std::move(*config), std::move(*semantic),
                         std::move(files),   std::move(parent)};
}

llvm::Expected<PreparedExternalToolInvocation>
prepare(llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
        const ResolvedCandidateGeneratorBinding &binding,
        const ArtifactStore &artifacts, const BlobStore &blobs,
        const ExternalToolPreparationContext &context) {
  auto invocation = facts(inputs, binding, artifacts, blobs);
  if (!invocation)
    return invocation.takeError();
  const YosysMappedChildren leaf;
  return prepareYosysSynthesisInvocation(
      invocation->config, invocation->contract, invocation->files,
      invocation->source.top(), {rtlPath.str()},
      invocation->parent ? &invocation->parent->mapped : &leaf, context);
}

llvm::Error
validateMappedChildInterfaces(const ParentComposition &parent,
                              const YosysStructureFacts &structure) {
  for (const auto &child : parent.boundaries) {
    auto module = structure.modules.find(child.definition);
    if (module == structure.modules.end() || !module->second.declaredBox ||
        module->second.ports.size() != child.ports.size())
      return invalid("native child library view changed the exact interface");
    for (const auto &port : child.ports) {
      llvm::StringRef name(port.locator.canonicalName);
      if (!name.consume_front(child.definition + "."))
        return invalid("child public port does not belong to its exact root");
      auto observed = module->second.ports.find(name.str());
      if (observed == module->second.ports.end() ||
          observed->second.bits.size() != port.geometry.bitWidth)
        return invalid("native child library view changed a public port width");
      RepresentationSignalDirection direction;
      switch (observed->second.direction) {
      case YosysPortGeometry::Direction::Input:
        direction = RepresentationSignalDirection::Input;
        break;
      case YosysPortGeometry::Direction::Output:
        direction = RepresentationSignalDirection::Output;
        break;
      case YosysPortGeometry::Direction::Inout:
        direction = RepresentationSignalDirection::Inout;
        break;
      }
      if (direction != port.geometry.direction)
        return invalid(
            "native child library view changed a public port direction");
    }
  }
  return llvm::Error::success();
}

llvm::Expected<ArtifactRootReference> publish(const InvocationFacts &facts,
                                              llvm::StringRef verilog,
                                              const ArtifactStore &artifacts,
                                              const BlobStore &blobs) {
  auto digest = blobs.put(bytesOf(verilog));
  if (!digest)
    return digest.takeError();
  auto format = RepresentationFormatDescriptorRef::get(
      RepresentationFormatKind::StructuralVerilogGateNetlist);
  if (!format)
    return format.takeError();
  std::vector<ImplementationPayload> payloads{
      {PayloadRole::Netlist, "netlist/block.v", *digest}};
  if (facts.parent)
    payloads.insert(payloads.end(), facts.parent->payloads.begin(),
                    facts.parent->payloads.end());
  const std::string constraint = facts.source.generationConstraint();
  if (!constraint.empty()) {
    auto clock = blobs.put(bytesOf(constraint));
    if (!clock)
      return clock.takeError();
    payloads.push_back(
        {PayloadRole::GenerationConstraint, "constraints/clock.sdc", *clock});
  }
  RepresentationLocator top{RepresentationObjectKind::Module,
                            facts.source.top().str()};
  auto canonicalPayloads = canonicalizeImplementationPayloadCatalog(payloads);
  if (!canonicalPayloads)
    return canonicalPayloads.takeError();
  payloads = std::move(*canonicalPayloads);
  auto index = indexRepresentation(*format, top, payloads, blobs);
  if (!index)
    return index.takeError();
  if (facts.parent) {
    std::map<std::string, std::uint64_t> observed;
    for (const auto &instance : index->rootModuleInstanceBindings())
      if (facts.parent->multiplicities.count(instance.definition.canonicalName))
        ++observed[instance.definition.canonicalName];
    if (observed != facts.parent->multiplicities)
      return invalid("synthesis changed the exact direct child multiplicities");
  }
  auto contract = renderYosysStandardCellBlackBoxContract(
      facts.config.standardCellLiberty(),
      index->unresolvedExternalDefinitions());
  auto contractDigest = blobs.put(bytesOf(contract));
  if (!contractDigest)
    return contractDigest.takeError();
  payloads.push_back({PayloadRole::BlackBoxContract, contractPayloadPath.str(),
                      *contractDigest});
  auto root = createImplementationRepresentationRoot(
      RepresentationRootVariant::GateNetlist, std::nullopt, *format,
      std::move(top), std::move(payloads));
  if (!root)
    return root.takeError();
  auto contracts = makeYosysStandardCellContractCatalog();
  if (!contracts)
    return contracts.takeError();
  auto result = finalizeBlockGateNetlist(
      {facts.source.reference(), facts.target.reference(),
       facts.config.technologyCorner(),
       openSourceYosysStandardCellContractRef.str(),
       facts.config.standardCellLiberty(), std::move(*root)},
      *contracts, artifacts, blobs);
  if (!result)
    return result.takeError();
  auto imported =
      importYosysBlockGateNetlist(result->reference(), artifacts, blobs);
  if (!imported)
    return imported.takeError();
  return imported->reference();
}

CandidateGeneratorProviderResult
incomplete(CandidateGeneratorIncompleteReason reason) {
  return CandidateGeneratorProviderResult{
      IncompleteCandidateGeneratorResult{reason, {{netlistOutput, {}}}, {}},
      {{synthesisWork, 1, 1}}};
}

llvm::Expected<CandidateGeneratorProviderResult>
import(llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
       const ResolvedCandidateGeneratorBinding &binding,
       const PreparedExternalToolInvocation &prepared,
       const ArtifactStore &artifacts, const BlobStore &blobs,
       const ExternalToolInvocationExecutionObservation *execution) {
  auto invocation = facts(inputs, binding, artifacts, blobs);
  if (!invocation)
    return invocation.takeError();
  auto expected = yosysSynthesisInvocationExpectation(
      invocation->contract, invocation->files, invocation->config);
  auto attempt =
      execution
          ? importExternalToolInvocationAttempt(prepared, expected, *execution)
          : importExternalToolInvocationAttempt(prepared, expected);
  if (!attempt)
    return attempt.takeError();
  if (std::holds_alternative<IncompleteExternalToolInvocationAttempt>(*attempt))
    return llvm::make_error<IncompleteExternalToolInvocationError>();
  if (const auto *failed =
          std::get_if<FailedExternalToolInvocationAttempt>(&*attempt)) {
    switch (failed->status) {
    case InvocationCompletionStatus::Success:
      return invalid("failed attempt carries a success status");
    case InvocationCompletionStatus::MissingEnvironment:
    case InvocationCompletionStatus::ModuleActivationFailed:
    case InvocationCompletionStatus::VersionMismatch:
      return incomplete(
          CandidateGeneratorIncompleteReason::ProviderUnavailable);
    case InvocationCompletionStatus::BundleContentMismatch:
      return invalid("invocation input material changed before execution");
    case InvocationCompletionStatus::ToolExit:
    case InvocationCompletionStatus::MissingOutput:
      return incomplete(CandidateGeneratorIncompleteReason::ExecutionFailed);
    }
  }
  auto imported =
      std::get<ImportedExternalToolInvocationBundle>(std::move(*attempt));
  auto output =
      readYosysSynthesisOutput(prepared, imported, invocation->source.top());
  if (!output)
    return output.takeError();
  if (invocation->parent)
    if (llvm::Error error = validateMappedChildInterfaces(*invocation->parent,
                                                          output->structure))
      return std::move(error);
  auto result = publish(*invocation, output->netlist, artifacts, blobs);
  if (!result)
    return result.takeError();
  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{
          {{netlistOutput, {*result}}},
          {{CandidateGeneratorLineageEdgeKind::MechanicalDerivation,
            netlistOutput,
            *result,
            {},
            {}}}},
      {{synthesisWork, 1, 1}}};
}

llvm::Expected<CandidateGeneratorProviderResult>
importWithoutExecution(llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
                       const ResolvedCandidateGeneratorBinding &binding,
                       const PreparedExternalToolInvocation &prepared,
                       const ArtifactStore &artifacts, const BlobStore &blobs) {
  return import(inputs, binding, prepared, artifacts, blobs, nullptr);
}

llvm::Expected<CandidateGeneratorProviderResult>
importWithExecution(llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
                    const ResolvedCandidateGeneratorBinding &binding,
                    const PreparedExternalToolInvocation &prepared,
                    const ExternalToolInvocationExecutionObservation &execution,
                    const ArtifactStore &artifacts, const BlobStore &blobs) {
  return import(inputs, binding, prepared, artifacts, blobs, &execution);
}

const CandidateGeneratorProvider provider{
    descriptor.reference(),
    CandidateGeneratorExternalPrepareImportProvider{
        prepare, importWithoutExecution, importWithExecution}};
const CandidateGeneratorProvider parentProvider{
    parentDescriptor.reference(),
    CandidateGeneratorExternalPrepareImportProvider{
        prepare, importWithoutExecution, importWithExecution}};

} // namespace

const dse::CandidateGeneratorDescriptor &
yosysBlockGateNetlistCandidateGeneratorDescriptor() {
  return descriptor;
}

llvm::Error registerYosysBlockGateNetlistCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindYosysBlockGateNetlistInputs(
    const ArtifactRootReference &blockSource,
    const ArtifactRootReference &implementationPlatform) {
  if (llvm::Error error = registerYosysBlockGateNetlistCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> inputs{
      {blockInput, {blockSource}}, {targetInput, {implementationPlatform}}};
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), inputs))
    return std::move(error);
  return inputs;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveYosysBlockGateNetlistBinding(
    const ResolvedYosysGateNetlistConfigView &config) {
  if (llvm::Error error = registerYosysBlockGateNetlistCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

const dse::CandidateGeneratorDescriptor &
yosysHierarchicalBlockGateNetlistCandidateGeneratorDescriptor() {
  return parentDescriptor;
}

llvm::Error registerYosysHierarchicalBlockGateNetlistCandidateGenerator() {
  if (llvm::Error error =
          registerCandidateGeneratorDescriptor(parentDescriptor))
    return error;
  return registerCandidateGeneratorProvider(parentProvider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindYosysHierarchicalBlockGateNetlistInputs(
    const ArtifactRootReference &blockSource,
    const ArtifactRootReference &implementationPlatform,
    llvm::ArrayRef<ArtifactRootReference> compiledChildren) {
  if (llvm::Error error =
          registerYosysHierarchicalBlockGateNetlistCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> inputs{
      {blockInput, {blockSource}},
      {targetInput, {implementationPlatform}},
      {childrenInput, compiledChildren.vec()}};
  llvm::sort(inputs.back().artifacts, artifactRootReferenceLess);
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          parentDescriptor.reference(), inputs))
    return std::move(error);
  return inputs;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveYosysHierarchicalBlockGateNetlistBinding(
    const ResolvedYosysGateNetlistConfigView &config) {
  if (llvm::Error error =
          registerYosysHierarchicalBlockGateNetlistCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(parentDescriptor.reference(),
                                                config.canonicalViewBytes(),
                                                config.digest());
}

llvm::Expected<FinalizedBlockGateNetlist>
importYosysBlockGateNetlist(const ArtifactRootReference &reference,
                            const ArtifactStore &artifacts,
                            const BlobStore &blobs) {
  auto contracts = makeYosysStandardCellContractCatalog();
  if (!contracts)
    return contracts.takeError();
  auto result = importBlockGateNetlist(reference, *contracts, artifacts, blobs);
  if (!result)
    return result.takeError();
  const auto &netlist = result->netlist();
  if (netlist.standardCellContract != openSourceYosysStandardCellContractRef)
    return invalid("mapped library belongs to another provider contract");
  auto index = indexRepresentationRoot(netlist.representation, blobs);
  if (!index)
    return index.takeError();
  auto expected = renderYosysStandardCellBlackBoxContract(
      netlist.standardCellLibrary, index->unresolvedExternalDefinitions());
  for (const auto &payload : netlist.representation.payloads)
    if (payload.role == PayloadRole::BlackBoxContract) {
      auto bytes = blobs.get(payload.blobDigest);
      if (!bytes)
        return bytes.takeError();
      if (textOf(*bytes) != expected)
        return invalid(
            "mapped library contract differs from the exact netlist closure");
    }
  return std::move(*result);
}

} // namespace loom::eda::open_source
