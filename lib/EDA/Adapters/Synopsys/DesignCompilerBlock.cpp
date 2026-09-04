#include "EDA/Adapters/Synopsys/DesignCompilerBlock.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "EDA/Adapters/AsicStandardCellContracts.h"
#include "Hardware/Implementation/RepresentationIndex.h"

#include <array>
#include <map>

namespace loom::eda::synopsys {
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
                                 "design_compiler_block_invalid: " + message);
}

llvm::ArrayRef<std::uint8_t> bytesOf(llvm::StringRef text) {
  return {reinterpret_cast<const std::uint8_t *>(text.data()), text.size()};
}

std::string textOf(llvm::ArrayRef<std::uint8_t> bytes) {
  return {reinterpret_cast<const char *>(bytes.data()), bytes.size()};
}

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted = adoptResolvedDesignCompilerGateNetlistConfigView(
      resolvedDesignCompilerGateNetlistConfigSchemaDescriptorBytes(), bytes,
      digest);
  return adopted ? llvm::Error::success() : adopted.takeError();
}

const CandidateGeneratorDescriptor descriptor{
    designCompilerBlockGateNetlistCandidateGeneratorKind,
    "synopsys.design_compiler.block_gate_netlist",
    "loom.eda.synopsys.design_compiler.block_gate_netlist.generator.v1",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{
        resolvedDesignCompilerGateNetlistConfigSchemaDescriptorBytes(),
        validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    work,
    nullptr,
    ProviderForm::ExternalPrepareImport};

const CandidateGeneratorDescriptor parentDescriptor{
    designCompilerHierarchicalBlockGateNetlistCandidateGeneratorKind,
    "synopsys.design_compiler.hierarchical_block_gate_netlist",
    "loom.eda.synopsys.design_compiler.hierarchical_block_gate_netlist."
    "generator.v1",
    parentInputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{
        resolvedDesignCompilerGateNetlistConfigSchemaDescriptorBytes(),
        validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    work,
    nullptr,
    ProviderForm::ExternalPrepareImport};

struct ParentComposition final {
  DesignCompilerMappedChildren mapped;
  std::vector<ImplementationPayload> payloads;
  std::map<std::string, std::uint64_t> multiplicities;
};

llvm::Expected<ParentComposition> deriveParentComposition(
    const FinalizedRtlBlockSource &source,
    const platform::FinalizedImplementationPlatform &target,
    const ResolvedDesignCompilerGateNetlistConfigView &config,
    llvm::ArrayRef<ArtifactRootReference> children,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    std::vector<MaterializedBundleFile> &files) {
  const auto &graph = source.projection().graph;
  const auto &dependencies = graph.modules[graph.topModule].dependencies;
  if (children.size() != dependencies.size())
    return invalid("compiled children do not cover the exact direct child set");
  std::map<std::string, std::size_t> required;
  ParentComposition result;
  for (const auto &dependency : dependencies) {
    const auto &definition = graph.modules[dependency.targetModule];
    required.emplace(definition.emittedName, dependency.targetModule);
    result.multiplicities.emplace(definition.emittedName,
                                  dependency.multiplicity);
  }
  // Payload units are immutable. Shared descendants are deduplicated only by
  // exact blob identity; the existing HDL admission rejects conflicting
  // definitions across all remaining units, before any vendor invocation.
  std::map<std::string, ImplementationPayload> payloads;
  std::optional<RepresentationLocator> childTop;
  for (const auto &reference : children) {
    auto child =
        importDesignCompilerBlockGateNetlist(reference, artifacts, blobs);
    if (!child)
      return child.takeError();
    const auto &netlist = child->netlist();
    if (netlist.implementationPlatform != target.reference() ||
        netlist.corner != config.technologyCorner() ||
        netlist.standardCellLibrary != config.standardCellLiberty())
      return invalid("compiled child has another platform, corner or library");
    auto childSource = importRtlBlockSource(netlist.source, artifacts, blobs);
    if (!childSource)
      return childSource.takeError();
    auto expected = required.find(childSource->top().str());
    if (expected == required.end())
      return invalid("compiled child is extra or repeats another direct child");
    if (llvm::Error error = verifyRtlBlockSourceSubgraphDerivation(
            source, expected->second, *childSource))
      return std::move(error);
    required.erase(expected);
    if (!childTop)
      childTop = netlist.representation.top;
    for (const auto &payload : netlist.representation.payloads) {
      if (payload.role != PayloadRole::Netlist)
        continue;
      const std::string digest = formatBlobDigestHex(payload.blobDigest);
      const std::string logicalName = "netlist/" + digest + ".v";
      if (!payloads
               .emplace(digest,
                        ImplementationPayload{PayloadRole::Netlist, logicalName,
                                              payload.blobDigest})
               .second)
        continue;
      auto bytes = blobs.get(payload.blobDigest);
      if (!bytes)
        return bytes.takeError();
      const std::string path = "inputs/children/" + digest + ".v";
      result.mapped.netlistPaths.push_back(path);
      files.push_back({path, textOf(*bytes), reference, false});
    }
  }
  for (const auto &entry : payloads)
    result.payloads.push_back(entry.second);
  auto format = RepresentationFormatDescriptorRef::get(
      RepresentationFormatKind::StructuralVerilogGateNetlist);
  if (!format)
    return format.takeError();
  auto index = indexRepresentation(*format, *childTop, result.payloads, blobs);
  if (!index)
    return index.takeError();
  for (const auto &definition : index->concreteModuleDefinitions()) {
    if (definition.canonicalName == source.top())
      return invalid("compiled child definition shadows the parent root");
    result.mapped.definitionNames.push_back(definition.canonicalName);
  }
  llvm::sort(result.mapped.netlistPaths);
  llvm::sort(result.mapped.definitionNames);
  return result;
}

struct InvocationFacts final {
  FinalizedRtlBlockSource source;
  platform::FinalizedImplementationPlatform target;
  ResolvedDesignCompilerGateNetlistConfigView config;
  ExternalToolSemanticContract contract;
  std::vector<MaterializedBundleFile> files;
  std::vector<std::string> constraints;
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
  auto config = adoptResolvedDesignCompilerGateNetlistConfigView(
      resolvedDesignCompilerGateNetlistConfigSchemaDescriptorBytes(),
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
  std::vector<std::string> constraints;
  const std::string constraint = source->generationConstraint();
  if (!constraint.empty()) {
    files.push_back(
        {constraintPath.str(), constraint, source->reference(), false});
    constraints.push_back(constraintPath.str());
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
  auto semantic = deriveExternalToolSemanticContract(inputs, binding);
  if (!semantic)
    return semantic.takeError();
  return InvocationFacts{std::move(*source), std::move(*target),
                         std::move(*config), std::move(*semantic),
                         std::move(files),   std::move(constraints),
                         std::move(parent)};
}

ExternalToolInvocationImportExpectation
expectation(const InvocationFacts &facts) {
  ExternalToolInvocationImportExpectation result;
  result.semanticContract = facts.contract;
  for (const auto &file : facts.files)
    result.semanticInputs.push_back(
        {file.relativePath, *file.sourceArtifact,
         computeBlobDigest(bytesOf(file.contents))});
  result.externalInputs.push_back({asicStandardCellLibertyInputSlot.str(),
                                   facts.config.standardCellLiberty()});
  result.declaredOutputs.push_back(designCompilerGateNetlistOutputPath.str());
  return result;
}

llvm::Expected<PreparedExternalToolInvocation>
prepare(llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
        const ResolvedCandidateGeneratorBinding &binding,
        const ArtifactStore &artifacts, const BlobStore &blobs,
        const ExternalToolPreparationContext &context) {
  auto invocation = facts(inputs, binding, artifacts, blobs);
  if (!invocation)
    return invocation.takeError();
  auto frozen = resolveDesignCompilerInvocation(invocation->config, context);
  if (!frozen)
    return frozen.takeError();
  auto driver = invocation->parent
                    ? renderDesignCompilerParentDriver(
                          invocation->source.top(), rtlPath,
                          invocation->parent->mapped, invocation->constraints,
                          frozen->externalFiles.front().absolutePath)
                    : renderDesignCompilerDriver(
                          invocation->source.top(), {rtlPath.str()},
                          invocation->constraints,
                          frozen->externalFiles.front().absolutePath,
                          DesignCompilerHierarchy::PreserveDefinitions);
  if (!driver)
    return driver.takeError();
  invocation->files.push_back(
      {"drivers/design-compiler.tcl", std::move(*driver), std::nullopt, false});
  const std::string executable = frozen->tool.executable;
  ExternalToolInvocationBundleSpec specification{
      invocation->contract,
      std::move(frozen->tool),
      frozen->toolVersionProbe,
      std::move(frozen->runtime),
      frozen->containerVersionProbe,
      {{executable, "-f", "drivers/design-compiler.tcl"}},
      std::move(frozen->inheritEnvironment),
      {designCompilerGateNetlistOutputPath.str()},
      std::move(invocation->files),
      std::move(frozen->externalFiles),
      {}};
  return finalizeExternalToolInvocationBundle(context.bundleDestination,
                                              specification);
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
  auto contract = renderSynopsysStandardCellBlackBoxContract(
      facts.config.standardCellLiberty(),
      index->unresolvedExternalDefinitions());
  if (!contract)
    return contract.takeError();
  auto contractDigest = blobs.put(bytesOf(*contract));
  if (!contractDigest)
    return contractDigest.takeError();
  payloads.push_back({PayloadRole::BlackBoxContract, contractPayloadPath.str(),
                      *contractDigest});
  auto root = createImplementationRepresentationRoot(
      RepresentationRootVariant::GateNetlist, std::nullopt, *format,
      std::move(top), std::move(payloads));
  if (!root)
    return root.takeError();
  auto contracts = makeSynopsysStandardCellContractCatalog();
  if (!contracts)
    return contracts.takeError();
  auto result = finalizeBlockGateNetlist(
      {facts.source.reference(), facts.target.reference(),
       facts.config.technologyCorner(),
       synopsysDesignCompilerStandardCellContractRef.str(),
       facts.config.standardCellLiberty(), std::move(*root)},
      *contracts, artifacts, blobs);
  if (!result)
    return result.takeError();
  auto imported = importDesignCompilerBlockGateNetlist(result->reference(),
                                                       artifacts, blobs);
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
  auto expected = expectation(*invocation);
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
  if (llvm::Error error = validateSynopsysOutputInventory(
          designCompilerDescriptor(), prepared.bundleRoot))
    return std::move(error);
  auto imported =
      std::get<ImportedExternalToolInvocationBundle>(std::move(*attempt));
  auto output = readExternalToolInvocationDeclaredOutput(
      imported, designCompilerGateNetlistOutputPath);
  if (!output)
    return output.takeError();
  auto result = publish(*invocation, *output, artifacts, blobs);
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
designCompilerBlockGateNetlistCandidateGeneratorDescriptor() {
  return descriptor;
}

llvm::Error registerDesignCompilerBlockGateNetlistCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindDesignCompilerBlockGateNetlistInputs(
    const ArtifactRootReference &blockSource,
    const ArtifactRootReference &implementationPlatform) {
  if (llvm::Error error =
          registerDesignCompilerBlockGateNetlistCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> inputs{
      {blockInput, {blockSource}}, {targetInput, {implementationPlatform}}};
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), inputs))
    return std::move(error);
  return inputs;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveDesignCompilerBlockGateNetlistBinding(
    const ResolvedDesignCompilerGateNetlistConfigView &config) {
  if (llvm::Error error =
          registerDesignCompilerBlockGateNetlistCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

const dse::CandidateGeneratorDescriptor &
designCompilerHierarchicalBlockGateNetlistCandidateGeneratorDescriptor() {
  return parentDescriptor;
}

llvm::Error
registerDesignCompilerHierarchicalBlockGateNetlistCandidateGenerator() {
  if (llvm::Error error =
          registerCandidateGeneratorDescriptor(parentDescriptor))
    return error;
  return registerCandidateGeneratorProvider(parentProvider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindDesignCompilerHierarchicalBlockGateNetlistInputs(
    const ArtifactRootReference &blockSource,
    const ArtifactRootReference &implementationPlatform,
    llvm::ArrayRef<ArtifactRootReference> compiledChildren) {
  if (llvm::Error error =
          registerDesignCompilerHierarchicalBlockGateNetlistCandidateGenerator())
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
resolveDesignCompilerHierarchicalBlockGateNetlistBinding(
    const ResolvedDesignCompilerGateNetlistConfigView &config) {
  if (llvm::Error error =
          registerDesignCompilerHierarchicalBlockGateNetlistCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(parentDescriptor.reference(),
                                                config.canonicalViewBytes(),
                                                config.digest());
}

llvm::Expected<FinalizedBlockGateNetlist>
importDesignCompilerBlockGateNetlist(const ArtifactRootReference &reference,
                                     const ArtifactStore &artifacts,
                                     const BlobStore &blobs) {
  auto contracts = makeSynopsysStandardCellContractCatalog();
  if (!contracts)
    return contracts.takeError();
  auto result = importBlockGateNetlist(reference, *contracts, artifacts, blobs);
  if (!result)
    return result.takeError();
  const auto &netlist = result->netlist();
  if (netlist.standardCellContract !=
      synopsysDesignCompilerStandardCellContractRef)
    return invalid("mapped library belongs to another provider contract");
  auto index = indexRepresentationRoot(netlist.representation, blobs);
  if (!index)
    return index.takeError();
  auto expected = renderSynopsysStandardCellBlackBoxContract(
      netlist.standardCellLibrary, index->unresolvedExternalDefinitions());
  if (!expected)
    return expected.takeError();
  for (const auto &payload : netlist.representation.payloads)
    if (payload.role == PayloadRole::BlackBoxContract) {
      auto bytes = blobs.get(payload.blobDigest);
      if (!bytes)
        return bytes.takeError();
      if (textOf(*bytes) != *expected)
        return invalid(
            "mapped library contract differs from the exact netlist closure");
    }
  return std::move(*result);
}

} // namespace loom::eda::synopsys
