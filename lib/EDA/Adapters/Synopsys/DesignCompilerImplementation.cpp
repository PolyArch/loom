#include "EDA/Adapters/Synopsys/DesignCompilerBlock.h"

#include "EDA/Adapters/AsicStandardCellContracts.h"
#include "Hardware/Implementation/RepresentationIndex.h"

#include <array>

namespace loom::eda::synopsys {
namespace {

using namespace dse;
using namespace hardware;
using namespace hardware::rtl;

constexpr llvm::StringLiteral configSchema =
    "loom.eda.synopsys.design_compiler.portable_gate_implementation.config.1";
constexpr CandidateGeneratorInputSlotRef implementationInput(0);
constexpr CandidateGeneratorInputSlotRef blockInput(1);
constexpr CandidateGeneratorOutputSlotRef implementationOutput(0);
constexpr CandidateGeneratorWorkUnitRef publicationWork(0);
constexpr std::array<CandidateGeneratorInputSlotDescriptor, 2> inputs{
    {{implementationInput, "portable_rtl_implementation",
      PlanValueRole::CandidateSet, &hardwareImplementationSchema,
      PlanValueCardinality::ExactlyOne},
     {blockInput, "complete_root_block_netlist", PlanValueRole::CandidateSet,
      &blockGateNetlistSchema, PlanValueCardinality::ExactlyOne}}};
constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputs{
    {{implementationOutput, "portable_gate_implementation",
      PlanValueRole::CandidateSet, &hardwareImplementationSchema,
      PlanValueCardinality::ExactlyOne}}};
constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 1> work{
    {{publicationWork, "whole_root_association"}}};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "design_compiler_implementation_invalid: " + message);
}

llvm::ArrayRef<std::uint8_t> schemaBytes() {
  return {reinterpret_cast<const std::uint8_t *>(configSchema.data()),
          configSchema.size()};
}

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  if (!bytes.empty())
    return invalid("whole-root association has no authored configuration");
  return validateComponentViewDigest(schemaBytes(), bytes, digest);
}

const CandidateGeneratorDescriptor descriptor{
    designCompilerPortableGateImplementationCandidateGeneratorKind,
    "synopsys.design_compiler.portable_gate_implementation",
    "loom.eda.synopsys.design_compiler.portable_gate_implementation.generator.v1",
    inputs,
    outputs,
    ResolvedDseConfigViewContract{schemaBytes(), validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    work,
    nullptr,
    ProviderForm::InProcess};

llvm::Expected<CandidateGeneratorProviderResult>
invoke(llvm::ArrayRef<CandidateGeneratorInputBinding> input,
       const ResolvedCandidateGeneratorBinding &binding,
       const ArtifactStore &artifacts, const BlobStore &blobs,
       const CandidateGeneratorInvocationView &invocation) {
  if (binding.descriptorRef() != descriptor.reference())
    return invalid("binding names another generator");
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), input))
    return std::move(error);
  if (llvm::Error error = validateConfig(binding.canonicalConfigBytes(),
                                         binding.configDigest()))
    return std::move(error);
  if (invocation.stopRequested())
    return CandidateGeneratorProviderResult{
        IncompleteCandidateGeneratorResult{
            CandidateGeneratorIncompleteReason::CancelledOrTimeout,
            {{implementationOutput, {}}},
            {}},
        {{publicationWork, 1, 0}}};

  auto implementation = importHardwareImplementation(
      input[implementationInput.ordinal()].artifacts.front(), artifacts, blobs);
  if (!implementation)
    return implementation.takeError();
  auto block = importDesignCompilerBlockGateNetlist(
      input[blockInput.ordinal()].artifacts.front(), artifacts, blobs);
  if (!block)
    return block.takeError();
  const auto &original = implementation->implementation();
  const auto &mapped = block->netlist();
  if (original.implementationPlatform() &&
      *original.implementationPlatform() != mapped.implementationPlatform)
    return invalid("portable RTL and mapped block select different platforms");
  auto abi = importConfigurationABI(original.configurationAbi(), artifacts);
  if (!abi)
    return abi.takeError();
  auto source = importRtlBlockSource(mapped.source, artifacts, blobs);
  if (!source)
    return source.takeError();
  if (llvm::Error error = verifyPortableRtlBlockSourceRootDerivation(
          *source, *abi, *implementation, blobs))
    return std::move(error);
  // Exact portable replay above owns the admitted empty activity, memory-macro
  // and external-binding catalogs. Specialized RTL requires its own association
  // owner; no synthesis correspondence is inferred for such bindings here.
  auto before = indexRepresentationRoot(original.representationRoot(), blobs);
  if (!before)
    return before.takeError();
  auto after = indexRepresentationRoot(mapped.representation, blobs);
  if (!after)
    return after.takeError();
  std::vector<ImplementationInterface> interfaces = original.interfaces().vec();
  const std::string prefix =
      original.representationRoot().top.canonicalName + ".";
  for (auto &interface : interfaces) {
    auto sourceFacts = before->lookup(interface.representationLocator);
    if (!sourceFacts)
      return sourceFacts.takeError();
    llvm::StringRef port = interface.representationLocator.canonicalName;
    if (interface.representationLocator.kind != RepresentationObjectKind::Port ||
        !port.consume_front(prefix) || port.empty() || port.contains('.'))
      return invalid("portable interface is not a direct root port");
    interface.representationLocator.canonicalName =
        mapped.representation.top.canonicalName + "." + port.str();
    auto mappedFacts = after->lookup(interface.representationLocator);
    if (!mappedFacts)
      return mappedFacts.takeError();
    if (!*sourceFacts || !*mappedFacts || !(**sourceFacts == **mappedFacts))
      return invalid("mapped root changed a public interface geometry");
  }
  std::optional<ImplementationPayloadKey> blackBox;
  for (const auto &payload : mapped.representation.payloads)
    if (payload.role == PayloadRole::BlackBoxContract)
      blackBox = {payload.role, payload.canonicalLogicalName};
  auto contracts = makeSynopsysStandardCellContractCatalog();
  if (!contracts)
    return contracts.takeError();
  HardwareImplementationDraft draft{
      original.fabric(),
      original.subject(),
      original.configurationAbi(),
      mapped.representation,
      mapped.implementationPlatform,
      std::move(interfaces),
      {},
      {},
      {{mapped.standardCellContract,
        {{asicStandardCellLibertyInputSlot.str(),
          ExplicitFileDependency{mapped.standardCellLibrary}}},
        {},
        after->unresolvedExternalDefinitions().vec(),
        std::move(blackBox)}}};
  auto published = finalizeHardwareImplementation(std::move(draft), *contracts,
                                                  artifacts, blobs);
  if (!published)
    return published.takeError();
  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{
          {{implementationOutput, {published->reference()}}},
          {{CandidateGeneratorLineageEdgeKind::MechanicalDerivation,
            implementationOutput,
            published->reference(),
            {},
            {}}}},
      {{publicationWork, 1, 1}}};
}

const CandidateGeneratorProvider provider{
    descriptor.reference(), CandidateGeneratorInProcessProvider{invoke}};

} // namespace

const CandidateGeneratorDescriptor &
designCompilerPortableGateImplementationCandidateGeneratorDescriptor() {
  return descriptor;
}

llvm::Error registerDesignCompilerPortableGateImplementationCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindDesignCompilerPortableGateImplementationInputs(
    const ArtifactRootReference &implementation,
    const ArtifactRootReference &blockNetlist) {
  if (llvm::Error error =
          registerDesignCompilerPortableGateImplementationCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> input{
      {implementationInput, {implementation}}, {blockInput, {blockNetlist}}};
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), input))
    return std::move(error);
  return input;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveDesignCompilerPortableGateImplementationBinding() {
  if (llvm::Error error =
          registerDesignCompilerPortableGateImplementationCandidateGenerator())
    return std::move(error);
  auto digest = computeComponentViewDigest(schemaBytes(), {});
  if (!digest)
    return digest.takeError();
  return ResolvedCandidateGeneratorBinding::get(descriptor.reference(), {},
                                                *digest);
}

} // namespace loom::eda::synopsys
