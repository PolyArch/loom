#include "EDA/Adapters/OpenSource/YosysBlock.h"

#include "EDA/Adapters/AsicStandardCellContracts.h"
#include "EDA/Adapters/PortableGateImplementation.h"

#include <array>

namespace loom::eda::open_source {
namespace {

using namespace dse;
using namespace hardware;
using namespace hardware::rtl;

constexpr llvm::StringLiteral configSchema =
    "loom.eda.open_source.yosys.portable_gate_implementation.config.1";
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
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "yosys_implementation_invalid: " + message);
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
    yosysPortableGateImplementationCandidateGeneratorKind,
    "open_source.yosys.portable_gate_implementation",
    "loom.eda.open_source.yosys.portable_gate_implementation.generator.v1",
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
  auto block = importYosysBlockGateNetlist(
      input[blockInput.ordinal()].artifacts.front(), artifacts, blobs);
  if (!block)
    return block.takeError();
  auto contracts = makeYosysStandardCellContractCatalog();
  if (!contracts)
    return contracts.takeError();
  auto published = associatePortableBlockGateNetlist(
      *implementation, *block, *contracts, artifacts, blobs);
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
yosysPortableGateImplementationCandidateGeneratorDescriptor() {
  return descriptor;
}

llvm::Error registerYosysPortableGateImplementationCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindYosysPortableGateImplementationInputs(
    const ArtifactRootReference &implementation,
    const ArtifactRootReference &blockNetlist) {
  if (llvm::Error error =
          registerYosysPortableGateImplementationCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> input{
      {implementationInput, {implementation}}, {blockInput, {blockNetlist}}};
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), input))
    return std::move(error);
  return input;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveYosysPortableGateImplementationBinding() {
  if (llvm::Error error =
          registerYosysPortableGateImplementationCandidateGenerator())
    return std::move(error);
  auto digest = computeComponentViewDigest(schemaBytes(), {});
  if (!digest)
    return digest.takeError();
  return ResolvedCandidateGeneratorBinding::get(descriptor.reference(), {},
                                                *digest);
}

} // namespace loom::eda::open_source
