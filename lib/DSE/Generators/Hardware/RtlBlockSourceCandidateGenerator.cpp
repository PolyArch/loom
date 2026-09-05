#include "DSE/RtlBlockSourceCandidateGenerator.h"

#include "Hardware/RTL/RtlBlockSource.h"

#include <array>
#include <limits>

namespace loom::dse {
namespace {

constexpr llvm::StringLiteral configSchema =
    "loom.rtl_block_source.selection.1";
constexpr CandidateGeneratorInputSlotRef implementationInput(0);
constexpr CandidateGeneratorOutputSlotRef sourceOutput(0);
constexpr CandidateGeneratorWorkUnitRef derivationWork(0);
constexpr std::array<CandidateGeneratorInputSlotDescriptor, 1> inputs{
    {{implementationInput, "portable_rtl_implementation",
      PlanValueRole::CandidateSet, &hardware::hardwareImplementationSchema,
      PlanValueCardinality::ExactlyOne}}};
constexpr std::array<CandidateGeneratorInputSlotDescriptor, 1> subgraphInputs{
    {{implementationInput, "parent_block_source", PlanValueRole::CandidateSet,
      &hardware::rtl::rtlBlockSourceSchema, PlanValueCardinality::ExactlyOne}}};
constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputs{
    {{sourceOutput, "occurrence_free_block_source", PlanValueRole::CandidateSet,
      &hardware::rtl::rtlBlockSourceSchema, PlanValueCardinality::ExactlyOne}}};
constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 1> work{
    {{derivationWork, "exact_block_source_derivation"}}};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "rtl_block_source_generator_invalid: " +
                                     message);
}

llvm::ArrayRef<std::uint8_t> schemaBytes() {
  return {reinterpret_cast<const std::uint8_t *>(configSchema.data()),
          configSchema.size()};
}

llvm::Expected<std::size_t> selection(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != sizeof(std::uint64_t))
    return invalid("selection must contain one definition ordinal");
  std::uint64_t ordinal = 0;
  for (std::uint8_t byte : bytes)
    ordinal = (ordinal << 8) | byte;
  if (ordinal > std::numeric_limits<std::size_t>::max())
    return invalid("selection exceeds host ordinal range");
  return static_cast<std::size_t>(ordinal);
}

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  if (llvm::Error error =
          validateComponentViewDigest(schemaBytes(), bytes, digest))
    return error;
  auto ordinal = selection(bytes);
  return ordinal ? llvm::Error::success() : ordinal.takeError();
}

const CandidateGeneratorDescriptor descriptor{
    rtlBlockSourceCandidateGeneratorKind,
    "rtl.block_source",
    "loom.rtl.block_source.generator.v1",
    inputs,
    outputs,
    ResolvedDseConfigViewContract{schemaBytes(), validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    work,
    nullptr,
    ProviderForm::InProcess};

const CandidateGeneratorDescriptor subgraphDescriptor{
    rtlBlockSourceSubgraphCandidateGeneratorKind,
    "rtl.block_subgraph_source",
    "loom.rtl.block_subgraph_source.generator.v1",
    subgraphInputs,
    outputs,
    ResolvedDseConfigViewContract{schemaBytes(), validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    work,
    nullptr,
    ProviderForm::InProcess};

struct Parent final {
  hardware::FinalizedHardwareImplementation implementation;
  hardware::FinalizedConfigurationABI abi;
  std::size_t definition;
};

llvm::Expected<Parent>
parent(llvm::ArrayRef<CandidateGeneratorInputBinding> input,
       const ResolvedCandidateGeneratorBinding &binding,
       const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (binding.descriptorRef() != descriptor.reference())
    return invalid("binding names another generator");
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), input))
    return std::move(error);
  if (llvm::Error error = validateConfig(binding.canonicalConfigBytes(),
                                         binding.configDigest()))
    return std::move(error);
  auto ordinal = selection(binding.canonicalConfigBytes());
  if (!ordinal)
    return ordinal.takeError();
  auto implementation = hardware::importHardwareImplementation(
      input.front().artifacts.front(), artifacts, blobs);
  if (!implementation)
    return implementation.takeError();
  auto abi = hardware::importConfigurationABI(
      implementation->implementation().configurationAbi(), artifacts);
  if (!abi)
    return abi.takeError();
  return Parent{std::move(*implementation), std::move(*abi), *ordinal};
}

llvm::Expected<hardware::rtl::FinalizedRtlBlockSource>
deriveSource(llvm::ArrayRef<CandidateGeneratorInputBinding> input,
             const ResolvedCandidateGeneratorBinding &binding,
             const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (binding.descriptorRef() == subgraphDescriptor.reference()) {
    if (llvm::Error error = validateCandidateGeneratorInputBindings(
            subgraphDescriptor.reference(), input))
      return std::move(error);
    if (llvm::Error error = validateConfig(binding.canonicalConfigBytes(),
                                           binding.configDigest()))
      return std::move(error);
    auto ordinal = selection(binding.canonicalConfigBytes());
    if (!ordinal)
      return ordinal.takeError();
    auto source = hardware::rtl::importRtlBlockSource(
        input.front().artifacts.front(), artifacts, blobs);
    if (!source)
      return source.takeError();
    return hardware::rtl::finalizeRtlBlockSourceSubgraph(*source, *ordinal,
                                                         artifacts, blobs);
  }
  auto derivedFrom = parent(input, binding, artifacts, blobs);
  if (!derivedFrom)
    return derivedFrom.takeError();
  return hardware::rtl::finalizePortableRtlBlockSource(
      derivedFrom->abi, derivedFrom->implementation, derivedFrom->definition,
      artifacts, blobs);
}

llvm::Expected<CandidateGeneratorProviderResult>
invoke(llvm::ArrayRef<CandidateGeneratorInputBinding> input,
       const ResolvedCandidateGeneratorBinding &binding,
       const ArtifactStore &artifacts, const BlobStore &blobs,
       const CandidateGeneratorInvocationView &invocation) {
  if (invocation.stopRequested())
    return CandidateGeneratorProviderResult{
        IncompleteCandidateGeneratorResult{
            CandidateGeneratorIncompleteReason::CancelledOrTimeout,
            {{sourceOutput, {}}},
            {}},
        {{derivationWork, 1, 0}}};
  auto source = deriveSource(input, binding, artifacts, blobs);
  if (!source)
    return source.takeError();
  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{
          {{sourceOutput, {source->reference()}}},
          {{CandidateGeneratorLineageEdgeKind::MechanicalDerivation,
            sourceOutput,
            source->reference(),
            {},
            {}}}},
      {{derivationWork, 1, 1}}};
}

const CandidateGeneratorProvider provider{
    descriptor.reference(), CandidateGeneratorInProcessProvider{invoke}};
const CandidateGeneratorProvider subgraphProvider{
    subgraphDescriptor.reference(),
    CandidateGeneratorInProcessProvider{invoke}};

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindInput(const CandidateGeneratorDescriptor &generator,
          const ArtifactRootReference &reference) {
  std::vector<CandidateGeneratorInputBinding> input{
      {implementationInput, {reference}}};
  if (llvm::Error error =
          validateCandidateGeneratorInputBindings(generator.reference(), input))
    return std::move(error);
  return input;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveBinding(const CandidateGeneratorDescriptor &generator,
               std::uint64_t definition) {
  std::vector<std::uint8_t> bytes;
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(definition >> shift));
  bytes.push_back(static_cast<std::uint8_t>(definition));
  auto digest = computeComponentViewDigest(schemaBytes(), bytes);
  if (!digest)
    return digest.takeError();
  return ResolvedCandidateGeneratorBinding::get(generator.reference(), bytes,
                                                *digest);
}

} // namespace

const CandidateGeneratorDescriptor &
rtlBlockSourceCandidateGeneratorDescriptor() {
  return descriptor;
}

llvm::Error registerRtlBlockSourceCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindRtlBlockSourceInputs(const ArtifactRootReference &implementation) {
  if (llvm::Error error = registerRtlBlockSourceCandidateGenerator())
    return std::move(error);
  return bindInput(descriptor, implementation);
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveRtlBlockSourceBinding(std::uint64_t definition) {
  if (llvm::Error error = registerRtlBlockSourceCandidateGenerator())
    return std::move(error);
  return resolveBinding(descriptor, definition);
}

const CandidateGeneratorDescriptor &
rtlBlockSourceSubgraphCandidateGeneratorDescriptor() {
  return subgraphDescriptor;
}

llvm::Error registerRtlBlockSourceSubgraphCandidateGenerator() {
  if (llvm::Error error =
          registerCandidateGeneratorDescriptor(subgraphDescriptor))
    return error;
  return registerCandidateGeneratorProvider(subgraphProvider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindRtlBlockSourceSubgraphInputs(const ArtifactRootReference &source) {
  if (llvm::Error error = registerRtlBlockSourceSubgraphCandidateGenerator())
    return std::move(error);
  return bindInput(subgraphDescriptor, source);
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveRtlBlockSourceSubgraphBinding(std::uint64_t definition) {
  if (llvm::Error error = registerRtlBlockSourceSubgraphCandidateGenerator())
    return std::move(error);
  return resolveBinding(subgraphDescriptor, definition);
}

llvm::Error verifyRtlBlockSourceDerivation(
    llvm::ArrayRef<CandidateGeneratorInputBinding> input,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactRootReference &sourceReference,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto derivedFrom = parent(input, binding, artifacts, blobs);
  if (!derivedFrom)
    return derivedFrom.takeError();
  auto source =
      hardware::rtl::importRtlBlockSource(sourceReference, artifacts, blobs);
  if (!source)
    return source.takeError();
  return hardware::rtl::verifyPortableRtlBlockSourceDerivation(
      *source, derivedFrom->abi, derivedFrom->implementation,
      derivedFrom->definition, blobs);
}

} // namespace loom::dse
