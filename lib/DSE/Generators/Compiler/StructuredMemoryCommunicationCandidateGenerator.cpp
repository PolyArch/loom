#include "DSE/StructuredMemoryCommunicationCandidateGenerator.h"
#include "DSE/StructuredOwnershipInvocationInternal.h"

#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Frontend/Compilation/FabricCapabilityIndex.h"
#include "Frontend/Compilation/StructuredMemoryCommunication.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Frontend/Lowering/CanonicalDataflowLowering.h"

#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

constexpr llvm::StringLiteral configDescriptor =
    "loom.structured_memory_communication_generator.config.3.0";

enum InputSlot : std::uint32_t {
  StructuredProgramsInput,
  FabricInput,
  InputSlotCount,
};

constexpr std::array<CandidateGeneratorInputSlotDescriptor, InputSlotCount>
    inputSlots = {{
        {CandidateGeneratorInputSlotRef(StructuredProgramsInput),
         "structured_program", PlanValueRole::CandidateSet,
         &frontend::structuredProgramArtifactSchema,
         PlanValueCardinality::FiniteSet},
        {CandidateGeneratorInputSlotRef(FabricInput), "fabric",
         PlanValueRole::CandidateSet, &fabric::fabricArtifactSchema,
         PlanValueCardinality::ExactlyOne},
    }};

constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputSlots = {{
    {CandidateGeneratorOutputSlotRef(0), "structured_program",
     PlanValueRole::CandidateSet, &frontend::structuredProgramArtifactSchema,
     PlanValueCardinality::FiniteSet},
}};

constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 2> workUnits = {{
    {CandidateGeneratorWorkUnitRef(0), "memory_scope"},
    {CandidateGeneratorWorkUnitRef(1), "memory_communication_decision"},
}};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "structured_memory_communication_generator_invalid: " + message);
}

llvm::ArrayRef<std::uint8_t> descriptorBytes() {
  return {reinterpret_cast<const std::uint8_t *>(configDescriptor.data()),
          configDescriptor.size()};
}

std::vector<std::uint8_t> encodeConfig(std::uint64_t limit) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(8);
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(limit >> shift));
  bytes.push_back(static_cast<std::uint8_t>(limit));
  return bytes;
}

llvm::Expected<std::uint32_t> decodeConfig(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() < 8)
    return invalid("truncated scope expansion limit");
  std::uint64_t limit = 0;
  for (std::uint8_t byte : bytes.take_front(8))
    limit = (limit << 8) | byte;
  if (bytes.size() != 8)
    return invalid("config has trailing bytes");
  if (limit == 0)
    return invalid("scope expansion limit must be positive");
  if (limit > std::numeric_limits<std::uint32_t>::max())
    return invalid("scope expansion limit exceeds uint32");
  return static_cast<std::uint32_t>(limit);
}

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted = adoptResolvedStructuredMemoryCommunicationGeneratorConfigView(
      descriptorBytes(), bytes, digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

llvm::Error
validateDecisionPayload(llvm::ArrayRef<std::uint8_t> bytes,
                        llvm::ArrayRef<ArtifactRootReference> parents,
                        const ArtifactStore &store) {
  auto adopted = frontend::adoptStructuredMemoryCommunicationDecision(bytes);
  if (!adopted)
    return adopted.takeError();
  if (parents.size() != 1 ||
      parents.front().schemaIdentity !=
          frontend::structuredProgramArtifactSchema.identity ||
      parents.front().schemaVersion !=
          frontend::structuredProgramArtifactSchema.version ||
      frontend::structuredMemoryCommunicationDecisionAnchor(*adopted).parent !=
          parents.front().artifact)
    return invalid("memory decision does not belong to its exact parent");
  auto parent = frontend::importStructuredProgram(parents.front(), store);
  if (!parent)
    return parent.takeError();
  auto domain = frontend::enumerateStructuredMemoryCommunicationDecisions(
      *parent, std::numeric_limits<std::uint64_t>::max());
  if (!domain)
    return domain.takeError();
  if (!llvm::is_contained(domain->decisions, *adopted))
    return invalid("memory decision is outside its exact parent domain");
  return llvm::Error::success();
}

const CandidateGeneratorOwnerLineagePayloadContract lineageContract{
    frontend::structuredMemoryCommunicationDecisionSchemaBytes(),
    validateDecisionPayload};

const CandidateGeneratorDescriptor descriptor{
    structuredMemoryCommunicationCandidateGeneratorKind,
    "compiler.structured_memory_communication",
    "loom.compiler.structured_memory_communication.generator.v3",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{descriptorBytes(), validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
    &lineageContract,
    ProviderForm::InProcess,
};

const ArtifactRootReference &
singleInput(llvm::ArrayRef<CandidateGeneratorInputBinding> bindings,
            InputSlot slot) {
  return bindings[slot].artifacts.front();
}

llvm::Expected<CandidateGeneratorProviderResult>
invokeProvider(llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
               const ResolvedCandidateGeneratorBinding &binding,
               const ArtifactStore &store, const BlobStore &blobs) {
  auto config = adoptResolvedStructuredMemoryCommunicationGeneratorConfigView(
      descriptorBytes(), binding.canonicalConfigBytes(),
      binding.configDigest());
  if (!config)
    return config.takeError();

  StructuredOwnershipInvocation *invocation =
      detail::StructuredOwnershipInvocationAccess::current();
  std::optional<fabric::FinalizedFabricRoot> importedFabric;
  const fabric::FinalizedFabricRoot *exactFabric = nullptr;
  if (invocation) {
    exactFabric =
        &detail::StructuredOwnershipInvocationAccess::fabric(*invocation);
    if (singleInput(inputBindings, FabricInput) != exactFabric->reference())
      return invalid("Fabric input differs from the bound invocation");
  } else {
    auto imported = fabric::importEntireFabricRoot(
        singleInput(inputBindings, FabricInput), store);
    if (!imported)
      return imported.takeError();
    importedFabric.emplace(std::move(*imported));
    exactFabric = &*importedFabric;
  }
  frontend::FabricCapabilityIndex capabilities(exactFabric->view());
  const lowering::CanonicalDataflowLoweringOptions loweringOptions =
      invocation ? detail::StructuredOwnershipInvocationAccess::loweringOptions(
                       *invocation)
                 : lowering::CanonicalDataflowLoweringOptions{};

  std::vector<ArtifactRootReference> outputs =
      inputBindings[StructuredProgramsInput].artifacts;
  std::vector<CandidateGeneratorLineageEdge> lineageEdges;
  std::uint64_t remainingScopes = config->scopeExpansionLimit();
  std::uint64_t inspectedMemoryScopes = 0;
  std::uint64_t decisionAttempts = 0;
  for (const ArtifactRootReference &reference :
       inputBindings[StructuredProgramsInput].artifacts) {
    if (remainingScopes == 0)
      break;
    auto parent = frontend::importStructuredProgram(reference, store);
    if (!parent)
      return parent.takeError();
    auto decisions = frontend::enumerateStructuredMemoryCommunicationDecisions(
        *parent, remainingScopes);
    if (!decisions)
      return decisions.takeError();
    if (decisions->inspectedMemoryScopes >
        std::numeric_limits<std::uint64_t>::max() - inspectedMemoryScopes)
      return invalid("memory-scope accounting overflows u64");
    inspectedMemoryScopes += decisions->inspectedMemoryScopes;
    if (decisions->inspectedMemoryScopes > remainingScopes)
      return invalid("memory scope domain exceeded its resolved limit");
    remainingScopes -= decisions->inspectedMemoryScopes;
    if (decisions->decisions.size() >
        std::numeric_limits<std::uint64_t>::max() - decisionAttempts)
      return invalid("memory-decision accounting overflows u64");
    decisionAttempts += decisions->decisions.size();
    outputs.reserve(outputs.size() + decisions->decisions.size());
    for (const frontend::StructuredMemoryCommunicationDecision &decision :
         decisions->decisions) {
      auto child = frontend::materializeStructuredMemoryCommunicationDecision(
          *parent, decision);
      if (!child)
        return child.takeError();
      auto projected =
          lowering::lowerStructuredProgramToCanonicalDataflowWithProjection(
              child->structuredProgram, loweringOptions);
      if (!projected)
        return projected.takeError();
      auto miss = capabilities.firstInadmissibleActor(projected->artifact);
      if (!miss)
        return miss.takeError();
      if (*miss)
        continue;
      auto published =
          frontend::publishStructuredProgram(child->structuredProgram, store);
      if (!published)
        return published.takeError();
      if (invocation)
        if (llvm::Error error = detail::StructuredOwnershipInvocationAccess::
                recordMemoryCommunicationCandidate(
                    *invocation, reference, *published, decision,
                    std::move(*child), std::move(*projected), store))
          return std::move(error);
      auto ownerPayload =
          frontend::encodeStructuredMemoryCommunicationDecision(decision);
      if (!ownerPayload)
        return ownerPayload.takeError();
      lineageEdges.push_back(CandidateGeneratorLineageEdge{
          CandidateGeneratorLineageEdgeKind::CandidateDecision,
          CandidateGeneratorOutputSlotRef(0),
          *published,
          {reference},
          std::move(*ownerPayload)});
      outputs.push_back(std::move(*published));
    }
  }
  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{
          {{CandidateGeneratorOutputSlotRef(0), std::move(outputs)}},
          std::move(lineageEdges)},
      {{CandidateGeneratorWorkUnitRef(0), inspectedMemoryScopes,
        inspectedMemoryScopes},
       {CandidateGeneratorWorkUnitRef(1), decisionAttempts, decisionAttempts}}};
}

const CandidateGeneratorProvider provider{
    descriptor.reference(),
    CandidateGeneratorInProcessProvider{invokeProvider}};

} // namespace

llvm::ArrayRef<std::uint8_t>
resolvedStructuredMemoryCommunicationGeneratorConfigSchemaBytes() {
  return descriptorBytes();
}

llvm::Expected<ResolvedStructuredMemoryCommunicationGeneratorConfigView>
projectResolvedStructuredMemoryCommunicationGeneratorConfigView(
    const ResolvedConfig &config) {
  const std::uint32_t limit =
      config.dse.memoryCommunication.scopeExpansionLimit;
  if (limit == 0)
    return invalid("scope expansion limit must be positive");
  std::vector<std::uint8_t> bytes = encodeConfig(limit);
  auto digest = computeComponentViewDigest(descriptorBytes(), bytes);
  if (!digest)
    return digest.takeError();
  return ResolvedStructuredMemoryCommunicationGeneratorConfigView(
      limit, std::move(bytes), std::move(*digest));
}

llvm::Expected<ResolvedStructuredMemoryCommunicationGeneratorConfigView>
adoptResolvedStructuredMemoryCommunicationGeneratorConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest) {
  if (schemaDescriptorBytes != descriptorBytes())
    return invalid("config descriptor does not match the exact owner");
  if (llvm::Error error = validateComponentViewDigest(
          schemaDescriptorBytes, canonicalViewBytes, digest))
    return std::move(error);
  auto limit = decodeConfig(canonicalViewBytes);
  if (!limit)
    return limit.takeError();
  std::vector<std::uint8_t> reencoded = encodeConfig(*limit);
  if (llvm::ArrayRef<std::uint8_t>(reencoded) != canonicalViewBytes)
    return invalid("decoded config does not re-encode to the source bytes");
  return ResolvedStructuredMemoryCommunicationGeneratorConfigView(
      *limit, std::move(reencoded), digest);
}

const CandidateGeneratorDescriptor &
structuredMemoryCommunicationCandidateGeneratorDescriptor() {
  return descriptor;
}

llvm::Error registerStructuredMemoryCommunicationCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindStructuredMemoryCommunicationCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> structuredPrograms,
    const ArtifactRootReference &fabric) {
  if (llvm::Error error =
          registerStructuredMemoryCommunicationCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings = {
      {CandidateGeneratorInputSlotRef(StructuredProgramsInput),
       structuredPrograms.vec()},
      {CandidateGeneratorInputSlotRef(FabricInput), {fabric}},
  };
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveStructuredMemoryCommunicationCandidateGeneratorBinding(
    const ResolvedStructuredMemoryCommunicationGeneratorConfigView &config) {
  if (llvm::Error error =
          registerStructuredMemoryCommunicationCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

} // namespace loom::dse
