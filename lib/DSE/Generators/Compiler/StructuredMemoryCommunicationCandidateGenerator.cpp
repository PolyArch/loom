#include "DSE/StructuredMemoryCommunicationCandidateGenerator.h"
#include "DSE/StructuredOwnershipInvocationInternal.h"

#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Frontend/Compilation/StructuredMemoryCommunication.h"
#include "Frontend/IR/StructuredProgramArtifact.h"

#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <deque>
#include <limits>
#include <set>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

constexpr llvm::StringLiteral configDescriptor =
    "loom.structured_memory_communication_generator.config.4.0";

enum InputSlot : std::uint32_t {
  StructuredProgramsInput,
  InputSlotCount,
};

constexpr std::array<CandidateGeneratorInputSlotDescriptor, InputSlotCount>
    inputSlots = {{
        {CandidateGeneratorInputSlotRef(StructuredProgramsInput),
         "structured_program", PlanValueRole::CandidateSet,
         &frontend::structuredProgramArtifactSchema,
         PlanValueCardinality::FiniteSet},
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
    "loom.compiler.structured_memory_communication.generator.v4",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{descriptorBytes(), validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
    &lineageContract,
    ProviderForm::InProcess,
};

llvm::Expected<CandidateGeneratorProviderResult>
invokeProvider(llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
               const ResolvedCandidateGeneratorBinding &binding,
               const ArtifactStore &store, const BlobStore &blobs,
               const ExecutionControlView &) {
  auto config = adoptResolvedStructuredMemoryCommunicationGeneratorConfigView(
      descriptorBytes(), binding.canonicalConfigBytes(),
      binding.configDigest());
  if (!config)
    return config.takeError();

  StructuredOwnershipInvocation *invocation =
      detail::StructuredOwnershipInvocationAccess::current();
  struct FrontierEntry final {
    ArtifactRootReference reference;
    bool initial = false;
  };

  std::vector<ArtifactRootReference> orderedInputs =
      inputBindings[StructuredProgramsInput].artifacts;
  llvm::sort(orderedInputs, artifactRootReferenceLess);
  std::vector<ArtifactRootReference> outputs;
  std::set<ArtifactRootReference, decltype(&artifactRootReferenceLess)> seen(
      &artifactRootReferenceLess);
  std::deque<FrontierEntry> frontier;
  for (const ArtifactRootReference &reference : orderedInputs)
    if (seen.insert(reference).second) {
      outputs.push_back(reference);
      frontier.push_back({reference, true});
    }
  std::vector<CandidateGeneratorLineageEdge> lineageEdges;
  std::uint64_t remainingScopes = config->scopeExpansionLimit();
  std::uint64_t inspectedMemoryScopes = 0;
  std::uint64_t decisionAttempts = 0;
  while (remainingScopes != 0 && !frontier.empty()) {
    FrontierEntry entry = std::move(frontier.front());
    frontier.pop_front();
    auto parent = frontend::importStructuredProgram(entry.reference, store);
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
    std::optional<frontend::StructuredEntityRef> trackedSpatialRegion;
    llvm::ArrayRef<frontend::StructuredOperationSourceProvenance>
        sourceProvenance;
    if (invocation) {
      auto tracked =
          detail::StructuredOwnershipInvocationAccess::ownedSpatialRegion(
              *invocation, entry.reference);
      if (!tracked)
        return tracked.takeError();
      trackedSpatialRegion = *tracked;
      auto provenance =
          detail::StructuredOwnershipInvocationAccess::sourceProvenance(
              *invocation, entry.reference);
      if (!provenance)
        return provenance.takeError();
      sourceProvenance = *provenance;
    }
    for (const frontend::StructuredMemoryCommunicationDecision &decision :
         decisions->decisions) {
      const bool channelDecision =
          frontend::structuredMemoryCommunicationDecisionKind(decision) ==
          frontend::StructuredMemoryCommunicationDecisionKind::
              PromoteOrderedBufferToChannel;
      if (!entry.initial && !channelDecision)
        continue;
      if (decisionAttempts == std::numeric_limits<std::uint64_t>::max())
        return invalid("memory-decision accounting overflows u64");
      ++decisionAttempts;
      auto child = frontend::materializeStructuredMemoryCommunicationDecision(
          *parent, decision, trackedSpatialRegion, sourceProvenance);
      if (!child)
        return child.takeError();
      auto published =
          frontend::publishStructuredProgram(child->structuredProgram, store);
      if (!published)
        return published.takeError();
      if (invocation)
        if (llvm::Error error = detail::StructuredOwnershipInvocationAccess::
                recordMemoryCommunicationCandidate(*invocation, entry.reference,
                                                   *published, decision,
                                                   std::move(*child), store))
          return std::move(error);
      auto ownerPayload =
          frontend::encodeStructuredMemoryCommunicationDecision(decision);
      if (!ownerPayload)
        return ownerPayload.takeError();
      lineageEdges.push_back(CandidateGeneratorLineageEdge{
          CandidateGeneratorLineageEdgeKind::CandidateDecision,
          CandidateGeneratorOutputSlotRef(0),
          *published,
          {entry.reference},
          std::move(*ownerPayload)});
      if (seen.insert(*published).second) {
        outputs.push_back(*published);
        if (channelDecision)
          frontier.push_back({*published, false});
      }
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
    llvm::ArrayRef<ArtifactRootReference> structuredPrograms) {
  if (llvm::Error error =
          registerStructuredMemoryCommunicationCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings = {
      {CandidateGeneratorInputSlotRef(StructuredProgramsInput),
       structuredPrograms.vec()},
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
