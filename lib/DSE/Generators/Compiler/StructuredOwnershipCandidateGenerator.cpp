#include "DSE/StructuredOwnershipCandidateGenerator.h"

#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/StructuredOwnership.h"
#include "DSE/StructuredOwnershipInvocationInternal.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

constexpr llvm::StringLiteral configDescriptor =
    "loom.structured_ownership_generator.config.1.0";

enum InputSlot : std::uint32_t {
  StructuredProgramInput,
  FabricInput,
  WorkloadInput,
  RuntimeInput,
  InputSlotCount,
};

constexpr std::array<CandidateGeneratorInputSlotDescriptor, InputSlotCount>
    inputSlots = {{
        {CandidateGeneratorInputSlotRef(StructuredProgramInput),
         "structured_program", PlanValueRole::CandidateSet,
         &frontend::structuredProgramArtifactSchema,
         PlanValueCardinality::ExactlyOne},
        {CandidateGeneratorInputSlotRef(FabricInput), "fabric",
         PlanValueRole::CandidateSet, &fabric::fabricArtifactSchema,
         PlanValueCardinality::ExactlyOne},
        {CandidateGeneratorInputSlotRef(WorkloadInput), "workload",
         PlanValueRole::CandidateSet, &sim::simulationWorkloadSchema,
         PlanValueCardinality::ExactlyOne},
        {CandidateGeneratorInputSlotRef(RuntimeInput), "runtime_input",
         PlanValueRole::CandidateSet, &sim::simulationRuntimeInputSchema,
         PlanValueCardinality::ExactlyOne},
    }};

constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 2> outputSlots = {{
    {CandidateGeneratorOutputSlotRef(0), "structured_program",
     PlanValueRole::CandidateSet, &frontend::structuredProgramArtifactSchema,
     PlanValueCardinality::NonEmptySet},
    {CandidateGeneratorOutputSlotRef(1), "accelerator_candidate",
     PlanValueRole::CandidateSet, &frontend::structuredProgramArtifactSchema,
     PlanValueCardinality::FiniteSet},
}};

constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 2> workUnits = {{
    {CandidateGeneratorWorkUnitRef(0), "scope_expansion"},
    {CandidateGeneratorWorkUnitRef(1), "ownership_decision"},
}};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "structured_ownership_generator_invalid: " +
                                     message);
}

llvm::ArrayRef<std::uint8_t> descriptorBytes() {
  return {reinterpret_cast<const std::uint8_t *>(configDescriptor.data()),
          configDescriptor.size()};
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

llvm::Expected<std::uint64_t> readU64(llvm::ArrayRef<std::uint8_t> bytes,
                                      std::size_t &offset) {
  if (bytes.size() - offset < 8)
    return invalid("truncated u64 field");
  std::uint64_t value = 0;
  for (unsigned ordinal = 0; ordinal != 8; ++ordinal)
    value = (value << 8) | bytes[offset++];
  return value;
}

bool structuredRefLess(const frontend::StructuredEntityRef &lhs,
                       const frontend::StructuredEntityRef &rhs) {
  if (lhs.parent.bytes() != rhs.parent.bytes())
    return lhs.parent.bytes() < rhs.parent.bytes();
  if (lhs.kind != rhs.kind)
    return static_cast<std::uint32_t>(lhs.kind) <
           static_cast<std::uint32_t>(rhs.kind);
  return lhs.ordinal < rhs.ordinal;
}

llvm::Expected<std::vector<frontend::StructuredEntityRef>>
canonicalRoots(llvm::ArrayRef<frontend::StructuredEntityRef> roots) {
  std::vector<frontend::StructuredEntityRef> canonical(roots.begin(),
                                                       roots.end());
  for (const frontend::StructuredEntityRef &root : canonical)
    if (root.kind != frontend::StructuredEntityKind::Operation)
      return invalid("protocol root is not an operation reference");
  llvm::sort(canonical, structuredRefLess);
  if (std::adjacent_find(canonical.begin(), canonical.end()) != canonical.end())
    return invalid("protocol roots contain a duplicate reference");
  if (!canonical.empty()) {
    const ArtifactIdentity &parent = canonical.front().parent;
    for (const frontend::StructuredEntityRef &root : canonical)
      if (root.parent != parent)
        return invalid("protocol roots have different Structured owners");
  }
  return canonical;
}

std::vector<std::uint8_t>
encodeConfig(std::uint64_t scopeExpansionLimit,
             llvm::ArrayRef<frontend::StructuredEntityRef> roots) {
  std::vector<std::uint8_t> bytes;
  const std::size_t rootBytes =
      roots.size() * frontend::structuredEntityRefWireSize;
  bytes.reserve(16 + rootBytes);
  appendU64(bytes, scopeExpansionLimit);
  appendU64(bytes, roots.size());
  for (const frontend::StructuredEntityRef &root : roots) {
    std::vector<std::uint8_t> encoded =
        frontend::encodeStructuredEntityRef(root);
    bytes.insert(bytes.end(), encoded.begin(), encoded.end());
  }
  return bytes;
}

struct DecodedConfig final {
  std::uint64_t scopeExpansionLimit;
  std::vector<frontend::StructuredEntityRef> roots;
};

llvm::Expected<DecodedConfig> decodeConfig(llvm::ArrayRef<std::uint8_t> bytes) {
  std::size_t offset = 0;
  auto scopeLimit = readU64(bytes, offset);
  auto rootCount = readU64(bytes, offset);
  if (!scopeLimit)
    return scopeLimit.takeError();
  if (!rootCount)
    return rootCount.takeError();
  if (*scopeLimit == 0)
    return invalid("scope expansion limit must be positive");
  if (*rootCount >
      (bytes.size() - offset) / frontend::structuredEntityRefWireSize)
    return invalid("protocol root count exceeds remaining bytes");
  if (*rootCount > std::numeric_limits<std::size_t>::max())
    return invalid("protocol root count is not host-representable");

  std::vector<frontend::StructuredEntityRef> roots;
  roots.reserve(static_cast<std::size_t>(*rootCount));
  for (std::uint64_t ordinal = 0; ordinal != *rootCount; ++ordinal) {
    auto root = frontend::decodeStructuredEntityRef(
        bytes.slice(offset, frontend::structuredEntityRefWireSize));
    if (!root)
      return root.takeError();
    roots.push_back(std::move(*root));
    offset += frontend::structuredEntityRefWireSize;
  }
  if (offset != bytes.size())
    return invalid("config has trailing bytes");
  auto canonical = canonicalRoots(roots);
  if (!canonical)
    return canonical.takeError();
  if (*canonical != roots)
    return invalid("protocol roots are not in canonical order");
  return DecodedConfig{*scopeLimit, std::move(roots)};
}

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted = adoptResolvedStructuredOwnershipGeneratorConfigView(
      descriptorBytes(), bytes, digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

llvm::Error
validateDecisionPayload(llvm::ArrayRef<std::uint8_t> bytes,
                        llvm::ArrayRef<ArtifactRootReference> parents,
                        const ArtifactStore &store) {
  auto adopted = frontend::adoptSpatialOwnershipDecision(bytes);
  if (!adopted)
    return adopted.takeError();
  if (parents.size() != 1 ||
      parents.front().schemaIdentity !=
          frontend::structuredProgramArtifactSchema.identity ||
      parents.front().schemaVersion !=
          frontend::structuredProgramArtifactSchema.version ||
      adopted->scope.selection.parent != parents.front().artifact)
    return invalid("ownership decision does not belong to its exact parent");
  auto parent = frontend::importStructuredProgram(parents.front(), store);
  if (!parent)
    return parent.takeError();
  auto domain = frontend::enumerateSpatialOwnershipDecisionDomain(
      *parent, adopted->scope.selection);
  if (!domain)
    return domain.takeError();
  if (!llvm::is_contained(*domain, adopted->point))
    return invalid("ownership decision is outside its exact parent domain");
  return llvm::Error::success();
}

const CandidateGeneratorOwnerLineagePayloadContract lineageContract{
    frontend::spatialOwnershipDecisionSchemaBytes(), validateDecisionPayload};

const CandidateGeneratorDescriptor descriptor{
    structuredOwnershipCandidateGeneratorKind,
    "compiler.structured_ownership",
    "loom.compiler.structured_ownership.generator.v1",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{descriptorBytes(), validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
    &lineageContract,
    ProviderForm::InProcess,
};

const ArtifactRootReference &
singleInput(llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
            InputSlot slot) {
  return inputBindings[slot].artifacts.front();
}

llvm::Expected<CandidateGeneratorProviderResult> invokeOwnershipProvider(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store, const BlobStore &blobs) {
  auto config = adoptResolvedStructuredOwnershipGeneratorConfigView(
      descriptorBytes(), binding.canonicalConfigBytes(),
      binding.configDigest());
  if (!config)
    return config.takeError();

  auto simulationInputs = sim::importStructuredProgramSimulationInputs(
      singleInput(inputBindings, WorkloadInput),
      singleInput(inputBindings, RuntimeInput), store);
  if (!simulationInputs)
    return simulationInputs.takeError();
  const ArtifactRootReference &structured =
      singleInput(inputBindings, StructuredProgramInput);
  if (structured.artifact != simulationInputs->structuredProgram.identity())
    return invalid("workload owner differs from the Structured input");

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
  for (const frontend::StructuredEntityRef &root :
       config->protocolCallableRoots())
    if (root.parent != simulationInputs->structuredProgram.identity())
      return invalid("protocol root belongs to a foreign Structured owner");

  StructuredOwnershipGenerationOptions options;
  options.scopeExpansionLimit = config->scopeExpansionLimit();
  options.candidateWorkerCount = defaultCandidateWorkerCount();
  options.protocolCallableRoots.assign(config->protocolCallableRoots().begin(),
                                       config->protocolCallableRoots().end());
  auto generated = generateStructuredOwnershipCandidates(
      simulationInputs->structuredProgram, simulationInputs->workload,
      simulationInputs->runtimeInput, *exactFabric, options, store);
  if (!generated)
    return generated.takeError();
  std::vector<ArtifactRootReference> allCandidates(
      generated->candidates.candidates().begin(),
      generated->candidates.candidates().end());
  std::vector<ArtifactRootReference> acceleratorCandidates;
  acceleratorCandidates.reserve(allCandidates.size());
  for (const ArtifactRootReference &candidate : allCandidates)
    if (candidate != structured)
      acceleratorCandidates.push_back(candidate);
  if (acceleratorCandidates.size() + 1 != allCandidates.size())
    return invalid("generated candidate set lost its exact source input");
  std::vector<CandidateGeneratorLineageEdge> lineageEdges;
  for (const StructuredOwnershipCandidateDisposition &disposition :
       generated->dispositions) {
    const auto *child = std::get_if<ArtifactRootReference>(&disposition.result);
    if (!child || *child == structured)
      continue;
    if (!disposition.coordinate.decision)
      return invalid("generated ownership child has no typed decision");
    if (!std::binary_search(acceleratorCandidates.begin(),
                            acceleratorCandidates.end(), *child,
                            artifactRootReferenceLess))
      return invalid("ownership lineage target is absent from candidate set");
    auto payload = frontend::encodeSpatialOwnershipDecision(
        frontend::SpatialOwnershipDecision{disposition.coordinate.scope,
                                           *disposition.coordinate.decision});
    if (!payload)
      return payload.takeError();
    lineageEdges.push_back(CandidateGeneratorLineageEdge{
        CandidateGeneratorLineageEdgeKind::CandidateDecision,
        CandidateGeneratorOutputSlotRef(0),
        *child,
        {structured},
        *payload});
    lineageEdges.push_back(CandidateGeneratorLineageEdge{
        CandidateGeneratorLineageEdgeKind::CandidateDecision,
        CandidateGeneratorOutputSlotRef(1),
        *child,
        {structured},
        std::move(*payload)});
  }
  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{
          {{CandidateGeneratorOutputSlotRef(0), std::move(allCandidates)},
           {CandidateGeneratorOutputSlotRef(1),
            std::move(acceleratorCandidates)}},
          std::move(lineageEdges)},
      {{CandidateGeneratorWorkUnitRef(0), generated->plannedScopeCount,
        generated->plannedScopeCount},
       {CandidateGeneratorWorkUnitRef(1), generated->decisionAttemptCount,
        generated->decisionAttemptCount}}};
}

const CandidateGeneratorProvider provider{
    descriptor.reference(),
    CandidateGeneratorInProcessProvider{invokeOwnershipProvider}};

} // namespace

llvm::ArrayRef<std::uint8_t>
resolvedStructuredOwnershipGeneratorConfigSchemaBytes() {
  return descriptorBytes();
}

llvm::Expected<ResolvedStructuredOwnershipGeneratorConfigView>
projectResolvedStructuredOwnershipGeneratorConfigView(
    const ResolvedConfig &config,
    llvm::ArrayRef<frontend::StructuredEntityRef> protocolCallableRoots) {
  if (config.dse.structuredOwnership.scopeExpansionLimit == 0)
    return invalid("scope expansion limit must be positive");
  auto roots = canonicalRoots(protocolCallableRoots);
  if (!roots)
    return roots.takeError();
  std::vector<std::uint8_t> bytes =
      encodeConfig(config.dse.structuredOwnership.scopeExpansionLimit, *roots);
  auto digest = computeComponentViewDigest(descriptorBytes(), bytes);
  if (!digest)
    return digest.takeError();
  return ResolvedStructuredOwnershipGeneratorConfigView(
      config.dse.structuredOwnership.scopeExpansionLimit, std::move(*roots),
      std::move(bytes), std::move(*digest));
}

llvm::Expected<ResolvedStructuredOwnershipGeneratorConfigView>
adoptResolvedStructuredOwnershipGeneratorConfigView(
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
      encodeConfig(decoded->scopeExpansionLimit, decoded->roots);
  if (llvm::ArrayRef<std::uint8_t>(reencoded) != canonicalViewBytes)
    return invalid("decoded config does not re-encode to the source bytes");
  return ResolvedStructuredOwnershipGeneratorConfigView(
      decoded->scopeExpansionLimit, std::move(decoded->roots),
      std::move(reencoded), digest);
}

const CandidateGeneratorDescriptor &
structuredOwnershipCandidateGeneratorDescriptor() {
  return descriptor;
}

llvm::Error registerStructuredOwnershipCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindStructuredOwnershipCandidateGeneratorInputs(
    const ArtifactRootReference &structuredProgram,
    const ArtifactRootReference &fabric, const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput) {
  if (llvm::Error error = registerStructuredOwnershipCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings = {
      {CandidateGeneratorInputSlotRef(StructuredProgramInput),
       {structuredProgram}},
      {CandidateGeneratorInputSlotRef(FabricInput), {fabric}},
      {CandidateGeneratorInputSlotRef(WorkloadInput), {workload}},
      {CandidateGeneratorInputSlotRef(RuntimeInput), {runtimeInput}},
  };
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveStructuredOwnershipCandidateGeneratorBinding(
    const ResolvedStructuredOwnershipGeneratorConfigView &config) {
  if (llvm::Error error = registerStructuredOwnershipCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

} // namespace loom::dse
