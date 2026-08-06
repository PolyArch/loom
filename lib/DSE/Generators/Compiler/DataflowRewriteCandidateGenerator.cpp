#include "DSE/DataflowRewriteCandidateGenerator.h"
#include "DSE/StructuredOwnershipInvocationInternal.h"

#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/Transforms/DataflowRewrite.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Frontend/Compilation/FabricCapabilityIndex.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

constexpr llvm::StringLiteral configDescriptor =
    "loom.dataflow_rewrite_generator.config.1.1";

enum InputSlot : std::uint32_t {
  CanonicalDataflowProgramsInput,
  FabricInput,
  InputSlotCount,
};

constexpr std::array<CandidateGeneratorInputSlotDescriptor, InputSlotCount>
    inputSlots = {{
        {CandidateGeneratorInputSlotRef(CanonicalDataflowProgramsInput),
         "canonical_dataflow", PlanValueRole::CandidateSet,
         &dataflow::canonicalDataflowSchema, PlanValueCardinality::FiniteSet},
        {CandidateGeneratorInputSlotRef(FabricInput), "fabric",
         PlanValueRole::CandidateSet, &fabric::fabricArtifactSchema,
         PlanValueCardinality::ExactlyOne},
    }};

constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputSlots = {
    {{CandidateGeneratorOutputSlotRef(0), "canonical_dataflow",
      PlanValueRole::CandidateSet, &dataflow::canonicalDataflowSchema,
      PlanValueCardinality::FiniteSet}}};

constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 1> workUnits = {{
    {CandidateGeneratorWorkUnitRef(0), "rewrite_expansion"},
}};

constexpr std::array<dataflow::DataflowRewriteKind, 3> rewriteKinds = {{
    dataflow::DataflowRewriteKind::PackUnpackRoundTripEliminate,
    dataflow::DataflowRewriteKind::ParallelizeSerializeRoundTripEliminate,
    dataflow::DataflowRewriteKind::ActivationPreservingConstantFold,
}};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "dataflow_rewrite_generator_invalid: " +
                                     message);
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

llvm::Expected<std::uint64_t> decodeConfig(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() < 8)
    return invalid("truncated scope expansion limit");
  std::uint64_t limit = 0;
  for (std::uint8_t byte : bytes.take_front(8))
    limit = (limit << 8) | byte;
  if (bytes.size() != 8)
    return invalid("config has trailing bytes");
  if (limit == 0)
    return invalid("scope expansion limit must be positive");
  return limit;
}

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted = adoptResolvedDataflowRewriteGeneratorConfigView(
      descriptorBytes(), bytes, digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

llvm::Error
validateDecisionPayload(llvm::ArrayRef<std::uint8_t> bytes,
                        llvm::ArrayRef<ArtifactRootReference> parents,
                        const ArtifactStore &store) {
  auto adopted = dataflow::adoptDataflowRewriteDecision(bytes);
  if (!adopted)
    return adopted.takeError();
  if (parents.size() != 1 ||
      parents.front().schemaIdentity !=
          dataflow::canonicalDataflowSchema.identity ||
      parents.front().schemaVersion !=
          dataflow::canonicalDataflowSchema.version)
    return invalid(
        "rewrite decision does not have one exact Canonical Dataflow parent");
  if (const auto *chunk =
          std::get_if<dataflow::ElementwiseVectorChunkRewrite>(&*adopted)) {
    if (chunk->actor.artifact != parents.front().artifact)
      return invalid("chunk decision does not belong to its exact parent");
  } else if (const auto *scalar =
                 std::get_if<dataflow::ElementwiseVectorScalarizeRewrite>(
                     &*adopted)) {
    if (scalar->actor.artifact != parents.front().artifact)
      return invalid("scalarization decision does not belong to its exact "
                     "parent");
  }
  auto parent = dataflow::importCanonicalDataflow(parents.front(), store);
  if (!parent)
    return parent.takeError();
  if (!std::holds_alternative<dataflow::DataflowRewriteKind>(*adopted)) {
    auto cost = dataflow::dataflowRewriteExpansionCost(*parent, *adopted);
    if (!cost)
      return cost.takeError();
  }
  return llvm::Error::success();
}

const CandidateGeneratorDescriptor descriptor{
    dataflowRewriteCandidateGeneratorKind,
    "compiler.dataflow_rewrite",
    "loom.compiler.dataflow_rewrite.generator.v2",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{descriptorBytes(), validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
    &dataflowRewriteCandidateLineagePayloadContract(),
};

const ArtifactRootReference &
singleInput(llvm::ArrayRef<CandidateGeneratorInputBinding> bindings,
            InputSlot slot) {
  return bindings[slot].artifacts.front();
}

struct SearchCandidate final {
  ArtifactRootReference reference;
  dataflow::CanonicalDataflowArtifact artifact;
};

llvm::Expected<CandidateGeneratorInvocationOutcome>
invokeProvider(llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
               const ResolvedCandidateGeneratorBinding &binding,
               const ArtifactStore &store) {
  auto config = adoptResolvedDataflowRewriteGeneratorConfigView(
      descriptorBytes(), binding.canonicalConfigBytes(),
      binding.configDigest());
  if (!config)
    return config.takeError();

  const ArtifactRootReference &fabricReference =
      singleInput(inputBindings, FabricInput);
  std::optional<fabric::FinalizedFabricRoot> importedFabric;
  const fabric::FinalizedFabricRoot *fabricRoot = nullptr;
  if (StructuredOwnershipInvocation *invocation =
          detail::StructuredOwnershipInvocationAccess::current()) {
    const fabric::FinalizedFabricRoot &bound =
        detail::StructuredOwnershipInvocationAccess::fabric(*invocation);
    if (bound.reference() != fabricReference)
      return invalid("bound Fabric differs from the active invocation");
    fabricRoot = &bound;
  } else {
    auto imported = fabric::importEntireFabricRoot(fabricReference, store);
    if (!imported)
      return imported.takeError();
    importedFabric.emplace(std::move(*imported));
    fabricRoot = &*importedFabric;
  }
  frontend::FabricCapabilityIndex capabilities(fabricRoot->view());

  std::vector<ArtifactRootReference> outputs;
  std::set<ArtifactRootReference, decltype(&artifactRootReferenceLess)> seen(
      &artifactRootReferenceLess);
  std::vector<CandidateGeneratorLineageEdge> generatedLineageEdges;
  std::deque<SearchCandidate> frontier;
  std::uint64_t expansions = 0;
  bool semanticLimitReached = false;

  const auto classify =
      [&](ArtifactRootReference reference,
          dataflow::CanonicalDataflowArtifact candidate) -> llvm::Error {
    if (!seen.insert(reference).second)
      return llvm::Error::success();
    auto miss = capabilities.firstInadmissibleActor(candidate);
    if (!miss)
      return miss.takeError();
    if (!*miss) {
      outputs.push_back(std::move(reference));
      return llvm::Error::success();
    }
    frontier.push_back(
        SearchCandidate{std::move(reference), std::move(candidate)});
    return llvm::Error::success();
  };

  const auto publishChild =
      [&](const ArtifactRootReference &parent,
          const dataflow::DataflowRewriteDecision &decision,
          dataflow::CanonicalDataflowArtifact child) -> llvm::Error {
    auto published = dataflow::publishCanonicalDataflow(child, store);
    if (!published)
      return published.takeError();
    if (StructuredOwnershipInvocation *invocation =
            detail::StructuredOwnershipInvocationAccess::current())
      if (llvm::Error error = detail::StructuredOwnershipInvocationAccess::
              recordDataflowRewriteCandidate(*invocation, parent, *published,
                                             decision, store))
        return error;
    auto ownerPayload = dataflow::encodeDataflowRewriteDecision(decision);
    if (!ownerPayload)
      return ownerPayload.takeError();
    generatedLineageEdges.push_back(CandidateGeneratorLineageEdge{
        CandidateGeneratorLineageEdgeKind::CandidateDecision,
        CandidateGeneratorOutputSlotRef(0),
        *published,
        {parent},
        std::move(*ownerPayload)});
    return classify(std::move(*published), std::move(child));
  };

  for (const ArtifactRootReference &reference :
       inputBindings[CanonicalDataflowProgramsInput].artifacts) {
    auto parent = dataflow::importCanonicalDataflow(reference, store);
    if (!parent)
      return parent.takeError();

    // Fixed catalog rules are independent one-hop alternatives from the exact
    // input. Their children become seeds for the same vector decomposition
    // search instead of being discarded merely because the first rewrite is
    // not yet Fabric-admissible.
    std::vector<std::pair<dataflow::DataflowRewriteDecision,
                          dataflow::CanonicalDataflowArtifact>>
        fixedChildren;

    for (dataflow::DataflowRewriteKind kind : rewriteKinds) {
      if (expansions == config->scopeExpansionLimit()) {
        semanticLimitReached = true;
        break;
      }
      ++expansions;
      auto child = dataflow::materializeDataflowRewrite(*parent, kind);
      if (!child)
        return child.takeError();
      if (!*child)
        continue;
      fixedChildren.emplace_back(dataflow::DataflowRewriteDecision{kind},
                                 std::move(**child));
    }
    if (semanticLimitReached)
      break;

    if (llvm::Error error = classify(reference, std::move(*parent)))
      return std::move(error);
    for (auto &[decision, child] : fixedChildren)
      if (llvm::Error error =
              publishChild(reference, decision, std::move(child)))
        return std::move(error);
  }

  while (!semanticLimitReached && !frontier.empty()) {
    SearchCandidate parent = std::move(frontier.front());
    frontier.pop_front();
    auto miss = capabilities.firstInadmissibleActor(parent.artifact);
    if (!miss)
      return miss.takeError();
    if (!*miss)
      return invalid("admitted candidate remained in the rewrite frontier");

    auto decisions = dataflow::enumerateElementwiseVectorDecompositionDecisions(
        parent.artifact, (*miss)->actor);
    if (!decisions)
      return decisions.takeError();
    for (const dataflow::DataflowRewriteDecision &decision : *decisions) {
      auto cost =
          dataflow::dataflowRewriteExpansionCost(parent.artifact, decision);
      if (!cost)
        return cost.takeError();
      if (*cost > config->scopeExpansionLimit() - expansions) {
        semanticLimitReached = true;
        break;
      }
      expansions += *cost;
      auto child =
          dataflow::materializeDataflowRewrite(parent.artifact, decision);
      if (!child)
        return child.takeError();
      if (!*child)
        return invalid("typed vector decomposition produced an identity");
      if (llvm::Error error =
              publishChild(parent.reference, decision, std::move(**child)))
        return std::move(error);
    }
  }

  CandidateGeneratorOutputBinding output{CandidateGeneratorOutputSlotRef(0),
                                         std::move(outputs)};
  std::set<ArtifactRootReference, decltype(&artifactRootReferenceLess)>
      requiredTargets(output.artifacts.begin(), output.artifacts.end(),
                      &artifactRootReferenceLess);
  std::vector<CandidateGeneratorLineageEdge> retainedLineageEdges;
  for (auto edge = generatedLineageEdges.rbegin();
       edge != generatedLineageEdges.rend(); ++edge) {
    if (requiredTargets.find(edge->output) == requiredTargets.end())
      continue;
    retainedLineageEdges.push_back(std::move(*edge));
    for (const ArtifactRootReference &parent :
         retainedLineageEdges.back().parents)
      requiredTargets.insert(parent);
  }
  std::reverse(retainedLineageEdges.begin(), retainedLineageEdges.end());
  if (semanticLimitReached)
    return CandidateGeneratorInvocationOutcome{
        IncompleteCandidateGeneratorInvocation{
            CandidateGeneratorIncompleteReason::SemanticLimitReached,
            {std::move(output)},
            std::move(retainedLineageEdges)}};
  return CandidateGeneratorInvocationOutcome{
      CompletedCandidateGeneratorInvocation{{std::move(output)},
                                            std::move(retainedLineageEdges)}};
}

const CandidateGeneratorProvider provider{descriptor.reference(),
                                          invokeProvider};

} // namespace

llvm::ArrayRef<std::uint8_t>
resolvedDataflowRewriteGeneratorConfigSchemaBytes() {
  return descriptorBytes();
}

llvm::Expected<ResolvedDataflowRewriteGeneratorConfigView>
projectResolvedDataflowRewriteGeneratorConfigView(
    const ResolvedConfig &config) {
  const std::uint64_t limit = config.dse.dataflowRewrite.scopeExpansionLimit;
  if (limit == 0)
    return invalid("scope expansion limit must be positive");
  std::vector<std::uint8_t> bytes = encodeConfig(limit);
  auto digest = computeComponentViewDigest(descriptorBytes(), bytes);
  if (!digest)
    return digest.takeError();
  return ResolvedDataflowRewriteGeneratorConfigView(limit, std::move(bytes),
                                                    std::move(*digest));
}

llvm::Expected<ResolvedDataflowRewriteGeneratorConfigView>
adoptResolvedDataflowRewriteGeneratorConfigView(
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
  return ResolvedDataflowRewriteGeneratorConfigView(
      *limit, std::move(reencoded), digest);
}

const CandidateGeneratorDescriptor &
dataflowRewriteCandidateGeneratorDescriptor() {
  return descriptor;
}

const CandidateGeneratorOwnerLineagePayloadContract &
dataflowRewriteCandidateLineagePayloadContract() {
  static const CandidateGeneratorOwnerLineagePayloadContract contract{
      dataflow::dataflowRewriteDecisionSchemaBytes(), validateDecisionPayload};
  return contract;
}

llvm::Error registerDataflowRewriteCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindDataflowRewriteCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> canonicalDataflowPrograms,
    const ArtifactRootReference &fabric) {
  if (llvm::Error error = registerDataflowRewriteCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings = {
      {CandidateGeneratorInputSlotRef(CanonicalDataflowProgramsInput),
       canonicalDataflowPrograms.vec()},
      {CandidateGeneratorInputSlotRef(FabricInput), {fabric}},
  };
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveDataflowRewriteCandidateGeneratorBinding(
    const ResolvedDataflowRewriteGeneratorConfigView &config) {
  if (llvm::Error error = registerDataflowRewriteCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

} // namespace loom::dse
