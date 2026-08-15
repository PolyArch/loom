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
#include <iterator>
#include <limits>
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
  if (limit > std::numeric_limits<std::uint32_t>::max())
    return invalid("scope expansion limit exceeds the uint32 owner domain");
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
  auto parent = dataflow::importCanonicalDataflow(parents.front(), store);
  if (!parent)
    return parent.takeError();
  auto materialized = dataflow::materializeDataflowRewrite(*parent, *adopted);
  if (!materialized)
    return materialized.takeError();
  if (!*materialized)
    return invalid("rewrite lineage payload is an identity decision");
  return llvm::Error::success();
}

const CandidateGeneratorDescriptor descriptor{
    dataflowRewriteCandidateGeneratorKind,
    "compiler.dataflow_rewrite",
    "loom.compiler.dataflow_rewrite.generator.v3",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{descriptorBytes(), validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
    &dataflowRewriteCandidateLineagePayloadContract(),
    ProviderForm::InProcess,
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

struct AttemptedDecisionKey final {
  ArtifactIdentity parent;
  std::vector<std::uint8_t> payload;
};

bool attemptedDecisionLess(const AttemptedDecisionKey &lhs,
                           const AttemptedDecisionKey &rhs) {
  if (lhs.parent != rhs.parent)
    return lhs.parent.bytes() < rhs.parent.bytes();
  return lhs.payload < rhs.payload;
}

llvm::Expected<CandidateGeneratorProviderResult>
invokeProvider(llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
               const ResolvedCandidateGeneratorBinding &binding,
               const ArtifactStore &store, const BlobStore &blobs,
               const ExecutionControlView &) {
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
  // This typed set is bound to dataflowRewriteDecisionSchemaBytes(), so the
  // schema identity and version in the normative attempted key are invariant
  // rather than copied into every entry.
  std::set<AttemptedDecisionKey, decltype(&attemptedDecisionLess)> attempted(
      &attemptedDecisionLess);
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
      outputs.push_back(reference);
    }
    frontier.push_back(
        SearchCandidate{std::move(reference), std::move(candidate)});
    return llvm::Error::success();
  };

  const auto publishChild =
      [&](const ArtifactRootReference &parent,
          const dataflow::DataflowRewriteDecision &decision,
          llvm::ArrayRef<dataflow::StaticGraphLaunchRef> parentLaunches,
          dataflow::MaterializedDataflowRewriteProjection child)
      -> llvm::Error {
    ArtifactRootReference reference{
        dataflow::canonicalDataflowSchema.identity.str(),
        dataflow::canonicalDataflowSchema.version, child.artifact.identity()};
    // The attempt has already consumed its logical work slot. An identity
    // reached through another lineage is neither republished nor re-enqueued.
    if (seen.find(reference) != seen.end())
      return llvm::Error::success();
    auto published = dataflow::publishCanonicalDataflow(child.artifact, store);
    if (!published)
      return published.takeError();
    if (StructuredOwnershipInvocation *invocation =
            detail::StructuredOwnershipInvocationAccess::current())
      if (llvm::Error error = detail::StructuredOwnershipInvocationAccess::
              recordDataflowRewriteCandidate(
                  *invocation, parent, *published, decision, parentLaunches,
                  child.trackedStaticGraphLaunches, store))
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
    return classify(std::move(*published), std::move(child.artifact));
  };

  std::vector<ArtifactRootReference> orderedInputs =
      inputBindings[CanonicalDataflowProgramsInput].artifacts;
  llvm::sort(orderedInputs, artifactRootReferenceLess);
  for (const ArtifactRootReference &reference : orderedInputs) {
    auto parent = dataflow::importCanonicalDataflow(reference, store);
    if (!parent)
      return parent.takeError();
    if (llvm::Error error = classify(reference, std::move(*parent)))
      return std::move(error);
  }

  while (!semanticLimitReached && !frontier.empty()) {
    SearchCandidate parent = std::move(frontier.front());
    frontier.pop_front();
    auto miss = capabilities.firstInadmissibleActor(parent.artifact);
    if (!miss)
      return miss.takeError();
    auto fixed =
        dataflow::enumerateFixedDataflowRewriteDecisions(parent.artifact);
    if (!fixed)
      return fixed.takeError();
    std::vector<dataflow::DataflowRewriteDecision> decisions =
        std::move(*fixed);
    if (*miss) {
      auto vectorDecisions =
          dataflow::enumerateElementwiseVectorDecompositionDecisions(
              parent.artifact, (*miss)->actor);
      if (!vectorDecisions)
        return vectorDecisions.takeError();
      decisions.insert(decisions.end(),
                       std::make_move_iterator(vectorDecisions->begin()),
                       std::make_move_iterator(vectorDecisions->end()));
    }
    if (!llvm::is_sorted(decisions, dataflow::dataflowRewriteDecisionLess))
      return invalid("rewrite decision domain is not canonically ordered");

    std::vector<dataflow::StaticGraphLaunchRef> parentLaunches;
    auto parentView = parent.artifact.view();
    if (!parentView)
      return parentView.takeError();
    parentLaunches.reserve(parentView->staticGraphLaunches().size());
    for (const dataflow::CanonicalStaticGraphLaunchView &launch :
         parentView->staticGraphLaunches())
      parentLaunches.push_back(launch.ref);

    for (const dataflow::DataflowRewriteDecision &decision : decisions) {
      auto payload = dataflow::encodeDataflowRewriteDecision(decision);
      if (!payload)
        return payload.takeError();
      if (!attempted
               .insert(AttemptedDecisionKey{parent.artifact.identity(),
                                            std::move(*payload)})
               .second)
        continue;
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
          dataflow::materializeDataflowRewriteWithTrackedStaticGraphLaunches(
              parent.artifact, decision, parentLaunches);
      if (!child)
        return child.takeError();
      if (!*child)
        continue;
      if (llvm::Error error = publishChild(parent.reference, decision,
                                           parentLaunches, std::move(**child)))
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
    return CandidateGeneratorProviderResult{
        IncompleteCandidateGeneratorResult{
            CandidateGeneratorIncompleteReason::SemanticLimitReached,
            {std::move(output)},
            std::move(retainedLineageEdges)},
        {{CandidateGeneratorWorkUnitRef(0), expansions, expansions}}};
  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{{std::move(output)},
                                        std::move(retainedLineageEdges)},
      {{CandidateGeneratorWorkUnitRef(0), expansions, expansions}}};
}

const CandidateGeneratorProvider provider{
    descriptor.reference(),
    CandidateGeneratorInProcessProvider{invokeProvider}};

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
