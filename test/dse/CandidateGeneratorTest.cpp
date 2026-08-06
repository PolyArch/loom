#include "DSE/CandidateGenerator.h"

#include "Common/ArtifactStore.h"
#include "Common/ComponentViewDigest.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <vector>

namespace {

using loom::ArtifactIdentity;
using loom::ArtifactRootReference;
using loom::ArtifactSchemaDescriptor;
using loom::ComponentViewDigest;
using loom::SchemaVersion;
using namespace loom::dse;

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "candidate generator test failure: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
}

void requireErrorContains(llvm::Error error, llvm::StringRef expected) {
  if (!error)
    fail("expected an error");
  std::string message = llvm::toString(std::move(error));
  if (message.find(expected.str()) == std::string::npos)
    fail(("error did not contain expected text: " + expected).str());
}

constexpr ArtifactSchemaDescriptor inputSchema{"loom.test.generator_input",
                                               SchemaVersion{1, 0}};
constexpr ArtifactSchemaDescriptor outputSchema{"loom.test.generator_output",
                                                SchemaVersion{1, 0}};
constexpr ArtifactSchemaDescriptor foreignSchema{"loom.test.foreign",
                                                 SchemaVersion{1, 0}};
constexpr std::array<std::uint8_t, 4> configSchema = {0x54, 0x45, 0x53, 0x54};

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  if (bytes != llvm::ArrayRef<std::uint8_t>({0x01, 0x02}))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "test config bytes are not canonical");
  return loom::validateComponentViewDigest(configSchema, bytes, digest);
}

constexpr std::array<std::uint8_t, 4> decisionSchema = {0x44, 0x45, 0x43, 0x31};

llvm::Error validateDecision(llvm::ArrayRef<std::uint8_t> bytes,
                             llvm::ArrayRef<ArtifactRootReference> parents,
                             const loom::ArtifactStore &) {
  if (bytes != llvm::ArrayRef<std::uint8_t>({0xaa}))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "test decision is not canonical");
  if (parents.size() != 1 ||
      !((parents.front().schemaIdentity == inputSchema.identity &&
         parents.front().schemaVersion == inputSchema.version) ||
        (parents.front().schemaIdentity == outputSchema.identity &&
         parents.front().schemaVersion == outputSchema.version)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "test decision has a foreign parent");
  return llvm::Error::success();
}

const CandidateGeneratorOwnerLineagePayloadContract lineageContract{
    decisionSchema, validateDecision};

constexpr std::array<CandidateGeneratorInputSlotDescriptor, 1> inputSlots = {{{
    CandidateGeneratorInputSlotRef(0),
    "subject",
    PlanValueRole::CandidateSet,
    &inputSchema,
    PlanValueCardinality::ExactlyOne,
}}};
constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputSlots = {
    {{CandidateGeneratorOutputSlotRef(0), "candidate",
      PlanValueRole::CandidateSet, &outputSchema,
      PlanValueCardinality::FiniteSet}}};
constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 1> workUnits = {{{
    CandidateGeneratorWorkUnitRef(0),
    "candidate_attempt",
}}};

const CandidateGeneratorDescriptor descriptor{
    CandidateGeneratorKind(0x7fff0001),
    "test.generator",
    "loom.test.generator.v1",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{configSchema, validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
    &lineageContract,
};

enum class ProviderMode {
  Valid,
  MissingLineage,
  MalformedPayload,
  MechanicalDecisionFields,
  UnpublishedOutput,
  ForeignParent,
  UnpublishedParent,
  RecursiveLineage,
  ReconvergentLineage,
  DisconnectedLineage,
  UnrootedLineage,
  CyclicLineage,
};

ProviderMode providerMode = ProviderMode::Valid;

ArtifactRootReference makeReference(const ArtifactSchemaDescriptor &schema,
                                    std::uint8_t fill) {
  std::array<std::uint8_t, ArtifactIdentity::byteSize> bytes{};
  bytes.fill(fill);
  return ArtifactRootReference{schema.identity.str(), schema.version,
                               take(ArtifactIdentity::fromBytes(bytes))};
}

llvm::Expected<CandidateGeneratorInvocationOutcome>
invokeProvider(llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
               const ResolvedCandidateGeneratorBinding &,
               const loom::ArtifactStore &store) {
  const auto publish =
      [&](std::uint8_t byte) -> llvm::Expected<ArtifactRootReference> {
    auto identity = store.put(
        outputSchema,
        loom::CanonicalSemanticBytes(std::vector<std::uint8_t>{byte}));
    if (!identity)
      return identity.takeError();
    return ArtifactRootReference{outputSchema.identity.str(),
                                 outputSchema.version, *identity};
  };
  ArtifactRootReference output = makeReference(outputSchema, 0x22);
  if (providerMode != ProviderMode::UnpublishedOutput) {
    auto published = publish(0x22);
    if (!published)
      return published.takeError();
    output = std::move(*published);
  }
  std::vector<CandidateGeneratorLineageEdge> edges;
  if (providerMode != ProviderMode::MissingLineage) {
    CandidateGeneratorLineageEdgeKind kind =
        CandidateGeneratorLineageEdgeKind::CandidateDecision;
    std::vector<ArtifactRootReference> parents = {
        inputs.front().artifacts.front()};
    std::vector<std::uint8_t> payload = {
        providerMode == ProviderMode::MalformedPayload ? std::uint8_t{0xbb}
                                                       : std::uint8_t{0xaa}};
    if (providerMode == ProviderMode::MechanicalDecisionFields)
      kind = CandidateGeneratorLineageEdgeKind::MechanicalDerivation;
    if (providerMode == ProviderMode::ForeignParent)
      parents = {makeReference(foreignSchema, 0x33)};
    if (providerMode == ProviderMode::UnpublishedParent)
      parents = {makeReference(inputSchema, 0x44)};
    edges.push_back({kind, CandidateGeneratorOutputSlotRef(0), output,
                     std::move(parents), std::move(payload)});
    if (providerMode == ProviderMode::Valid)
      edges.push_back(edges.front());
  }
  if (providerMode == ProviderMode::RecursiveLineage ||
      providerMode == ProviderMode::ReconvergentLineage ||
      providerMode == ProviderMode::DisconnectedLineage ||
      providerMode == ProviderMode::UnrootedLineage ||
      providerMode == ProviderMode::CyclicLineage) {
    edges.clear();
    auto first = publish(0x23);
    auto second = publish(0x24);
    if (!first)
      return first.takeError();
    if (!second)
      return second.takeError();
    if (providerMode == ProviderMode::RecursiveLineage) {
      edges.push_back({CandidateGeneratorLineageEdgeKind::CandidateDecision,
                       CandidateGeneratorOutputSlotRef(0),
                       *first,
                       {inputs.front().artifacts.front()},
                       {0xaa}});
      edges.push_back({CandidateGeneratorLineageEdgeKind::CandidateDecision,
                       CandidateGeneratorOutputSlotRef(0),
                       output,
                       {*first},
                       {0xaa}});
    } else if (providerMode == ProviderMode::ReconvergentLineage) {
      edges.push_back({CandidateGeneratorLineageEdgeKind::CandidateDecision,
                       CandidateGeneratorOutputSlotRef(0),
                       *first,
                       {inputs.front().artifacts.front()},
                       {0xaa}});
      edges.push_back({CandidateGeneratorLineageEdgeKind::CandidateDecision,
                       CandidateGeneratorOutputSlotRef(0),
                       *second,
                       {inputs.front().artifacts.front()},
                       {0xaa}});
      edges.push_back({CandidateGeneratorLineageEdgeKind::CandidateDecision,
                       CandidateGeneratorOutputSlotRef(0),
                       output,
                       {*first},
                       {0xaa}});
      edges.push_back({CandidateGeneratorLineageEdgeKind::CandidateDecision,
                       CandidateGeneratorOutputSlotRef(0),
                       output,
                       {*second},
                       {0xaa}});
    } else if (providerMode == ProviderMode::DisconnectedLineage) {
      edges.push_back({CandidateGeneratorLineageEdgeKind::CandidateDecision,
                       CandidateGeneratorOutputSlotRef(0),
                       output,
                       {inputs.front().artifacts.front()},
                       {0xaa}});
      edges.push_back({CandidateGeneratorLineageEdgeKind::CandidateDecision,
                       CandidateGeneratorOutputSlotRef(0),
                       *first,
                       {inputs.front().artifacts.front()},
                       {0xaa}});
    } else if (providerMode == ProviderMode::UnrootedLineage) {
      edges.push_back({CandidateGeneratorLineageEdgeKind::CandidateDecision,
                       CandidateGeneratorOutputSlotRef(0),
                       output,
                       {*first},
                       {0xaa}});
    } else {
      edges.push_back({CandidateGeneratorLineageEdgeKind::CandidateDecision,
                       CandidateGeneratorOutputSlotRef(0),
                       *first,
                       {*second},
                       {0xaa}});
      edges.push_back({CandidateGeneratorLineageEdgeKind::CandidateDecision,
                       CandidateGeneratorOutputSlotRef(0),
                       *second,
                       {*first},
                       {0xaa}});
      output = *second;
    }
  }
  return CompletedCandidateGeneratorInvocation{
      {{CandidateGeneratorOutputSlotRef(0), {output}}}, std::move(edges)};
}

void exerciseRegistryAndBinding() {
  requireSuccess(registerCandidateGeneratorDescriptor(descriptor));
  requireSuccess(registerCandidateGeneratorDescriptor(descriptor));
  requireSuccess(registerCandidateGeneratorProvider(
      CandidateGeneratorProvider{descriptor.reference(), invokeProvider}));

  const CandidateGeneratorDescriptor *resolved =
      descriptor.reference().descriptor();
  if (resolved != &descriptor ||
      findCandidateGeneratorDescriptor(descriptor.kind) != &descriptor)
    fail("exact descriptor reference did not resolve the registered owner");

  const std::array<std::uint8_t, 2> canonicalConfig = {0x01, 0x02};
  const ComponentViewDigest digest =
      take(loom::computeComponentViewDigest(configSchema, canonicalConfig));
  std::vector<CandidateGeneratorInputBinding> inputs = {
      {CandidateGeneratorInputSlotRef(0), {makeReference(inputSchema, 0x11)}}};
  requireSuccess(
      validateCandidateGeneratorInputBindings(descriptor.reference(), inputs));
  std::vector<CandidateGeneratorInputBinding> duplicateInputs = {
      {CandidateGeneratorInputSlotRef(0),
       {makeReference(inputSchema, 0x11), makeReference(inputSchema, 0x11)}}};
  requireErrorContains(validateCandidateGeneratorInputBindings(
                           descriptor.reference(), duplicateInputs),
                       "canonical");
  auto binding = take(ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), canonicalConfig, digest));
  if (binding.descriptorRef() != descriptor.reference() ||
      !llvm::equal(binding.canonicalConfigBytes(), canonicalConfig) ||
      binding.configDigest() != digest)
    fail("resolved binding did not preserve exact descriptor-owned config");

  std::vector<CandidateGeneratorInputBinding> wrongSchema = {
      {CandidateGeneratorInputSlotRef(0), {makeReference(outputSchema, 0x22)}}};
  requireErrorContains(validateCandidateGeneratorInputBindings(
                           descriptor.reference(), wrongSchema),
                       "does not accept artifact schema");

  const ComponentViewDigest wrongDigest = take(loom::computeComponentViewDigest(
      configSchema, std::array<std::uint8_t, 1>{0x03}));
  auto stale = ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), canonicalConfig, wrongDigest);
  if (stale)
    fail("binding accepted a stale config digest");
  requireErrorContains(stale.takeError(), "digest");

  CandidateGeneratorDescriptor conflicting = descriptor;
  conflicting.spelling = "test.generator.conflict";
  requireErrorContains(registerCandidateGeneratorDescriptor(conflicting),
                       "conflicting registration");

  llvm::SmallString<128> storePath;
  if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
          "loom-candidate-generator", storePath))
    fail("cannot create ArtifactStore directory");
  loom::ArtifactStore store(storePath);
  const ArtifactIdentity inputIdentity = take(
      store.put(inputSchema,
                loom::CanonicalSemanticBytes(std::vector<std::uint8_t>{0x11})));
  inputs.front().artifacts = {ArtifactRootReference{
      inputSchema.identity.str(), inputSchema.version, inputIdentity}};

  providerMode = ProviderMode::Valid;
  auto validOutcome = take(invokeCandidateGenerator(inputs, binding, store));
  const auto *valid =
      std::get_if<CompletedCandidateGeneratorInvocation>(&validOutcome);
  if (!valid || valid->lineageEdges.size() != 1 ||
      valid->lineageEdges.front().ownerPayload !=
          std::vector<std::uint8_t>{0xaa})
    fail("controller did not canonicalize valid owner lineage");

  providerMode = ProviderMode::MissingLineage;
  auto missing = invokeCandidateGenerator(inputs, binding, store);
  if (missing)
    fail("controller accepted a generated output without lineage");
  requireErrorContains(missing.takeError(), "no lineage edge");

  providerMode = ProviderMode::MalformedPayload;
  auto malformed = invokeCandidateGenerator(inputs, binding, store);
  if (malformed)
    fail("controller accepted a malformed owner decision payload");
  requireErrorContains(malformed.takeError(), "not canonical");

  providerMode = ProviderMode::MechanicalDecisionFields;
  auto mechanical = invokeCandidateGenerator(inputs, binding, store);
  if (mechanical)
    fail("controller accepted decision fields on mechanical lineage");
  requireErrorContains(mechanical.takeError(), "mechanical lineage");

  providerMode = ProviderMode::UnpublishedOutput;
  auto unpublished = invokeCandidateGenerator(inputs, binding, store);
  if (unpublished)
    fail("controller accepted an output that was never durably published");
  requireErrorContains(unpublished.takeError(), "stored object is missing");

  providerMode = ProviderMode::ForeignParent;
  auto foreignParent = invokeCandidateGenerator(inputs, binding, store);
  if (foreignParent)
    fail("controller accepted an owner payload with a foreign parent");
  requireErrorContains(foreignParent.takeError(), "foreign parent");

  providerMode = ProviderMode::UnpublishedParent;
  auto unpublishedParent = invokeCandidateGenerator(inputs, binding, store);
  if (unpublishedParent)
    fail("controller accepted lineage from an unpublished parent");
  requireErrorContains(unpublishedParent.takeError(),
                       "stored object is missing");

  providerMode = ProviderMode::RecursiveLineage;
  auto recursive = take(invokeCandidateGenerator(inputs, binding, store));
  const auto *recursiveCompleted =
      std::get_if<CompletedCandidateGeneratorInvocation>(&recursive);
  if (!recursiveCompleted || recursiveCompleted->lineageEdges.size() != 2)
    fail("controller did not preserve a rooted internal lineage node");

  providerMode = ProviderMode::ReconvergentLineage;
  auto reconvergent = take(invokeCandidateGenerator(inputs, binding, store));
  const auto *reconvergentCompleted =
      std::get_if<CompletedCandidateGeneratorInvocation>(&reconvergent);
  if (!reconvergentCompleted || reconvergentCompleted->lineageEdges.size() != 4)
    fail("controller did not preserve a rooted reconvergent lineage DAG");

  providerMode = ProviderMode::DisconnectedLineage;
  auto disconnected = invokeCandidateGenerator(inputs, binding, store);
  if (disconnected)
    fail("controller accepted lineage unrelated to a returned output");
  requireErrorContains(disconnected.takeError(), "does not reach an output");

  providerMode = ProviderMode::UnrootedLineage;
  auto unrooted = invokeCandidateGenerator(inputs, binding, store);
  if (unrooted)
    fail("controller accepted lineage without an invocation root");
  requireErrorContains(unrooted.takeError(), "not an invocation input");

  providerMode = ProviderMode::CyclicLineage;
  auto cyclic = invokeCandidateGenerator(inputs, binding, store);
  if (cyclic)
    fail("controller accepted cyclic candidate lineage");
  requireErrorContains(cyclic.takeError(), "cycle");

  llvm::sys::fs::remove_directories(storePath);
}

} // namespace

int main() {
  exerciseRegistryAndBinding();
  return 0;
}
