#include "DSE/CandidateGenerator.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobDigest.h"
#include "Common/BlobStore.h"
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

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
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
    loom::ProviderForm::InProcess,
};

enum class ProviderMode {
  Valid,
  MissingWorkSummary,
  OverconsumedWork,
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

llvm::Expected<CandidateGeneratorProviderResult>
invokeProvider(llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
               const ResolvedCandidateGeneratorBinding &,
               const loom::ArtifactStore &store, const loom::BlobStore &,
               const loom::ExecutionControlView &) {
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
  std::vector<CandidateGeneratorWorkUnitSummary> workSummary;
  if (providerMode != ProviderMode::MissingWorkSummary) {
    const std::uint64_t planned = 1;
    const std::uint64_t consumed =
        providerMode == ProviderMode::OverconsumedWork ? 2 : 1;
    workSummary.push_back(
        {CandidateGeneratorWorkUnitRef(0), planned, consumed});
  }
  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{
          {{CandidateGeneratorOutputSlotRef(0), {output}}}, std::move(edges)},
      std::move(workSummary)};
}

llvm::Expected<loom::external_tool::PreparedExternalToolInvocation>
prepareStub(llvm::ArrayRef<CandidateGeneratorInputBinding>,
            const ResolvedCandidateGeneratorBinding &,
            const loom::ArtifactStore &, const loom::BlobStore &,
            const loom::external_tool::ExternalToolPreparationContext &) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "test external prepare stub");
}

llvm::Expected<CandidateGeneratorProviderResult>
importStub(llvm::ArrayRef<CandidateGeneratorInputBinding>,
           const ResolvedCandidateGeneratorBinding &,
           const loom::external_tool::PreparedExternalToolInvocation &,
           const loom::ArtifactStore &, const loom::BlobStore &) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "test external import stub");
}

llvm::Expected<loom::external_tool::PreparedExternalToolInvocation>
prepareStubAlternate(
    llvm::ArrayRef<CandidateGeneratorInputBinding>,
    const ResolvedCandidateGeneratorBinding &, const loom::ArtifactStore &,
    const loom::BlobStore &,
    const loom::external_tool::ExternalToolPreparationContext &) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "test external prepare stub");
}

llvm::Expected<CandidateGeneratorProviderResult>
importStubAlternate(llvm::ArrayRef<CandidateGeneratorInputBinding>,
                    const ResolvedCandidateGeneratorBinding &,
                    const loom::external_tool::PreparedExternalToolInvocation &,
                    const loom::ArtifactStore &, const loom::BlobStore &) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "test external import stub");
}

// One hundred planned and consumed attempts with exactly two outputs: work
// accounting never collapses to output cardinality.
llvm::Expected<CandidateGeneratorProviderResult>
importStubValid(llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
                const ResolvedCandidateGeneratorBinding &,
                const loom::external_tool::PreparedExternalToolInvocation &,
                const loom::ArtifactStore &store, const loom::BlobStore &) {
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
  auto first = publish(0x22);
  if (!first)
    return first.takeError();
  auto second = publish(0x23);
  if (!second)
    return second.takeError();
  std::vector<CandidateGeneratorLineageEdge> edges;
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
  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{
          {{CandidateGeneratorOutputSlotRef(0), {*first, *second}}},
          std::move(edges)},
      {{CandidateGeneratorWorkUnitRef(0), 100, 100}}};
}

llvm::Expected<CandidateGeneratorProviderResult> importStubIncomplete(
    llvm::ArrayRef<CandidateGeneratorInputBinding>,
    const ResolvedCandidateGeneratorBinding &,
    const loom::external_tool::PreparedExternalToolInvocation &,
    const loom::ArtifactStore &, const loom::BlobStore &) {
  return CandidateGeneratorProviderResult{
      IncompleteCandidateGeneratorResult{
          CandidateGeneratorIncompleteReason::ExecutionFailed,
          {{CandidateGeneratorOutputSlotRef(0), {}}},
          {}},
      {{CandidateGeneratorWorkUnitRef(0), 100, 100}}};
}

llvm::Expected<CandidateGeneratorProviderResult>
importStubMalformed(llvm::ArrayRef<CandidateGeneratorInputBinding>,
                    const ResolvedCandidateGeneratorBinding &,
                    const loom::external_tool::PreparedExternalToolInvocation &,
                    const loom::ArtifactStore &store, const loom::BlobStore &) {
  auto identity =
      store.put(outputSchema,
                loom::CanonicalSemanticBytes(std::vector<std::uint8_t>{0x24}));
  if (!identity)
    return identity.takeError();
  ArtifactRootReference output{outputSchema.identity.str(),
                               outputSchema.version, *identity};
  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{
          {{CandidateGeneratorOutputSlotRef(0), {output}}}, {}},
      {{CandidateGeneratorWorkUnitRef(0), 1, 1}}};
}

void externalProviderFormAdmission() {
  static_assert(static_cast<std::uint32_t>(loom::ProviderForm::InProcess) == 0,
                "InProcess must keep stable tag 0");
  static_assert(static_cast<std::uint32_t>(
                    loom::ProviderForm::ExternalPrepareImport) == 1,
                "ExternalPrepareImport must keep stable tag 1");
  static_assert(static_cast<std::uint32_t>(
                    CandidateGeneratorIncompleteReason::ExecutionFailed) == 4,
                "ExecutionFailed must keep stable tag 4");
  static_assert(static_cast<std::uint32_t>(
                    CandidateGeneratorIncompleteReason::CancelledOrTimeout) ==
                    5,
                "CancelledOrTimeout must keep stable tag 5");

  // A registry-1.0 descriptor reference is not reinterpreted.
  auto legacy = CandidateGeneratorDescriptorRef::get(
      {"loom.candidate_generator_descriptor", SchemaVersion{1, 0}},
      descriptor.kind);
  if (legacy)
    fail("a registry-1.0 descriptor reference was reinterpreted");
  llvm::consumeError(legacy.takeError());

  CandidateGeneratorDescriptor external = descriptor;
  external.kind = CandidateGeneratorKind(0x7fff0002);
  external.spelling = "test.generator.external";
  external.providerForm = loom::ProviderForm::ExternalPrepareImport;
  requireSuccess(registerCandidateGeneratorDescriptor(external));

  // In-process callbacks cannot serve an external descriptor.
  requireErrorContains(
      registerCandidateGeneratorProvider(CandidateGeneratorProvider{
          external.reference(),
          CandidateGeneratorInProcessProvider{invokeProvider}}),
      "provider form");
  // Both external callbacks are required.
  requireErrorContains(
      registerCandidateGeneratorProvider(CandidateGeneratorProvider{
          external.reference(),
          CandidateGeneratorExternalPrepareImportProvider{prepareStub,
                                                          nullptr}}),
      "prepare and import");
  // The matching closed external form registers.
  requireSuccess(registerCandidateGeneratorProvider(CandidateGeneratorProvider{
      external.reference(), CandidateGeneratorExternalPrepareImportProvider{
                                prepareStub, importStub}}));
  // A second provider for the same descriptor is rejected.
  requireErrorContains(
      registerCandidateGeneratorProvider(CandidateGeneratorProvider{
          external.reference(),
          CandidateGeneratorExternalPrepareImportProvider{
              prepareStubAlternate, importStubAlternate}}),
      "conflicting provider registration");

  const std::array<std::uint8_t, 2> canonicalConfig = {0x01, 0x02};
  const ComponentViewDigest digest =
      take(loom::computeComponentViewDigest(configSchema, canonicalConfig));
  std::vector<CandidateGeneratorInputBinding> inputs = {
      {CandidateGeneratorInputSlotRef(0), {makeReference(inputSchema, 0x11)}}};
  llvm::SmallString<128> storePath;
  if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
          "loom-external-generator", storePath))
    fail("cannot create ArtifactStore directory");
  llvm::SmallString<128> blobPath(storePath);
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  loom::ArtifactStore store(storePath);
  const loom::BlobStore blobs(blobPath);
  const ArtifactIdentity inputIdentity = take(
      store.put(inputSchema,
                loom::CanonicalSemanticBytes(std::vector<std::uint8_t>{0x11})));
  inputs.front().artifacts = {ArtifactRootReference{
      inputSchema.identity.str(), inputSchema.version, inputIdentity}};

  // A registered external descriptor without any provider still rejects the
  // in-process facade for form, never a provider-absence outcome.
  CandidateGeneratorDescriptor providerlessExternal = external;
  providerlessExternal.kind = CandidateGeneratorKind(0x7fff0003);
  providerlessExternal.spelling = "test.generator.external_providerless";
  requireSuccess(registerCandidateGeneratorDescriptor(providerlessExternal));
  auto providerlessBinding = take(ResolvedCandidateGeneratorBinding::get(
      providerlessExternal.reference(), canonicalConfig, digest));
  auto providerlessOutcome =
      invokeCandidateGenerator(inputs, providerlessBinding, store, blobs);
  if (providerlessOutcome)
    fail("the facade accepted an external descriptor without a provider");
  requireErrorContains(providerlessOutcome.takeError(), "external");

  // A missing InProcess provider alone keeps the typed ProviderUnavailable
  // outcome with the all-zero dense work summary.
  CandidateGeneratorDescriptor providerlessInProcess = descriptor;
  providerlessInProcess.kind = CandidateGeneratorKind(0x7fff0004);
  providerlessInProcess.spelling = "test.generator.in_process_providerless";
  requireSuccess(registerCandidateGeneratorDescriptor(providerlessInProcess));
  auto inProcessBinding = take(ResolvedCandidateGeneratorBinding::get(
      providerlessInProcess.reference(), canonicalConfig, digest));
  auto unavailable =
      take(invokeCandidateGenerator(inputs, inProcessBinding, store, blobs));
  const auto *unavailableIncomplete =
      std::get_if<IncompleteCandidateGeneratorResult>(&unavailable.outcome);
  if (!unavailableIncomplete ||
      unavailableIncomplete->reason !=
          CandidateGeneratorIncompleteReason::ProviderUnavailable ||
      unavailable.workSummary != std::vector<CandidateGeneratorWorkUnitSummary>{
                                     {CandidateGeneratorWorkUnitRef(0), 0, 0}})
    fail("a missing in-process provider lost the zero dense work summary");

  // The in-process facade never invokes a registered external provider.
  auto binding = take(ResolvedCandidateGeneratorBinding::get(
      external.reference(), canonicalConfig, digest));
  const loom::external_tool::ExternalToolSemanticContract semanticContract =
      take(deriveExternalToolSemanticContract(inputs, binding));
  if (semanticContract.providerIdentity !=
      external.implementationSemanticIdentity)
    fail("external semantic contract lost the generator provider identity");
  const auto *generatorClosure =
      std::get_if<loom::external_tool::CandidateGeneratorInvocationClosure>(
          &semanticContract.semanticClosure);
  if (!generatorClosure)
    fail("external semantic contract used the wrong closure form");
  std::vector<std::uint8_t> expectedInputs;
  appendU64(expectedInputs, 1);
  appendU32(expectedInputs, 0);
  appendU64(expectedInputs, 1);
  const std::vector<std::uint8_t> rootBytes =
      loom::encodeArtifactRootReference(inputs.front().artifacts.front());
  expectedInputs.insert(expectedInputs.end(), rootBytes.begin(),
                        rootBytes.end());
  if (generatorClosure->typedInputBindings != expectedInputs)
    fail("external semantic contract changed typed input-binding bytes");
  std::vector<std::uint8_t> expectedBinding =
      canonicalCandidateGeneratorDescriptorReferenceBytes(
          binding.descriptorRef());
  appendU64(expectedBinding, canonicalConfig.size());
  expectedBinding.insert(expectedBinding.end(), canonicalConfig.begin(),
                         canonicalConfig.end());
  expectedBinding.insert(expectedBinding.end(), digest.bytes().begin(),
                         digest.bytes().end());
  if (generatorClosure->resolvedBinding != expectedBinding ||
      generatorClosure->bindingIdentity !=
          deriveCandidateGeneratorBindingIdentity(binding.descriptorRef(),
                                                  canonicalConfig)
              .bytes())
    fail("external semantic contract changed resolved binding ownership");
  if (semanticContract.resultImporterIdentity !=
      "dc4fb4b088e761ffe13197599910723ed687927885bb1973b084e0d5edbad4ac")
    fail("external semantic contract changed the generator importer identity");
  auto inProcessContract =
      deriveExternalToolSemanticContract(inputs, inProcessBinding);
  if (inProcessContract)
    fail("an in-process generator acquired an external semantic contract");
  requireErrorContains(inProcessContract.takeError(), "ExternalPrepareImport");
  auto outcome = invokeCandidateGenerator(inputs, binding, store, blobs);
  if (outcome)
    fail("the in-process facade invoked an external provider");
  requireErrorContains(outcome.takeError(), "external");

  // The external facades reject the in-process form before any lookup.
  auto inProcessFacadeBinding = take(ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), canonicalConfig, digest));
  loom::external_tool::ExternalToolPreparationContext context{
      loom::external_tool::defaultLocalToolConfig(), storePath.str().str()};
  auto wrongPrepare = prepareCandidateGeneratorInvocation(
      inputs, inProcessFacadeBinding, store, blobs, context);
  if (wrongPrepare)
    fail("an in-process descriptor used the external prepare facade");
  requireErrorContains(wrongPrepare.takeError(), "in-process");
  const loom::BlobDigest zeroDigest = loom::computeBlobDigest({});
  auto wrongImport = importCandidateGeneratorInvocation(
      inputs, inProcessFacadeBinding,
      loom::external_tool::PreparedExternalToolInvocation{"unused", zeroDigest},
      store, blobs);
  if (wrongImport)
    fail("an in-process descriptor used the external import facade");
  requireErrorContains(wrongImport.takeError(), "in-process");

  // The external facades dispatch the registered external provider.
  auto prepared = prepareCandidateGeneratorInvocation(inputs, binding, store,
                                                      blobs, context);
  if (prepared)
    fail("the external prepare facade returned a bundle from a stub");
  requireErrorContains(prepared.takeError(), "test external prepare stub");
  auto imported = importCandidateGeneratorInvocation(
      inputs, binding,
      loom::external_tool::PreparedExternalToolInvocation{"unused", zeroDigest},
      store, blobs);
  if (imported)
    fail("the external import facade returned a result from a stub");
  requireErrorContains(imported.takeError(), "test external import stub");

  // A successful external import is validated against the full closure: one
  // hundred planned and consumed attempts with exactly two outputs never
  // collapse to output cardinality.
  CandidateGeneratorDescriptor validExternal = external;
  validExternal.kind = CandidateGeneratorKind(0x7fff0005);
  validExternal.spelling = "test.generator.external_valid";
  requireSuccess(registerCandidateGeneratorDescriptor(validExternal));
  requireSuccess(registerCandidateGeneratorProvider(CandidateGeneratorProvider{
      validExternal.reference(),
      CandidateGeneratorExternalPrepareImportProvider{prepareStub,
                                                      importStubValid}}));
  auto validBinding = take(ResolvedCandidateGeneratorBinding::get(
      validExternal.reference(), canonicalConfig, digest));
  auto validResult = take(importCandidateGeneratorInvocation(
      inputs, validBinding,
      loom::external_tool::PreparedExternalToolInvocation{"unused", zeroDigest},
      store, blobs));
  const auto *validCompleted =
      std::get_if<CompletedCandidateGeneratorResult>(&validResult.outcome);
  if (!validCompleted || validCompleted->outputBindings.size() != 1 ||
      validCompleted->outputBindings.front().artifacts.size() != 2)
    fail("external import lost the exact output bindings");
  if (validResult.workSummary !=
      std::vector<CandidateGeneratorWorkUnitSummary>{
          {CandidateGeneratorWorkUnitRef(0), 100, 100}})
    fail("external import collapsed work accounting to output cardinality");

  // An incomplete import with no outputs retains the exact consumed work.
  CandidateGeneratorDescriptor incompleteExternal = external;
  incompleteExternal.kind = CandidateGeneratorKind(0x7fff0006);
  incompleteExternal.spelling = "test.generator.external_incomplete";
  requireSuccess(registerCandidateGeneratorDescriptor(incompleteExternal));
  requireSuccess(registerCandidateGeneratorProvider(CandidateGeneratorProvider{
      incompleteExternal.reference(),
      CandidateGeneratorExternalPrepareImportProvider{prepareStub,
                                                      importStubIncomplete}}));
  auto incompleteBinding = take(ResolvedCandidateGeneratorBinding::get(
      incompleteExternal.reference(), canonicalConfig, digest));
  auto incompleteResult = take(importCandidateGeneratorInvocation(
      inputs, incompleteBinding,
      loom::external_tool::PreparedExternalToolInvocation{"unused", zeroDigest},
      store, blobs));
  const auto *incompleteOutcome =
      std::get_if<IncompleteCandidateGeneratorResult>(
          &incompleteResult.outcome);
  if (!incompleteOutcome ||
      incompleteOutcome->reason !=
          CandidateGeneratorIncompleteReason::ExecutionFailed ||
      !incompleteOutcome->retainedOutputBindings.front().artifacts.empty())
    fail("external import lost the incomplete outcome");
  if (incompleteResult.workSummary !=
      std::vector<CandidateGeneratorWorkUnitSummary>{
          {CandidateGeneratorWorkUnitRef(0), 100, 100}})
    fail("incomplete external import lost the exact consumed work");

  // A malformed provider result is rejected by the controller validation.
  CandidateGeneratorDescriptor malformedExternal = external;
  malformedExternal.kind = CandidateGeneratorKind(0x7fff0007);
  malformedExternal.spelling = "test.generator.external_malformed";
  requireSuccess(registerCandidateGeneratorDescriptor(malformedExternal));
  requireSuccess(registerCandidateGeneratorProvider(CandidateGeneratorProvider{
      malformedExternal.reference(),
      CandidateGeneratorExternalPrepareImportProvider{prepareStub,
                                                      importStubMalformed}}));
  auto malformedBinding = take(ResolvedCandidateGeneratorBinding::get(
      malformedExternal.reference(), canonicalConfig, digest));
  auto malformedResult = importCandidateGeneratorInvocation(
      inputs, malformedBinding,
      loom::external_tool::PreparedExternalToolInvocation{"unused", zeroDigest},
      store, blobs);
  if (malformedResult)
    fail("a malformed external import was accepted");
  requireErrorContains(malformedResult.takeError(), "lineage");

  if (std::error_code error = llvm::sys::fs::remove_directories(storePath))
    fail("could not remove ArtifactStore directory: " + error.message());
}

void exerciseRegistryAndBinding() {
  requireSuccess(registerCandidateGeneratorDescriptor(descriptor));
  requireSuccess(registerCandidateGeneratorDescriptor(descriptor));
  requireSuccess(registerCandidateGeneratorProvider(CandidateGeneratorProvider{
      descriptor.reference(),
      CandidateGeneratorInProcessProvider{invokeProvider}}));

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
  llvm::SmallString<128> blobPath(storePath);
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  loom::ArtifactStore store(storePath);
  const loom::BlobStore blobs(blobPath);
  const ArtifactIdentity inputIdentity = take(
      store.put(inputSchema,
                loom::CanonicalSemanticBytes(std::vector<std::uint8_t>{0x11})));
  inputs.front().artifacts = {ArtifactRootReference{
      inputSchema.identity.str(), inputSchema.version, inputIdentity}};

  providerMode = ProviderMode::Valid;
  auto validOutcome =
      take(invokeCandidateGenerator(inputs, binding, store, blobs));
  const auto *valid =
      std::get_if<CompletedCandidateGeneratorResult>(&validOutcome.outcome);
  if (!valid || valid->lineageEdges.size() != 1 ||
      validOutcome.workSummary !=
          std::vector<CandidateGeneratorWorkUnitSummary>{
              {CandidateGeneratorWorkUnitRef(0), 1, 1}} ||
      valid->lineageEdges.front().ownerPayload !=
          std::vector<std::uint8_t>{0xaa})
    fail("controller did not canonicalize valid owner lineage");

  providerMode = ProviderMode::MissingWorkSummary;
  auto missingWork = invokeCandidateGenerator(inputs, binding, store, blobs);
  if (missingWork)
    fail("controller accepted a missing work-unit summary");
  requireErrorContains(missingWork.takeError(), "work summary");

  providerMode = ProviderMode::OverconsumedWork;
  auto overconsumed = invokeCandidateGenerator(inputs, binding, store, blobs);
  if (overconsumed)
    fail("controller accepted consumed work above planned work");
  requireErrorContains(overconsumed.takeError(), "exceeds planned");

  providerMode = ProviderMode::MissingLineage;
  auto missing = invokeCandidateGenerator(inputs, binding, store, blobs);
  if (missing)
    fail("controller accepted a generated output without lineage");
  requireErrorContains(missing.takeError(), "no lineage edge");

  providerMode = ProviderMode::MalformedPayload;
  auto malformed = invokeCandidateGenerator(inputs, binding, store, blobs);
  if (malformed)
    fail("controller accepted a malformed owner decision payload");
  requireErrorContains(malformed.takeError(), "not canonical");

  providerMode = ProviderMode::MechanicalDecisionFields;
  auto mechanical = invokeCandidateGenerator(inputs, binding, store, blobs);
  if (mechanical)
    fail("controller accepted decision fields on mechanical lineage");
  requireErrorContains(mechanical.takeError(), "mechanical lineage");

  providerMode = ProviderMode::UnpublishedOutput;
  auto unpublished = invokeCandidateGenerator(inputs, binding, store, blobs);
  if (unpublished)
    fail("controller accepted an output that was never durably published");
  requireErrorContains(unpublished.takeError(), "stored object is missing");

  providerMode = ProviderMode::ForeignParent;
  auto foreignParent = invokeCandidateGenerator(inputs, binding, store, blobs);
  if (foreignParent)
    fail("controller accepted an owner payload with a foreign parent");
  requireErrorContains(foreignParent.takeError(), "foreign parent");

  providerMode = ProviderMode::UnpublishedParent;
  auto unpublishedParent =
      invokeCandidateGenerator(inputs, binding, store, blobs);
  if (unpublishedParent)
    fail("controller accepted lineage from an unpublished parent");
  requireErrorContains(unpublishedParent.takeError(),
                       "stored object is missing");

  providerMode = ProviderMode::RecursiveLineage;
  auto recursive =
      take(invokeCandidateGenerator(inputs, binding, store, blobs));
  const auto *recursiveCompleted =
      std::get_if<CompletedCandidateGeneratorResult>(&recursive.outcome);
  if (!recursiveCompleted || recursiveCompleted->lineageEdges.size() != 2)
    fail("controller did not preserve a rooted internal lineage node");

  providerMode = ProviderMode::ReconvergentLineage;
  auto reconvergent =
      take(invokeCandidateGenerator(inputs, binding, store, blobs));
  const auto *reconvergentCompleted =
      std::get_if<CompletedCandidateGeneratorResult>(&reconvergent.outcome);
  if (!reconvergentCompleted || reconvergentCompleted->lineageEdges.size() != 4)
    fail("controller did not preserve a rooted reconvergent lineage DAG");

  providerMode = ProviderMode::DisconnectedLineage;
  auto disconnected = invokeCandidateGenerator(inputs, binding, store, blobs);
  if (disconnected)
    fail("controller accepted lineage unrelated to a returned output");
  requireErrorContains(disconnected.takeError(), "does not reach an output");

  providerMode = ProviderMode::UnrootedLineage;
  auto unrooted = invokeCandidateGenerator(inputs, binding, store, blobs);
  if (unrooted)
    fail("controller accepted lineage without an invocation root");
  requireErrorContains(unrooted.takeError(), "not an invocation input");

  providerMode = ProviderMode::CyclicLineage;
  auto cyclic = invokeCandidateGenerator(inputs, binding, store, blobs);
  if (cyclic)
    fail("controller accepted cyclic candidate lineage");
  requireErrorContains(cyclic.takeError(), "cycle");

  llvm::sys::fs::remove_directories(storePath);
}

} // namespace

void bindingIdentityDerivationUsesExactFraming() {
  auto reference = take(CandidateGeneratorDescriptorRef::get(
      candidateGeneratorDescriptorSchema, CandidateGeneratorKind(9)));
  const std::vector<std::uint8_t> expectedReference = {
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x23, 0x6c, 0x6f, 0x6f,
      0x6d, 0x2e, 0x63, 0x61, 0x6e, 0x64, 0x69, 0x64, 0x61, 0x74, 0x65,
      0x5f, 0x67, 0x65, 0x6e, 0x65, 0x72, 0x61, 0x74, 0x6f, 0x72, 0x5f,
      0x64, 0x65, 0x73, 0x63, 0x72, 0x69, 0x70, 0x74, 0x6f, 0x72, 0x00,
      0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09};
  const std::vector<std::uint8_t> referenceBytes =
      canonicalCandidateGeneratorDescriptorReferenceBytes(reference);
  if (referenceBytes != expectedReference)
    fail("descriptor reference canonical framing changed");

  const std::array<std::uint8_t, 2> config = {0x01, 0x02};
  const loom::BlobDigest identity =
      deriveCandidateGeneratorBindingIdentity(reference, config);
  if (loom::formatBlobDigestHex(identity) !=
      "2c41f65b6070e47fa7a55c915c5bc98fe66f438cb3d7c60edf641113abe49652")
    fail("binding identity derivation framing changed");

  auto zeroReference = take(CandidateGeneratorDescriptorRef::get(
      candidateGeneratorDescriptorSchema, CandidateGeneratorKind(0)));
  const loom::BlobDigest emptyIdentity =
      deriveCandidateGeneratorBindingIdentity(zeroReference, {});
  if (loom::formatBlobDigestHex(emptyIdentity) !=
      "0fcee1e00f4bf2c64e0cf114cfe954136fe65a45dd52952c7cfbac8c7e508634")
    fail("empty-config binding identity derivation framing changed");
}

int main() {
  exerciseRegistryAndBinding();
  bindingIdentityDerivationUsesExactFraming();
  externalProviderFormAdmission();
  return 0;
}
