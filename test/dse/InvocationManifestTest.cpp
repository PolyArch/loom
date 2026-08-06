#include "DSE/InvocationManifest.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Common/ComponentViewDigest.h"
#include "Config/ResolvedConfig.h"
#include "DSE/ResolvedConfigView.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/ModelDescriptor.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace loom;
using namespace loom::dse;

[[noreturn]] void fail(const std::string &message) {
  std::cerr << "invocation manifest test failure: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireErrorContains(llvm::Error error, llvm::StringRef needle) {
  const std::string message = llvm::toString(std::move(error));
  if (message.find(needle.str()) == std::string::npos)
    fail("expected error containing '" + needle.str() + "', got: " + message);
}

constexpr ArtifactSchemaDescriptor sourceSchema{
    "loom.test.invocation_manifest_source", SchemaVersion{1, 0}};
constexpr ArtifactSchemaDescriptor candidateSchema{
    "loom.test.invocation_manifest_candidate", SchemaVersion{1, 0}};
constexpr std::array<std::uint8_t, 4> configSchema = {0x49, 0x4d, 0x41, 0x4e};

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  if (bytes != llvm::ArrayRef<std::uint8_t>({0x01}))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "manifest test config is not canonical");
  return validateComponentViewDigest(configSchema, bytes, digest);
}

constexpr std::array<CandidateGeneratorInputSlotDescriptor, 1> inputs = {{{
    CandidateGeneratorInputSlotRef(0),
    "source",
    PlanValueRole::CandidateSet,
    &sourceSchema,
    PlanValueCardinality::ExactlyOne,
}}};
constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputs = {{{
    CandidateGeneratorOutputSlotRef(0),
    "candidate",
    PlanValueRole::CandidateSet,
    &candidateSchema,
    PlanValueCardinality::NonEmptySet,
}}};
constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 1> workUnits = {{{
    CandidateGeneratorWorkUnitRef(0),
    "candidate_attempt",
}}};

const CandidateGeneratorDescriptor generator{
    CandidateGeneratorKind(0x7fff7100),
    "test.invocation_manifest",
    "loom.test.invocation_manifest.v1",
    inputs,
    outputs,
    ResolvedDseConfigViewContract{configSchema, validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
    nullptr,
};

constexpr evaluation::EvaluationCaseKind evidenceCaseKind(0x7fff7100);
constexpr evaluation::EvaluationModelKind evidenceModelKind(0x7fff7100);
constexpr evaluation::CaseSubjectRoleRef evidenceSubjectRole(0);

evaluation::EvaluationCaseSignatureRef evidenceSignatureRef() {
  return llvm::cantFail(evaluation::EvaluationCaseSignatureRef::get(
      evaluation::evaluationSchemaVersion(), evidenceCaseKind));
}

const ArtifactSchemaDescriptor *const evidenceSubjectSchemas[] = {
    &sourceSchema};
const evaluation::CaseSubjectRoleDescriptor evidenceSubjectRoles[] = {{
    evidenceSubjectRole,
    "subject",
    evaluation::SubjectRoleCardinality::ExactlyOne,
    evidenceSubjectSchemas,
    nullptr,
}};
const evaluation::EvaluationCaseSignatureDescriptor evidenceCaseSignature{
    evidenceCaseKind,
    "invocation_manifest_evidence_case",
    "One exact source used by InvocationManifest Evidence records.",
    evidenceSubjectRoles,
    evaluation::ArtifactRequirement::Forbidden,
    {},
    evaluation::ArtifactRequirement::Forbidden,
    {},
    nullptr,
    evaluation::AbsentReferenceCycle{},
    {}};

struct EmptyEvidenceModelConfig final {};

llvm::ArrayRef<std::uint8_t> evidenceModelConfigSchema() {
  static constexpr llvm::StringLiteral descriptor =
      "loom.test.invocation_manifest.evidence_model_config.1.0";
  return {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
          descriptor.size()};
}

llvm::Expected<evaluation::OwnerValue>
projectEvidenceModelConfig(const ResolvedConfig &) {
  return evaluation::OwnerValue::get(EmptyEvidenceModelConfig{});
}

llvm::Expected<std::vector<std::uint8_t>>
encodeEvidenceModelConfig(const evaluation::OwnerValue &value) {
  if (!value.getIf<EmptyEvidenceModelConfig>())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Evidence model config has the wrong type");
  return std::vector<std::uint8_t>{};
}

llvm::Expected<evaluation::OwnerValue>
adoptEvidenceModelConfig(llvm::ArrayRef<std::uint8_t> bytes,
                         const ComponentViewDigest &) {
  if (!bytes.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Evidence model config must have empty canonical bytes");
  return evaluation::OwnerValue::get(EmptyEvidenceModelConfig{});
}

const evaluation::ScopeFormRef evidenceScopeForms[] = {
    evaluation::ScopeFormRef(0)};
const evaluation::MetricCapability evidenceMetricCapabilities[] = {{
    evaluation::MetricKind::Runtime,
    evidenceScopeForms,
    evaluation::allObservationFormsMask(),
}};
const evaluation::EvaluationModelDescriptor evidenceModelDescriptor{
    evidenceModelKind,
    "invocation_manifest_evidence_model",
    "loom.test.invocation_manifest.evidence_model.v1",
    evidenceSignatureRef(),
    {},
    evidenceMetricCapabilities,
    {},
    {},
    {},
    {evidenceModelConfigSchema(), &projectEvidenceModelConfig,
     &encodeEvidenceModelConfig, &adoptEvidenceModelConfig},
    {},
    evaluation::EvaluationExecutionMethod::Analytic,
    {},
    evaluation::DeterminismContract::Deterministic,
    {}};

bool generationStopsEarly = false;

ArtifactRootReference publish(const ArtifactStore &store,
                              const ArtifactSchemaDescriptor &schema,
                              std::uint8_t byte) {
  const ArtifactIdentity identity = take(store.put(
      schema, CanonicalSemanticBytes(std::vector<std::uint8_t>{byte})));
  return {schema.identity.str(), schema.version, identity};
}

llvm::Expected<CandidateGeneratorProviderResult>
generate(llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
         const ResolvedCandidateGeneratorBinding &, const ArtifactStore &store,
         const BlobStore &) {
  const ArtifactRootReference candidate = publish(store, candidateSchema, 0x31);
  if (generationStopsEarly)
    return CandidateGeneratorProviderResult{
        IncompleteCandidateGeneratorResult{
            CandidateGeneratorIncompleteReason::SemanticLimitReached,
            {{CandidateGeneratorOutputSlotRef(0), {candidate}}},
            {{CandidateGeneratorLineageEdgeKind::MechanicalDerivation,
              CandidateGeneratorOutputSlotRef(0),
              candidate,
              {},
              {}}}},
        {{CandidateGeneratorWorkUnitRef(0), 4, 1}}};
  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{
          {{CandidateGeneratorOutputSlotRef(0), {candidate}}},
          {{CandidateGeneratorLineageEdgeKind::MechanicalDerivation,
            CandidateGeneratorOutputSlotRef(0),
            candidate,
            {},
            {}}}},
      {{CandidateGeneratorWorkUnitRef(0), inputBindings.size(), 1}}};
}

void registerOwner() {
  if (llvm::Error error =
          evaluation::registerEvaluationCaseSignature(evidenceCaseSignature))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = evaluation::registerEvaluationModelDescriptor(
          evidenceModelDescriptor))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = registerCandidateGeneratorDescriptor(generator))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = registerCandidateGeneratorProvider(
          CandidateGeneratorProvider{generator.reference(),
                                     CandidateGeneratorInProcessProvider{
                                         generate}}))
    fail(llvm::toString(std::move(error)));
}

ArtifactRootReference publishEvidence(const ArtifactStore &store,
                                      const ArtifactRootReference &subject,
                                      std::uint64_t observation) {
  evaluation::CaseArtifactResolution resolution =
      take(evaluation::CaseArtifactResolution::get({{subject, {}}}));
  evaluation::EvaluationSubjectBindings subjects =
      take(evaluation::EvaluationSubjectBindings::get(
          {{evidenceSubjectRole, {subject}}}));
  evaluation::EvaluationCase evaluationCase =
      take(evaluation::EvaluationCase::get(
          evidenceSignatureRef(), std::move(subjects), std::nullopt,
          std::nullopt, {}, resolution, store));
  evaluation::MetricRequest metric = take(evaluation::MetricRequest::get(
      {evaluation::MetricKind::Runtime,
       evaluation::EvaluationScope{evaluation::ScopeFormRef(0), {}}},
      {}, evaluationCase, resolution, store));
  evaluation::ResolvedModelBinding model =
      take(evaluation::ResolvedModelBinding::project(
          evidenceModelDescriptor.reference(), {}, defaultResolvedConfig()));
  evaluation::EvaluationRequest request =
      take(evaluation::EvaluationRequest::get(evaluationCase, {metric}, {},
                                              std::move(model), 0, resolution,
                                              store));
  take(evaluation::publishEvaluationRequest(request, store));
  evaluation::EvaluationEvidence evidence =
      take(evaluation::EvaluationEvidence::get(
          request, {},
          evaluation::CompletedEvidence{
              {{evaluation::UncertaintyKind::ExactWithinModel,
                evaluation::PointObservation{
                    take(evaluation::DecimalValue::get(observation, 0))},
                {}}},
              {}},
          resolution, store));
  return take(evaluation::publishEvaluationEvidence(evidence, store));
}

struct Fixture final {
  ResolvedConfig config;
  ArtifactRootReference source;
  ArtifactRootReference preexistingEvidence;
  DsePlanGenerateInvocationRecords records;
  ArtifactRootReference selected;

  Fixture(ResolvedConfig config, ArtifactRootReference source,
          ArtifactRootReference preexistingEvidence,
          DsePlanGenerateInvocationRecords records,
          ArtifactRootReference selected)
      : config(std::move(config)), source(std::move(source)),
        preexistingEvidence(std::move(preexistingEvidence)),
        records(std::move(records)), selected(std::move(selected)) {}
};

Fixture makeFixture(const ArtifactStore &store, const BlobStore &blobs,
                    bool stopEarly = false) {
  const ComponentViewDigest digest =
      take(computeComponentViewDigest(configSchema, {0x01}));
  ResolvedConfig config = defaultResolvedConfig();
  config.dse.modelAuthorizations.clear();
  config.dse.evidenceObligationTemplates.clear();
  config.dse.objectiveCatalogs = {};
  config.dse.qualityGatePolicies.clear();
  ArtifactRootReference source = publish(store, sourceSchema, 0x11);
  config.dse.planNodes = {GeneratePlanNodeDefinition{
      generator.reference(), {ExactPlanArtifacts{{source}}}, {0x01}, digest}};
  const ArtifactIdentity storedConfig = take(store.put(
      ResolvedConfig::artifactSchema, canonicalResolvedConfigBytes(config)));
  if (storedConfig != resolvedConfigIdentity(config))
    fail("resolved config publication changed its identity");

  ResolvedDseConfigView view = take(projectResolvedDseConfigView(config));
  generationStopsEarly = stopEarly;
  DsePlanExecutionOutcome execution = take(executeDsePlan(view, store, blobs));
  generationStopsEarly = false;
  ArtifactRootReference selected = source;
  if (auto *completed = std::get_if<CompletedDsePlanExecution>(&execution)) {
    if (stopEarly)
      fail("early-stop test plan unexpectedly completed");
    selected = completed->resolve({0, 0}).front();
  } else {
    if (!stopEarly)
      fail("test plan did not complete");
    const auto &incomplete = std::get<IncompleteDsePlanExecution>(execution);
    const GenerateInvocationRecord *record =
        incomplete.incompleteGenerateInvocation();
    if (!record || record->outputBindings.size() != 1 ||
        record->outputBindings.front().artifacts.size() != 1)
      fail("early-stop test plan lost its retained candidate");
    selected = record->outputBindings.front().artifacts.front();
  }
  DsePlanGenerateInvocationRecords records =
      takeDsePlanGenerateInvocationRecords(std::move(execution));
  ArtifactRootReference evidence = publishEvidence(store, source, 0x21);
  return Fixture(std::move(config), std::move(source), std::move(evidence),
                 std::move(records), selected);
}

DseRunClosure makeClosure(const Fixture &fixture, const ArtifactStore &store,
                          llvm::ArrayRef<ArtifactRootReference> inputs) {
  return take(DseRunClosure::get(
      take(DseProducerSemanticBuildIdentity::get("loom.test.build.v1")), inputs,
      fixture.config, {fixture.preexistingEvidence}, store));
}

void testRunKeyAndRoundTrip(const ArtifactStore &store,
                            const BlobStore &blobs) {
  Fixture fixture = makeFixture(store, blobs);
  const std::array<ArtifactRootReference, 3> duplicateInputs = {
      fixture.selected, fixture.source, fixture.source};
  const std::array<ArtifactRootReference, 2> reversedInputs = {
      fixture.source, fixture.selected};
  DseRunClosure first = makeClosure(fixture, store, duplicateInputs);
  DseRunClosure second = makeClosure(fixture, store, reversedInputs);
  if (first.runKey() != second.runKey())
    fail("canonical input order changed the run key");

  const ArtifactRootReference secondEvidence =
      publishEvidence(store, fixture.source, 0x22);
  InvocationManifest manifest = take(InvocationManifest::get(
      std::move(first), 7, std::nullopt, fixture.config, fixture.records,
      InvocationCompletedSelection{
          {fixture.selected}, {secondEvidence, fixture.preexistingEvidence}},
      store));
  InvocationManifest reordered = take(InvocationManifest::get(
      makeClosure(fixture, store, reversedInputs), 7, std::nullopt,
      fixture.config, fixture.records,
      InvocationCompletedSelection{
          {fixture.selected}, {fixture.preexistingEvidence, secondEvidence}},
      store));
  if (manifest.canonicalBytes() != reordered.canonicalBytes())
    fail("set-valued authoring order changed canonical Manifest bytes");
  InvocationManifest adopted = take(adoptInvocationManifest(
      manifest.canonicalBytes(), fixture.config, store));
  if (adopted.canonicalBytes() != manifest.canonicalBytes() ||
      adopted.occurrence() != manifest.occurrence() ||
      adopted.generateRecords().size() != 1 ||
      !adopted.generateRecords().front().completed)
    fail("canonical manifest roundtrip changed the record");
  auto totals = [](const InvocationManifest &value) {
    std::pair<std::uint64_t, std::uint64_t> result{0, 0};
    for (const InvocationGenerateRecord &record : value.generateRecords())
      for (const CandidateGeneratorWorkUnitSummary &unit :
           record.workSummary.units) {
        result.first += unit.planned;
        result.second += unit.consumed;
      }
    return result;
  };
  if (totals(manifest) != std::pair<std::uint64_t, std::uint64_t>{1, 1} ||
      totals(adopted) != totals(manifest) ||
      totals(reordered) != totals(manifest))
    fail("Manifest work totals changed across canonical reordering or import");

  DseRunClosure changed = take(DseRunClosure::get(
      take(DseProducerSemanticBuildIdentity::get("loom.test.build.v2")),
      reversedInputs, fixture.config, {fixture.preexistingEvidence}, store));
  if (changed.runKey() == manifest.occurrence().runKey)
    fail("producer identity did not change the run key");
}

void testIncompleteWorkPreservation(const ArtifactStore &store,
                                    const BlobStore &blobs) {
  Fixture fixture = makeFixture(store, blobs, true);
  InvocationManifest manifest = take(InvocationManifest::get(
      makeClosure(fixture, store, {fixture.source}), 3, std::nullopt,
      fixture.config, fixture.records,
      InvocationIncomplete{
          0,
          CandidateGeneratorIncompleteReason::SemanticLimitReached,
          {},
          {fixture.selected},
          {}},
      store));
  InvocationManifest adopted = take(adoptInvocationManifest(
      manifest.canonicalBytes(), fixture.config, store));
  if (adopted.generateRecords().size() != 1 ||
      adopted.generateRecords().front().completed ||
      adopted.generateRecords().front().workSummary.units.size() != 1) {
    fail("incomplete Manifest lost its partial Generate record");
  }
  const CandidateGeneratorWorkUnitSummary &unit =
      adopted.generateRecords().front().workSummary.units.front();
  if (unit.planned != 4 || unit.consumed != 1)
    fail("incomplete Manifest changed planned or consumed logical work");
}

void testIncompleteReasonRoundTripCoverage(const ArtifactStore &store,
                                           const BlobStore &blobs) {
  Fixture fixture = makeFixture(store, blobs, true);
  for (std::uint32_t ordinal = 0; ordinal <= 5; ++ordinal) {
    const auto reason =
        static_cast<CandidateGeneratorIncompleteReason>(ordinal);
    InvocationManifest manifest = take(InvocationManifest::get(
        makeClosure(fixture, store, {fixture.source}), 3, std::nullopt,
        fixture.config, fixture.records,
        InvocationIncomplete{0, reason, {}, {fixture.selected}, {}}, store));
    InvocationManifest adopted = take(adoptInvocationManifest(
        manifest.canonicalBytes(), fixture.config, store));
    if (adopted.canonicalBytes() != manifest.canonicalBytes())
      fail("an incomplete reason changed across manifest reimport");
    const auto *adoptedIncomplete =
        std::get_if<InvocationIncomplete>(&adopted.outcome());
    if (!adoptedIncomplete)
      fail("an incomplete reason outcome changed form");
    const auto *candidateReason =
        std::get_if<CandidateGeneratorIncompleteReason>(
            &adoptedIncomplete->reason);
    if (!candidateReason || *candidateReason != reason)
      fail("an incomplete reason did not round-trip exactly");
  }
}

void testStrictFailures(const ArtifactStore &store, const BlobStore &blobs) {
  Fixture fixture = makeFixture(store, blobs);
  DseRunClosure closure = makeClosure(fixture, store, {fixture.source});
  InvocationManifest manifest = take(InvocationManifest::get(
      std::move(closure), 1, std::nullopt, fixture.config, fixture.records,
      InvocationCompletedSelection{{fixture.selected},
                                   {fixture.preexistingEvidence}},
      store));

  std::vector<std::uint8_t> trailing(manifest.canonicalBytes().begin(),
                                     manifest.canonicalBytes().end());
  trailing.push_back(0);
  auto trailingResult =
      adoptInvocationManifest(trailing, fixture.config, store);
  if (trailingResult)
    fail("manifest importer accepted trailing bytes");
  requireErrorContains(trailingResult.takeError(), "canonical");

  auto emptySelection = InvocationManifest::get(
      makeClosure(fixture, store, {fixture.source}), 2, std::nullopt,
      fixture.config, fixture.records,
      InvocationCompletedSelection{{}, {fixture.preexistingEvidence}}, store);
  if (emptySelection)
    fail("CompletedSelection accepted an empty selected set");
  requireErrorContains(emptySelection.takeError(), "nonempty");

  std::array<std::uint8_t, ArtifactIdentity::byteSize> missingBytes{};
  missingBytes.fill(0xee);
  ArtifactRootReference missing{
      sourceSchema.identity.str(), sourceSchema.version,
      take(ArtifactIdentity::fromBytes(missingBytes))};
  auto missingClosure = DseRunClosure::get(
      take(DseProducerSemanticBuildIdentity::get("loom.test.build.v1")),
      {missing}, fixture.config, {}, store);
  if (missingClosure)
    fail("run closure accepted an unreadable semantic input");
  requireErrorContains(missingClosure.takeError(), "artifact_store_missing");

  const ArtifactRootReference malformedEvidence =
      publish(store, evaluation::EvaluationEvidence::artifactSchema, 0x7f);
  auto malformedClosure = DseRunClosure::get(
      take(DseProducerSemanticBuildIdentity::get("loom.test.build.v1")),
      {fixture.source}, fixture.config, {malformedEvidence}, store);
  if (malformedClosure)
    fail("run closure accepted malformed EvaluationEvidence bytes");
  requireErrorContains(malformedClosure.takeError(), "JSON");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("expected one ArtifactStore directory argument");
  if (std::error_code error = llvm::sys::fs::create_directories(argv[1]))
    fail("unable to create ArtifactStore directory: " + error.message());
  registerOwner();
  ArtifactStore store(argv[1]);
  llvm::SmallString<128> blobPath(argv[1]);
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const BlobStore blobs(blobPath);
  testRunKeyAndRoundTrip(store, blobs);
  testIncompleteWorkPreservation(store, blobs);
  testIncompleteReasonRoundTripCoverage(store, blobs);
  testStrictFailures(store, blobs);
  return 0;
}
