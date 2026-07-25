#include "Evaluation/Evidence.h"
#include "Evaluation/OwnerError.h"

#include "Common/ArtifactFinalizer.h"
#include "Common/ArtifactStore.h"
#include "Common/ResolvedConfig.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::evaluation;

namespace {

[[noreturn]] void fail(const char *test, const std::string &message) {
  std::cerr << test << ": " << message << '\n';
  std::exit(1);
}

void require(const char *test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T>
T takeExpected(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectErrorContains(const char *test, llvm::Expected<T> value,
                         llvm::StringRef expected) {
  if (value)
    fail(test, "expected an error");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected),
          "unexpected error: " + message);
}

template <typename T>
void expectSimulationExecutionOwnerUnavailable(const char *test,
                                                llvm::Expected<T> value) {
  if (value)
    fail(test, "expected the SimulationExecution owner to be unavailable");
  bool matched = false;
  llvm::Error remaining = llvm::handleErrors(
      value.takeError(),
      [&](const EvaluationOwnerUnavailableError &failure) -> llvm::Error {
        matched = failure.ownerIdentity() == "loom.simulation_execution" &&
                  failure.ownerVersion() == SchemaVersion{1, 0};
        return llvm::Error::success();
      });
  if (remaining)
    fail(test, llvm::toString(std::move(remaining)));
  require(test, matched, "wrong Evaluation owner-unavailable failure");
}

class TemporaryDirectory {
public:
  explicit TemporaryDirectory(const char *test) {
    if (std::error_code error =
            llvm::sys::fs::createUniqueDirectory("loom-evidence", path_))
      fail(test, error.message());
  }

  ~TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::remove_directories(path_))
      std::cerr << "temporary directory cleanup failed: " << error.message()
                << '\n';
  }

  llvm::StringRef path() const { return path_; }

private:
  llvm::SmallString<128> path_;
};

CanonicalSemanticBytes subjectBytes() {
  return CanonicalSemanticBytes(std::vector<std::uint8_t>{0x51});
}

constexpr ArtifactSchemaDescriptor subjectSchema{"loom.test.evidence.subject",
                                                  {1, 0}};
constexpr EvaluationCaseKind caseKind{41};
constexpr FindingKind findingKind{42};
constexpr EvaluationModelKind modelKind{43};
constexpr FindingKind terminalFindingKind{44};

struct TestFindingOccurrence {
  std::uint8_t value;
};

EvaluationCaseSignatureRef signatureRef() {
  return llvm::cantFail(
      EvaluationCaseSignatureRef::get(evaluationSchemaVersion(), caseKind));
}

CaseSubjectRoleRef subjectRole() { return CaseSubjectRoleRef(0); }

ArtifactRootReference subjectArtifact() {
  return ArtifactRootReference{subjectSchema.identity.str(),
                               subjectSchema.version,
                               finalizeArtifactIdentity(subjectSchema,
                                                        subjectBytes())};
}

void ensureSubjectStored(const char *test, const ArtifactStore &store) {
  const ArtifactIdentity identity =
      takeExpected(test, store.put(subjectSchema, subjectBytes()));
  require(test, identity == subjectArtifact().artifact,
          "stored subject identity changed");
}

const ArtifactSchemaDescriptor *const acceptedSubjectSchemas[] = {
    &subjectSchema};

const CaseSubjectRoleDescriptor subjectRoles[] = {
    {subjectRole(), "subject", SubjectRoleCardinality::ExactlyOne,
     acceptedSubjectSchemas, nullptr},
};

const ScopeFormDescriptor findingScopeForms[] = {
    {ScopeFormRef(0), "the entire exact case", {}, WholeExactCaseScope{},
     nullptr},
};

const EvaluationCaseSignatureDescriptor signatureDescriptor{
    caseKind,
    "evidence_test_case",
    "One exact subject used by persistent Evidence tests.",
    subjectRoles,
    ArtifactRequirement::Forbidden,
    {},
    ArtifactRequirement::Forbidden,
    {},
    nullptr,
    AbstractCaseCycle{},
    {}};

llvm::Expected<std::vector<std::uint8_t>>
encodeOccurrence(const OwnerValue &occurrence) {
  const auto *typed = occurrence.getIf<TestFindingOccurrence>();
  if (!typed || typed->value != 0x2a)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unexpected finding occurrence value");
  return std::vector<std::uint8_t>{typed->value};
}

llvm::Expected<OwnerValue>
decodeOccurrence(llvm::ArrayRef<std::uint8_t> canonicalPayload) {
  if (canonicalPayload.empty() || canonicalPayload.front() != 0x2a)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unexpected finding occurrence payload");
  return OwnerValue::get(TestFindingOccurrence{canonicalPayload.front()});
}

llvm::Error validateOccurrence(const OwnerValue &occurrence,
                               const FindingOccurrenceContext &context) {
  if (!occurrence.getIf<TestFindingOccurrence>())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "finding occurrence has the wrong type");
  if (context.findingRequestOrdinal() != FindingRequestOrdinal(0))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "finding occurrence has a foreign ordinal");
  if (!context.resolveOutput(ModelOutputSlotRef(0), 0))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "finding occurrence output is unresolved");
  return llvm::Error::success();
}

const FindingDescriptor findingDescriptor{
    findingKind,
    "evidence_test_finding",
    "One typed finding used to verify persistent result totality.",
    findingScopeForms,
    {},
    {{"loom.test.evidence.finding_occurrence", {1, 0}}, &encodeOccurrence,
     &decodeOccurrence, &validateOccurrence},
    std::nullopt};

const FindingDescriptor terminalFindingDescriptor{
    terminalFindingKind,
    "evidence_test_terminal_finding",
    "One terminal finding whose occurrence requires SimulationExecution.",
    findingScopeForms,
    {},
    {{"loom.test.evidence.terminal_occurrence", {1, 0}}, &encodeOccurrence,
     &decodeOccurrence, &validateOccurrence},
    FindingPayloadSchemaDescriptor{"loom.simulation_execution", {1, 0}}};

const ScopeFormRef supportedFindingForms[] = {ScopeFormRef(0)};
const ScopeFormRef supportedMetricForms[] = {ScopeFormRef(0)};
const MetricCapability metricCapabilities[] = {
    {MetricKind::CycleCount, supportedMetricForms, allObservationFormsMask()},
};
const FindingCapability findingCapabilities[] = {
    {findingKind, supportedFindingForms, allFindingResultFormsMask()},
    {terminalFindingKind, supportedFindingForms, allFindingResultFormsMask()},
};

struct EmptyModelConfigView {};

llvm::ArrayRef<std::uint8_t> modelConfigSchemaBytes() {
  static constexpr llvm::StringLiteral descriptor =
      "loom.test.evidence.model_config.1.0";
  return {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
          descriptor.size()};
}

llvm::Expected<OwnerValue> projectEmptyModelConfig(const ResolvedConfig &) {
  return OwnerValue::get(EmptyModelConfigView{});
}

llvm::Expected<std::vector<std::uint8_t>>
encodeEmptyModelConfig(const OwnerValue &view) {
  if (!view.getIf<EmptyModelConfigView>())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "model config has the wrong type");
  return std::vector<std::uint8_t>{};
}

llvm::Expected<OwnerValue>
adoptEmptyModelConfig(llvm::ArrayRef<std::uint8_t> canonicalBytes,
                      const ComponentViewDigest &) {
  if (!canonicalBytes.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "model config must be empty");
  return OwnerValue::get(EmptyModelConfigView{});
}

const ResolvedModelConfigViewContract modelConfigView{
    modelConfigSchemaBytes(), &projectEmptyModelConfig,
    &encodeEmptyModelConfig, &adoptEmptyModelConfig};

const ModelOutputSlotDescriptor outputSlots[] = {
    {ModelOutputSlotRef(0),
     "result",
     &subjectSchema,
     {ArtifactCollectionCardinality::ExactlyOne,
      ArtifactCollectionCardinality::Forbidden,
      ArtifactCollectionCardinality::Forbidden,
      ArtifactCollectionCardinality::ZeroOrOne}},
};

const EvaluationModelDescriptor modelDescriptor{
    modelKind,
    "evidence_test_model",
    "loom.test.evidence.model",
    signatureRef(),
    {},
    metricCapabilities,
    findingCapabilities,
    {},
    outputSlots,
    modelConfigView,
    {},
    EvaluationExecutionMethod::Analytic,
    {},
    DeterminismContract::Deterministic,
    {}};

EvaluationSubjectBindings subjectBindings(const char *test) {
  return takeExpected(test, EvaluationSubjectBindings::get(
                                {{subjectRole(), {subjectArtifact()}}}));
}

CaseArtifactResolution caseResolution(const char *test) {
  return takeExpected(
      test, CaseArtifactResolution::get({{subjectArtifact(), {}}}));
}

EvaluationRequest requestForFinding(const char *test,
                                    const ArtifactStore &store,
                                    FindingKind kind) {
  ensureSubjectStored(test, store);
  const CaseArtifactResolution resolution = caseResolution(test);
  const EvaluationCase evaluationCase = takeExpected(
      test, EvaluationCase::get(signatureRef(), subjectBindings(test),
                                std::nullopt, std::nullopt, {}, resolution,
                                store));
  const FindingRequest finding = takeExpected(
      test, FindingRequest::get(
                FindingQuery{kind, EvaluationScope{ScopeFormRef(0), {}}},
                {}, evaluationCase, resolution, store));
  const ResolvedModelBinding binding = takeExpected(
      test, ResolvedModelBinding::project(modelDescriptor.reference(), {},
                                          defaultResolvedConfig()));
  return takeExpected(test,
                      EvaluationRequest::get(evaluationCase, {}, {finding},
                                             binding, 0, resolution, store));
}

EvaluationRequest request(const char *test, const ArtifactStore &store) {
  return requestForFinding(test, store, findingKind);
}

EvaluationRequest metricAndFindingRequest(const char *test,
                                          const ArtifactStore &store) {
  ensureSubjectStored(test, store);
  const CaseArtifactResolution resolution = caseResolution(test);
  const EvaluationCase evaluationCase = takeExpected(
      test, EvaluationCase::get(signatureRef(), subjectBindings(test),
                                std::nullopt, std::nullopt, {}, resolution,
                                store));
  const MetricRequest metric = takeExpected(
      test, MetricRequest::get(
                MetricQuery{MetricKind::CycleCount,
                            EvaluationScope{ScopeFormRef(0), {}}},
                {}, evaluationCase, resolution, store));
  const FindingRequest finding = takeExpected(
      test, FindingRequest::get(
                FindingQuery{findingKind,
                             EvaluationScope{ScopeFormRef(0), {}}},
                {}, evaluationCase, resolution, store));
  const ResolvedModelBinding binding = takeExpected(
      test, ResolvedModelBinding::project(modelDescriptor.reference(), {},
                                          defaultResolvedConfig()));
  return takeExpected(test,
                      EvaluationRequest::get(evaluationCase, {metric},
                                             {finding}, binding, 0, resolution,
                                             store));
}

ArtifactRootReference putArtifact(const char *test, const ArtifactStore &store,
                                  std::uint8_t byte) {
  const ArtifactIdentity identity = takeExpected(
      test, store.put(subjectSchema,
                      CanonicalSemanticBytes(std::vector<std::uint8_t>{byte})));
  return ArtifactRootReference{subjectSchema.identity.str(),
                               subjectSchema.version, identity};
}

CompletedEvidence completedWith(FindingResult finding) {
  return CompletedEvidence{{}, {std::move(finding)}};
}

FindingResult absentFinding() {
  return FindingResult{AbsentFinding{}};
}

void completedEvidenceIsTotalAndCanonical() {
  TemporaryDirectory directory(__func__);
  const ArtifactStore store(directory.path());
  const EvaluationRequest evaluationRequest = request(__func__, store);
  takeExpected(__func__, publishEvaluationRequest(evaluationRequest, store));
  const ArtifactRootReference output = putArtifact(__func__, store, 0x61);

  const EvaluationEvidence evidence = takeExpected(
      __func__, EvaluationEvidence::get(
                    evaluationRequest,
                    {{ModelOutputSlotRef(0), {output}}},
                    completedWith(absentFinding()),
                    caseResolution(__func__), store));
  require(__func__, evidence.outcomeKind() == EvidenceOutcomeKind::Completed,
          "completed Evidence changed outcome kind");
  const auto *completed = std::get_if<CompletedEvidence>(&evidence.outcome());
  require(__func__, completed && completed->metricResults.empty() &&
                        completed->findingResults.size() == 1,
          "completed Evidence lost total result tables");

  const std::string canonical = serializeEvaluationEvidence(evidence);
  require(__func__, !llvm::StringRef(canonical).contains("severity") &&
                        !llvm::StringRef(canonical).contains("case_key") &&
                        !llvm::StringRef(canonical).contains("model_key") &&
                        !llvm::StringRef(canonical).contains(
                            "metric_request_ordinal") &&
                        !llvm::StringRef(canonical).contains(
                            "finding_request_ordinal"),
          "Evidence serialized a forbidden authority");
  const EvaluationEvidence parsed = takeExpected(
      __func__, parseEvaluationEvidence(canonical, caseResolution(__func__),
                                        store));
  require(__func__, serializeEvaluationEvidence(parsed) == canonical,
          "Evidence canonical roundtrip changed bytes");
  require(__func__, evaluationEvidenceIdentity(parsed) ==
                        evaluationEvidenceIdentity(evidence),
          "Evidence canonical roundtrip changed identity");

  const ArtifactRootReference published =
      takeExpected(__func__, publishEvaluationEvidence(evidence, store));
  const EvaluationEvidence imported = takeExpected(
      __func__, importEvaluationEvidence(published, caseResolution(__func__),
                                         store));
  require(__func__, serializeEvaluationEvidence(imported) == canonical,
          "ArtifactStore import changed Evidence semantics");
}

void completedEvidenceRejectsGapsAndInvalidFindings() {
  TemporaryDirectory directory(__func__);
  const ArtifactStore store(directory.path());
  const EvaluationRequest evaluationRequest = request(__func__, store);
  takeExpected(__func__, publishEvaluationRequest(evaluationRequest, store));
  const ArtifactRootReference output = putArtifact(__func__, store, 0x62);

  expectErrorContains(
      __func__,
      EvaluationEvidence::get(evaluationRequest, {},
                              completedWith(absentFinding()),
                              caseResolution(__func__), store),
      "not total over descriptor output slots");
  expectErrorContains(
      __func__, EvaluationEvidence::get(
                    evaluationRequest, {{ModelOutputSlotRef(0), {}}},
                    completedWith(absentFinding()),
                    caseResolution(__func__), store),
      "declared cardinality");
  expectErrorContains(
      __func__, EvaluationEvidence::get(
                    evaluationRequest,
                    {{ModelOutputSlotRef(0), {output}}}, CompletedEvidence{},
                    caseResolution(__func__), store),
      "not total over finding requests");
  expectErrorContains(
      __func__,
      EvaluationEvidence::get(
          evaluationRequest, {{ModelOutputSlotRef(0), {output}}},
          CompletedEvidence{{}, {absentFinding(), absentFinding()}},
          caseResolution(__func__), store),
      "not total over finding requests");

  MetricResult unsolicited{UncertaintyKind::ExactWithinModel,
                           PointObservation{IntegerValue(7)}, {}};
  expectErrorContains(
      __func__, EvaluationEvidence::get(
                    evaluationRequest,
                    {{ModelOutputSlotRef(0), {output}}},
                    CompletedEvidence{{unsolicited}, {absentFinding()}},
                    caseResolution(__func__), store),
      "not total over metric requests");

  FindingResult emptyPresent{PresentFinding{{}}};
  expectErrorContains(
      __func__, EvaluationEvidence::get(
                    evaluationRequest,
                    {{ModelOutputSlotRef(0), {output}}},
                    completedWith(std::move(emptyPresent)),
                    caseResolution(__func__), store),
      "requires at least one occurrence");
  FindingResult invalidPresent{PresentFinding{{
      FindingOccurrence::get(TestFindingOccurrence{0xff})}}};
  expectErrorContains(
      __func__, EvaluationEvidence::get(
                    evaluationRequest,
                    {{ModelOutputSlotRef(0), {output}}},
                    completedWith(std::move(invalidPresent)),
                    caseResolution(__func__), store),
      "unexpected finding occurrence value");

  FindingResult validPresent{PresentFinding{{
      FindingOccurrence::get(TestFindingOccurrence{0x2a})}}};
  const EvaluationEvidence present = takeExpected(
      __func__, EvaluationEvidence::get(
                    evaluationRequest,
                    {{ModelOutputSlotRef(0), {output}}},
                    completedWith(std::move(validPresent)),
                    caseResolution(__func__), store));
  const std::string presentJson = serializeEvaluationEvidence(present);
  require(__func__, llvm::StringRef(presentJson).contains("\"occurrences\":[\"2a\"]") &&
                        !llvm::StringRef(presentJson).contains(
                            "\"kind\":\"inline_payload\""),
          "Evidence duplicated the Finding owner occurrence wire");
  const EvaluationEvidence parsedPresent = takeExpected(
      __func__, parseEvaluationEvidence(presentJson, caseResolution(__func__),
                                        store));
  const auto &parsedCompleted =
      std::get<CompletedEvidence>(parsedPresent.outcome());
  const auto &parsedOccurrences =
      std::get<PresentFinding>(parsedCompleted.findingResults[0].result)
          .occurrences;
  const TestFindingOccurrence *parsedOccurrence =
      parsedOccurrences[0].getIf<TestFindingOccurrence>();
  require(__func__, parsedOccurrence && parsedOccurrence->value == 0x2a,
          "Evidence import did not retain the owner-typed occurrence");

  std::string noncanonicalJson = presentJson;
  const std::size_t occurrencePosition = noncanonicalJson.find("\"2a\"");
  require(__func__, occurrencePosition != std::string::npos,
          "could not locate the finding occurrence payload");
  noncanonicalJson.replace(occurrencePosition, 4, "\"2a00\"");
  expectErrorContains(
      __func__, parseEvaluationEvidence(noncanonicalJson,
                                        caseResolution(__func__), store),
      "not canonical");
  FindingResult notApplicable{
      NotApplicableFinding{NotApplicableReason::UndefinedForSubject}};
  takeExpected(__func__, EvaluationEvidence::get(
                             evaluationRequest,
                             {{ModelOutputSlotRef(0), {output}}},
                             completedWith(std::move(notApplicable)),
                             caseResolution(__func__), store));
}

void completedMetricResultsUseRequestOwnedSemantics() {
  TemporaryDirectory directory(__func__);
  const ArtifactStore store(directory.path());
  const EvaluationRequest evaluationRequest =
      metricAndFindingRequest(__func__, store);
  takeExpected(__func__, publishEvaluationRequest(evaluationRequest, store));
  const ArtifactRootReference output = putArtifact(__func__, store, 0x66);
  const MetricResult metric{UncertaintyKind::ExactWithinModel,
                            PointObservation{IntegerValue(7)}, {}};
  const EvaluationEvidence evidence = takeExpected(
      __func__, EvaluationEvidence::get(
                    evaluationRequest,
                    {{ModelOutputSlotRef(0), {output}}},
                    CompletedEvidence{{metric}, {absentFinding()}},
                    caseResolution(__func__), store));
  const std::string canonical = serializeEvaluationEvidence(evidence);
  require(__func__, !llvm::StringRef(canonical).contains("cycle_count") &&
                        !llvm::StringRef(canonical).contains("scope") &&
                        !llvm::StringRef(canonical).contains("unit"),
          "MetricResult copied Request or registry authority");
  require(__func__,
          !llvm::StringRef(canonical).contains("evaluation.metric"),
          "Evidence revived the standalone metric wire identity");

  // The nested point value round-trips through the Evidence artifact store.
  const ArtifactRootReference published =
      takeExpected(__func__, publishEvaluationEvidence(evidence, store));
  const EvaluationEvidence imported = takeExpected(
      __func__, importEvaluationEvidence(published, caseResolution(__func__),
                                         store));
  require(__func__, serializeEvaluationEvidence(imported) == canonical,
          "nested metric value encoding did not round-trip");

  const MetricResult wrongValue{
      UncertaintyKind::ExactWithinModel,
      PointObservation{takeExpected(__func__, DecimalValue::get(7, 0))}, {}};
  expectErrorContains(
      __func__, EvaluationEvidence::get(
                    evaluationRequest,
                    {{ModelOutputSlotRef(0), {output}}},
                    CompletedEvidence{{wrongValue}, {absentFinding()}},
                    caseResolution(__func__), store),
      "cycle_count requires integer values");
}

void noncompletedEvidenceHasNoResultTables() {
  TemporaryDirectory directory(__func__);
  const ArtifactStore store(directory.path());
  const EvaluationRequest evaluationRequest = request(__func__, store);
  takeExpected(__func__, publishEvaluationRequest(evaluationRequest, store));
  const ArtifactRootReference output = putArtifact(__func__, store, 0x63);

  const EvaluationEvidence unsupported = takeExpected(
      __func__, EvaluationEvidence::get(
                    evaluationRequest,
                    {{ModelOutputSlotRef(0), {}}},
                    UnsupportedEvidence{
                        OutcomeReason::RuntimeCapabilityUnavailable},
                    caseResolution(__func__), store));
  const std::string canonical = serializeEvaluationEvidence(unsupported);
  std::string withResults = canonical;
  const std::size_t position = withResults.rfind("}}");
  require(__func__, position != std::string::npos,
          "could not locate the outcome boundary");
  withResults.insert(position, ",\"metric_results\":[]");
  expectErrorContains(
      __func__, parseEvaluationEvidence(withResults, caseResolution(__func__),
                                        store),
      "unknown field 'metric_results'");

  expectErrorContains(
      __func__, EvaluationEvidence::get(
                    evaluationRequest,
                    {{ModelOutputSlotRef(0), {output}}},
                    UnsupportedEvidence{
                        OutcomeReason::RuntimeCapabilityUnavailable},
                    caseResolution(__func__), store),
      "declared cardinality");
  takeExpected(__func__, EvaluationEvidence::get(
                             evaluationRequest,
                             {{ModelOutputSlotRef(0), {output}}},
                             CancelledOrTimeoutEvidence{
                                 OutcomeReason::ExecutionLimitReached},
                             caseResolution(__func__), store));
}

void detailedBundleFieldIsUnknown() {
  TemporaryDirectory directory(__func__);
  const ArtifactStore store(directory.path());
  const EvaluationRequest evaluationRequest = request(__func__, store);
  takeExpected(__func__, publishEvaluationRequest(evaluationRequest, store));
  const EvaluationEvidence evidence = takeExpected(
      __func__, EvaluationEvidence::get(
                    evaluationRequest, {{ModelOutputSlotRef(0), {}}},
                    UnsupportedEvidence{
                        OutcomeReason::RuntimeCapabilityUnavailable},
                    caseResolution(__func__), store));
  std::string withDetailedBundles = serializeEvaluationEvidence(evidence);
  require(__func__,
          !llvm::StringRef(withDetailedBundles).contains(
              "detailed_bundle_refs"),
          "canonical Evidence serialized detailed_bundle_refs");
  const std::string field = ",\"detailed_bundle_refs\":[]";
  const std::size_t rootEnd = withDetailedBundles.rfind('}');
  require(__func__, rootEnd != std::string::npos,
          "could not locate the Evidence root boundary");
  withDetailedBundles.insert(rootEnd, field);
  expectErrorContains(
      __func__, parseEvaluationEvidence(withDetailedBundles,
                                        caseResolution(__func__), store),
      "unknown field 'detailed_bundle_refs'");
}

void terminalFindingFailsClosedWithoutExecutionOwner() {
  TemporaryDirectory directory(__func__);
  const ArtifactStore store(directory.path());
  const EvaluationRequest evaluationRequest =
      requestForFinding(__func__, store, terminalFindingKind);
  takeExpected(__func__, publishEvaluationRequest(evaluationRequest, store));
  const ArtifactRootReference output = putArtifact(__func__, store, 0x66);
  FindingResult present{PresentFinding{{
      FindingOccurrence::get(TestFindingOccurrence{0x2a})}}};
  expectSimulationExecutionOwnerUnavailable(
      __func__, EvaluationEvidence::get(
                    evaluationRequest,
                    {{ModelOutputSlotRef(0), {output}}},
                    completedWith(std::move(present)),
                    caseResolution(__func__), store));
}

} // namespace

int main() {
  if (llvm::Error error = registerEvaluationCaseSignature(signatureDescriptor))
    fail("registration", llvm::toString(std::move(error)));
  if (llvm::Error error = registerFindingDescriptor(findingDescriptor))
    fail("registration", llvm::toString(std::move(error)));
  if (llvm::Error error = registerFindingDescriptor(terminalFindingDescriptor))
    fail("registration", llvm::toString(std::move(error)));
  if (llvm::Error error =
          registerEvaluationModelDescriptor(modelDescriptor))
    fail("registration", llvm::toString(std::move(error)));
  completedEvidenceIsTotalAndCanonical();
  completedEvidenceRejectsGapsAndInvalidFindings();
  completedMetricResultsUseRequestOwnedSemantics();
  noncompletedEvidenceHasNoResultTables();
  detailedBundleFieldIsUnknown();
  terminalFindingFailsClosedWithoutExecutionOwner();
  return 0;
}
