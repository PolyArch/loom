#include "Evaluation/Case.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/Finding.h"
#include "Evaluation/ModelDescriptor.h"
#include "Evaluation/OwnerError.h"
#include "Evaluation/Request.h"

#include "Common/Artifact.h"
#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "ArtifactLocalReferenceRegistry.h"
#include "ImplementationPlatform/TechnologyCorner.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"

#include <algorithm>
#include <array>
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

class TemporaryDirectory {
public:
  explicit TemporaryDirectory(const char *test) {
    if (std::error_code error =
            llvm::sys::fs::createUniqueDirectory("loom-evaluation", path_))
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

void expectErrorContains(const char *test, llvm::Error error,
                         llvm::StringRef expected) {
  if (!error)
    fail(test, "expected an error");
  const std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains(expected),
          "unexpected error: " + message);
}

template <typename T>
void expectErrorContains(const char *test, llvm::Expected<T> value,
                         llvm::StringRef expected) {
  if (value)
    fail(test, "expected an error");
  expectErrorContains(test, value.takeError(), expected);
}

template <typename T>
void expectError(const char *test, llvm::Expected<T> value) {
  if (value)
    fail(test, "expected an error");
  llvm::consumeError(value.takeError());
}

void expectOwnerCodecUnavailable(const char *test, llvm::Error error) {
  if (!error)
    fail(test, "expected an unavailable owner codec");
  bool matched = false;
  llvm::Error remaining = llvm::handleErrors(
      std::move(error),
      [&](const ArtifactLocalReferenceError &failure) -> llvm::Error {
        matched = failure.kind() ==
                  ArtifactLocalReferenceErrorKind::OwnerCodecUnavailable;
        return llvm::Error::success();
      });
  if (remaining)
    fail(test, llvm::toString(std::move(remaining)));
  require(test, matched, "wrong local-reference capability failure");
}

void expectSimulationExecutionOwnerUnavailable(const char *test,
                                                llvm::Error error) {
  if (!error)
    fail(test, "expected the SimulationExecution owner to be unavailable");
  bool matched = false;
  llvm::Error remaining = llvm::handleErrors(
      std::move(error),
      [&](const EvaluationOwnerUnavailableError &failure) -> llvm::Error {
        matched = failure.ownerIdentity() == "loom.simulation_execution" &&
                  failure.ownerVersion() == SchemaVersion{1, 0};
        return llvm::Error::success();
      });
  if (remaining)
    fail(test, llvm::toString(std::move(remaining)));
  require(test, matched, "wrong Evaluation owner-unavailable failure");
}

template <typename T>
void expectSimulationExecutionOwnerUnavailable(const char *test,
                                                llvm::Expected<T> value) {
  if (value)
    fail(test, "expected the SimulationExecution owner to be unavailable");
  expectSimulationExecutionOwnerUnavailable(test, value.takeError());
}

template <typename T>
void expectOwnerCodecUnavailable(const char *test, llvm::Expected<T> value) {
  if (value)
    fail(test, "expected an unavailable owner codec");
  expectOwnerCodecUnavailable(test, value.takeError());
}

ArtifactIdentity testArtifact(std::initializer_list<std::uint8_t> prefix) {
  ArtifactIdentity::Storage bytes{};
  require(__func__, prefix.size() <= bytes.size(),
          "test identity prefix is too long");
  std::copy(prefix.begin(), prefix.end(), bytes.begin());
  return takeExpected(__func__, ArtifactIdentity::fromBytes(bytes));
}

DecimalValue decimal(const char *test, std::int64_t coefficient,
                     std::int64_t exponent) {
  return takeExpected(test, DecimalValue::get(coefficient, exponent));
}

ExactRatio ratio(const char *test, std::uint64_t numerator,
                 std::uint64_t denominator) {
  return takeExpected(test, ExactRatio::get(numerator, denominator));
}

constexpr ArtifactSchemaDescriptor subjectSchema{"loom.test.subject", {1, 0}};
constexpr EvaluationCaseKind testCaseKind{17};
constexpr EvaluationCaseKind noCycleCaseKind{18};
constexpr EvaluationCaseKind clockAliasCaseKind{19};
constexpr EvaluationCaseKind failingCycleCaseKind{20};
constexpr std::uint32_t testLocalKind = 9;
constexpr std::uint32_t clockLocalKind = 10;

EvaluationCaseSignatureRef testSignatureRef() {
  return llvm::cantFail(
      EvaluationCaseSignatureRef::get(evaluationSchemaVersion(), testCaseKind));
}

EvaluationCaseSignatureRef noCycleSignatureRef() {
  return llvm::cantFail(EvaluationCaseSignatureRef::get(
      evaluationSchemaVersion(), noCycleCaseKind));
}

EvaluationCaseSignatureRef clockAliasSignatureRef() {
  return llvm::cantFail(EvaluationCaseSignatureRef::get(
      evaluationSchemaVersion(), clockAliasCaseKind));
}

EvaluationCaseSignatureRef failingCycleSignatureRef() {
  return llvm::cantFail(EvaluationCaseSignatureRef::get(
      evaluationSchemaVersion(), failingCycleCaseKind));
}

CaseSubjectRoleRef subjectRole() { return CaseSubjectRoleRef(0); }

ArtifactRootReference root(const ArtifactSchemaDescriptor &schema,
                           ArtifactIdentity artifact) {
  return ArtifactRootReference{schema.identity.str(), schema.version,
                               std::move(artifact)};
}

ArtifactRootReference subjectArtifact() {
  return root(subjectSchema, testArtifact({0x11}));
}

ArtifactRootReference dependencyArtifact() {
  return root(subjectSchema, testArtifact({0x22}));
}

ArtifactRootReference foreignArtifact() {
  return root(subjectSchema, testArtifact({0x33}));
}

ArtifactRootReference platformArtifact() {
  return root(platform::implementationPlatformSchema, testArtifact({0x44}));
}

const ArtifactSchemaDescriptor *const acceptedSubjectSchemas[] = {
    &subjectSchema};

const CaseSubjectRoleDescriptor subjectRoles[] = {
    {subjectRole(), "subject", SubjectRoleCardinality::ExactlyOne,
     acceptedSubjectSchemas, nullptr},
};

SubjectReferenceType subjectRootType() {
  return SubjectReferenceType{ArtifactRootType{subjectSchema}};
}

SubjectReferenceType subjectLocalType() {
  return SubjectReferenceType{
      ArtifactLocalType{{subjectSchema, testLocalKind}}};
}

SubjectReferenceType clockLocalType() {
  return SubjectReferenceType{
      ArtifactLocalType{{subjectSchema, clockLocalKind}}};
}

llvm::Expected<SubjectTargetRef>
failCycleResolution(const EvaluationCase &, const CaseArtifactResolution &,
                    const ArtifactStore &) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "test reference-cycle resolution failure");
}

llvm::Error validateClockLocalPayload(llvm::ArrayRef<std::uint8_t> payload) {
  if (payload.size() != 1)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "test clock local target payload must be one byte");
  return llvm::Error::success();
}

llvm::Error
validateClockLocalTarget(const CanonicalSemanticBytes &,
                         const EncodedArtifactLocalReference &reference) {
  if (reference.ownerLocalKind != clockLocalKind)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "wrong test clock local kind");
  return llvm::Error::success();
}

const std::vector<ConditionApplicabilityPattern> baseConditionPatterns = {
    {EvaluationConditionKind::ProcessCorner,
     {testSignatureRef(), {{subjectRole(), subjectRootType()}}}},
    {EvaluationConditionKind::RequiredClockPeriod,
     {testSignatureRef(), {{subjectRole(), subjectRootType()}}}},
    {EvaluationConditionKind::RelativeClockSchedule,
     {testSignatureRef(),
      {{subjectRole(), subjectRootType()},
       {subjectRole(), subjectRootType()}}}},
    {EvaluationConditionKind::ActivityBinding,
     {testSignatureRef(), {{subjectRole(), subjectRootType()}}}},
};

const EvaluationCaseSignatureDescriptor testSignature{
    testCaseKind,
    "test_subject_case",
    "One exact test subject and its dependency closure.",
    subjectRoles,
    ArtifactRequirement::Forbidden,
    {},
    ArtifactRequirement::Forbidden,
    {},
    nullptr,
    AbstractCaseCycle{},
    baseConditionPatterns};

const EvaluationCaseSignatureDescriptor noCycleSignature{
    noCycleCaseKind,
    "test_subject_without_cycle",
    "One exact test subject without a whole-case reference cycle.",
    subjectRoles,
    ArtifactRequirement::Forbidden,
    {},
    ArtifactRequirement::Forbidden,
    {},
    nullptr,
    AbsentReferenceCycle{},
    {}};

CaseSubjectRoleRef clockReferenceRole() { return CaseSubjectRoleRef(0); }
CaseSubjectRoleRef clockDependentRole() { return CaseSubjectRoleRef(1); }

const CaseSubjectRoleDescriptor clockAliasRoles[] = {
    {clockReferenceRole(), "clock_reference_anchor",
     SubjectRoleCardinality::ExactlyOne, acceptedSubjectSchemas, nullptr},
    {clockDependentRole(), "clock_dependent_anchor",
     SubjectRoleCardinality::ExactlyOne, acceptedSubjectSchemas, nullptr},
};

const std::vector<ConditionApplicabilityPattern> clockAliasBasePatterns = {
    {EvaluationConditionKind::RelativeClockSchedule,
     {clockAliasSignatureRef(),
      {{clockReferenceRole(), clockLocalType()},
       {clockDependentRole(), clockLocalType()}}}},
};

const EvaluationCaseSignatureDescriptor clockAliasSignature{
    clockAliasCaseKind,
    "test_clock_alias_case",
    "Two clock anchors whose closures reach one shared clock domain.",
    clockAliasRoles,
    ArtifactRequirement::Forbidden,
    {},
    ArtifactRequirement::Forbidden,
    {},
    nullptr,
    AbstractCaseCycle{},
    clockAliasBasePatterns};

const EvaluationCaseSignatureDescriptor failingCycleSignature{
    failingCycleCaseKind,
    "test_failing_cycle_case",
    "One exact subject-cycle resolver used to prove basis ownership.",
    subjectRoles,
    ArtifactRequirement::Forbidden,
    {},
    ArtifactRequirement::Forbidden,
    {},
    nullptr,
    ExactSubjectCycle{subjectRootType(), &failCycleResolution},
    {}};

EvaluationSubjectBindings bindings(const char *test) {
  return takeExpected(test, EvaluationSubjectBindings::get(
                                {{subjectRole(), {subjectArtifact()}}}));
}

CaseArtifactResolution resolution(const char *test) {
  return takeExpected(test, CaseArtifactResolution::get(
                                {{subjectArtifact(),
                                  {dependencyArtifact(), platformArtifact()}},
                                 {dependencyArtifact(), {}},
                                 {platformArtifact(), {}}}));
}

const ArtifactStore &artifactStore() {
  static const ArtifactStore store(
      "/nonexistent/loom-evaluation-case-test-artifacts");
  return store;
}

EvaluationCase
evaluationCase(const char *test, llvm::ArrayRef<EvaluationCondition> conditions,
               EvaluationCaseSignatureRef signature = testSignatureRef()) {
  return takeExpected(test, EvaluationCase::get(signature, bindings(test),
                                                std::nullopt, std::nullopt,
                                                conditions, resolution(test),
                                                artifactStore()));
}

SubjectTargetRef rootTarget(ArtifactRootReference target) {
  return SubjectTargetRef{subjectRole(), subjectArtifact(), std::move(target)};
}

ArtifactRootReference storeSubject(const char *test, const ArtifactStore &store,
                                   std::uint8_t byte) {
  const ArtifactIdentity identity = takeExpected(
      test, store.put(subjectSchema,
                      CanonicalSemanticBytes(std::vector<std::uint8_t>{byte})));
  return root(subjectSchema, identity);
}

SubjectTargetRef localTarget() {
  return SubjectTargetRef{
      subjectRole(), subjectArtifact(),
      EncodedArtifactLocalReference{subjectArtifact(), testLocalKind, {0x01}}};
}

EvaluationCondition clockPeriod(const char *test, std::int64_t coefficient) {
  return EvaluationCondition{RequiredClockPeriodCondition{
      rootTarget(subjectArtifact()), decimal(test, coefficient, -10)}};
}

llvm::Error verifyDistinctTargets(llvm::ArrayRef<SubjectTargetRef> targets) {
  if (targets[0].target == targets[1].target)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "relation targets must be distinct");
  return llvm::Error::success();
}

const ScopeRoleDescriptor rootRoles[] = {{ScopeRoleRef(0), "target"}};
const ScopeRoleDescriptor relationRoles[] = {{ScopeRoleRef(0), "source"},
                                             {ScopeRoleRef(1), "destination"}};

const std::vector<OrderedTargetPattern> rootPatterns = {
    {testSignatureRef(), {{subjectRole(), subjectRootType()}}},
};
const std::vector<OrderedTargetPattern> localPatterns = {
    {testSignatureRef(), {{subjectRole(), subjectLocalType()}}},
};
const std::vector<OrderedTargetPattern> relationPatterns = {
    {testSignatureRef(),
     {{subjectRole(), subjectRootType()}, {subjectRole(), subjectRootType()}}},
};

const ScopeFormDescriptor scopeForms[] = {
    {ScopeFormRef(0), "the entire exact case", {}, WholeExactCaseScope{},
     nullptr},
    {ScopeFormRef(1), "one exact root", rootRoles,
     ExactTargetPatternsScope{rootPatterns}, nullptr},
    {ScopeFormRef(2), "one exact local target", rootRoles,
     ExactTargetPatternsScope{localPatterns}, nullptr},
    {ScopeFormRef(3), "an ordered root relation", relationRoles,
     ExactTargetPatternsScope{relationPatterns}, &verifyDistinctTargets},
};

constexpr FindingKind testFindingKind{23};
constexpr FindingKind secondFindingKind{24};
constexpr EvaluationModelKind firstModelKind{31};
constexpr EvaluationModelKind secondModelKind{32};
constexpr EvaluationModelKind slottedModelKind{34};

struct EmptyFindingOccurrence {};

llvm::Expected<std::vector<std::uint8_t>>
encodeEmptyFindingOccurrence(const OwnerValue &occurrence) {
  if (!occurrence.getIf<EmptyFindingOccurrence>())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "test finding occurrence has the wrong owner type");
  return std::vector<std::uint8_t>{};
}

llvm::Expected<OwnerValue>
decodeEmptyFindingOccurrence(llvm::ArrayRef<std::uint8_t> canonicalPayload) {
  if (!canonicalPayload.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "test finding occurrence must be empty");
  return OwnerValue::get(EmptyFindingOccurrence{});
}

llvm::Error validateEmptyFindingOccurrence(
    const OwnerValue &occurrence, const FindingOccurrenceContext &) {
  if (!occurrence.getIf<EmptyFindingOccurrence>())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "test finding occurrence has the wrong owner type");
  return llvm::Error::success();
}

struct EmptyModelConfigView {};

llvm::ArrayRef<std::uint8_t> emptyModelConfigSchemaBytes() {
  static constexpr llvm::StringLiteral descriptor =
      "loom.test.evaluation.model_config.1.0";
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
                                   "test model config has the wrong type");
  return std::vector<std::uint8_t>{};
}

llvm::Expected<OwnerValue>
adoptEmptyModelConfig(llvm::ArrayRef<std::uint8_t> canonicalBytes,
                      const ComponentViewDigest &) {
  if (!canonicalBytes.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "test model config must be empty");
  return OwnerValue::get(EmptyModelConfigView{});
}

const FindingOccurrenceCodec testFindingOccurrenceCodec{
    {"loom.test.evaluation.empty_finding_occurrence", {1, 0}},
    &encodeEmptyFindingOccurrence,
    &decodeEmptyFindingOccurrence,
    &validateEmptyFindingOccurrence};

const FindingDescriptor testFindingDescriptor{
    testFindingKind,
    "test_finding",
    "A typed condition used by the Evaluation persistence anchors.",
    {scopeForms, 1},
    {},
    testFindingOccurrenceCodec,
    std::nullopt};

const FindingDescriptor secondFindingDescriptor{
    secondFindingKind,
    "test_secondary",
    "A second scoped finding used to anchor request ordering.",
    {scopeForms, 1},
    {},
    testFindingOccurrenceCodec,
    std::nullopt};

const ScopeFormRef testFindingScopeForms[] = {ScopeFormRef(0)};
const ScopeFormRef testMetricScopeForms[] = {ScopeFormRef(0)};

const MetricCapability metricCapabilities[] = {
    {MetricKind::CycleCount, testMetricScopeForms, allObservationFormsMask()},
};

const FindingCapability findingCapabilities[] = {
    {testFindingKind, testFindingScopeForms, allFindingResultFormsMask()},
    {secondFindingKind, testFindingScopeForms, allFindingResultFormsMask()},
};

const ResolvedModelConfigViewContract emptyModelConfigView{
    emptyModelConfigSchemaBytes(), &projectEmptyModelConfig,
    &encodeEmptyModelConfig, &adoptEmptyModelConfig};

const ModeledPhenomenon analyticPhenomena[] = {
    ModeledPhenomenon::CanonicalDataflow};
const EvaluationInteractionMode incrementalModes[] = {
    EvaluationInteractionMode::Incremental};

EvaluationInteractionDomainRef testInteractionDomainRef() {
  return llvm::cantFail(EvaluationInteractionDomainRef::get(
      "loom.test.mapping_interaction", {1, 0}, 7));
}

llvm::Error validateTestInteractionProtocol(EvaluationInteractionMode mode) {
  if (mode != EvaluationInteractionMode::Incremental)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "test interaction mode is unsupported");
  return llvm::Error::success();
}

const EvaluationInteractionDomainDescriptor testInteractionDomain{
    testInteractionDomainRef(), "A typed test candidate interaction domain.",
    incrementalModes, &validateTestInteractionProtocol};
const EvaluationInteractionCapability analyticInteractions[] = {
    {testInteractionDomainRef(), incrementalModes},
};

const EvaluationModelDescriptor firstModelDescriptor{
    firstModelKind,
    "test_model_first",
    "loom.test.model.first",
    testSignatureRef(),
    {},
    metricCapabilities,
    findingCapabilities,
    {},
    {},
    emptyModelConfigView,
    analyticPhenomena,
    EvaluationExecutionMethod::Analytic,
    analyticInteractions,
    DeterminismContract::Deterministic,
    {}};

const EvaluationModelDescriptor secondModelDescriptor{
    secondModelKind,
    "test_model_second",
    "loom.test.model.second",
    testSignatureRef(),
    {},
    {},
    findingCapabilities,
    {},
    {},
    emptyModelConfigView,
    {},
    EvaluationExecutionMethod::Analytic,
    {},
    DeterminismContract::Deterministic,
    {}};

const ArtifactSchemaDescriptor *const modelInputSchemas[] = {&subjectSchema};

const ModelInputSlotDescriptor modelInputSlots[] = {
    {ModelInputSlotRef(0), "calibration", modelInputSchemas,
     ArtifactCollectionCardinality::ExactlyOne, nullptr},
};

const ModelOutputSlotDescriptor modelOutputSlots[] = {
    {ModelOutputSlotRef(0),
     "execution",
     &subjectSchema,
     {ArtifactCollectionCardinality::ExactlyOne,
      ArtifactCollectionCardinality::Forbidden,
      ArtifactCollectionCardinality::Forbidden,
      ArtifactCollectionCardinality::ZeroOrOne}},
};

const FindingQuery mandatoryTerminalFindings[] = {
    {testFindingKind, EvaluationScope{ScopeFormRef(0), {}}},
};

const EvaluationModelDescriptor slottedModelDescriptor{
    slottedModelKind,
    "test_model_slotted",
    "loom.test.model.slotted",
    testSignatureRef(),
    {},
    {},
    findingCapabilities,
    modelInputSlots,
    modelOutputSlots,
    emptyModelConfigView,
    analyticPhenomena,
    EvaluationExecutionMethod::Simulation,
    {},
    DeterminismContract::Deterministic,
    mandatoryTerminalFindings};

constexpr EvaluationModelKind clockAliasModelKind{40};
constexpr EvaluationModelKind failingCycleModelKind{41};

const ModelConditionCapability clockAliasConditionCapabilities[] = {
    {clockAliasBasePatterns[0], ConditionDisposition::Consumed}};

const EvaluationModelDescriptor clockAliasModelDescriptor{
    clockAliasModelKind,
    "test_model_clock_alias",
    "loom.test.model.clock_alias",
    clockAliasSignatureRef(),
    clockAliasConditionCapabilities,
    metricCapabilities,
    findingCapabilities,
    {},
    {},
    emptyModelConfigView,
    analyticPhenomena,
    EvaluationExecutionMethod::Analytic,
    analyticInteractions,
    DeterminismContract::Deterministic,
    {}};

const EvaluationModelDescriptor failingCycleModelDescriptor{
    failingCycleModelKind,
    "test_model_failing_cycle",
    "loom.test.model.failing_cycle",
    failingCycleSignatureRef(),
    {},
    metricCapabilities,
    findingCapabilities,
    {},
    {},
    emptyModelConfigView,
    analyticPhenomena,
    EvaluationExecutionMethod::Analytic,
    analyticInteractions,
    DeterminismContract::Deterministic,
    {}};

ResolvedModelBinding
modelBinding(const char *test, const EvaluationModelDescriptor &descriptor,
             std::vector<ModelInputBinding> inputs = {}) {
  return takeExpected(test, ResolvedModelBinding::project(
                                descriptor.reference(), std::move(inputs),
                                defaultResolvedConfig()));
}

void sharedSignatureDerivesOneCaseKeyAcrossDescriptors() {
  const EvaluationCase firstCase =
      evaluationCase(__func__, {}, firstModelDescriptor.caseSignature);
  const EvaluationCase secondCase =
      evaluationCase(__func__, {}, secondModelDescriptor.caseSignature);
  require(__func__, baseCaseKey(firstCase) == baseCaseKey(secondCase),
          "model descriptors changed the shared case key");

  const CaseArtifactResolution resolved = resolution(__func__);
  const FindingRequest finding = takeExpected(
      __func__,
      FindingRequest::get(
          FindingQuery{testFindingKind, EvaluationScope{ScopeFormRef(0), {}}},
          {}, firstCase, resolved, artifactStore()));
  const FindingRequest findingRequests[] = {finding};

  const ResolvedModelBinding firstBinding =
      modelBinding(__func__, firstModelDescriptor);
  const ResolvedModelBinding secondBinding =
      modelBinding(__func__, secondModelDescriptor);
  const EvaluationRequest firstRequest = takeExpected(
      __func__,
      EvaluationRequest::get(firstCase, {}, findingRequests, firstBinding, 0,
                             resolved, artifactStore()));
  const EvaluationRequest secondRequest = takeExpected(
      __func__,
      EvaluationRequest::get(secondCase, {}, findingRequests, secondBinding, 0,
                             resolved, artifactStore()));

  require(__func__, firstRequest.metricRequests().empty(),
          "finding-only request gained a metric request");
  require(__func__, firstRequest.findingRequests().size() == 1,
          "finding-only request lost its finding request");
  require(__func__,
          evaluationRequestIdentity(firstRequest) !=
              evaluationRequestIdentity(secondRequest),
          "distinct model bindings produced one request identity");
}

FindingRequest findingRequest(const char *test, FindingKind kind,
                              const EvaluationCase &evaluationCase,
                              const CaseArtifactResolution &resolved) {
  return takeExpected(
      test, FindingRequest::get(
                FindingQuery{kind, EvaluationScope{ScopeFormRef(0), {}}}, {},
                evaluationCase, resolved, artifactStore()));
}

void requestVerifierRejectsNoncanonicalAndForeignBindings() {
  const EvaluationCase current = evaluationCase(__func__, {});
  const CaseArtifactResolution resolved = resolution(__func__);
  const FindingRequest first =
      findingRequest(__func__, testFindingKind, current, resolved);
  const FindingRequest second =
      findingRequest(__func__, secondFindingKind, current, resolved);
  const EvaluationCondition quantile{QuantileCondition{ratio(__func__, 9, 10)}};
  expectErrorContains(
      __func__, FindingRequest::get(first.query(), {quantile}, current,
                                    resolved, artifactStore()),
      "not permitted in finding-request conditions");

  const ResolvedModelBinding firstBinding =
      modelBinding(__func__, firstModelDescriptor);
  expectErrorContains(__func__,
                      EvaluationRequest::get(current, {}, {}, firstBinding, 0,
                                             resolved, artifactStore()),
                      "requires a metric or finding request");
  expectErrorContains(__func__,
                      EvaluationRequest::get(current, {}, {first, first},
                                             firstBinding, 0, resolved,
                                             artifactStore()),
                      "duplicate finding request");
  expectErrorContains(__func__,
                      EvaluationRequest::get(current, {}, {first}, firstBinding,
                                             1, resolved, artifactStore()),
                      "replicate_index zero");
  const std::vector<std::uint8_t> invalidConfigBytes{0x01};
  const ComponentViewDigest invalidConfigDigest = takeExpected(
      __func__, computeComponentViewDigest(emptyModelConfigSchemaBytes(),
                                           invalidConfigBytes));
  expectErrorContains(__func__,
                      ResolvedModelBinding::adopt(
                          firstModelDescriptor.reference(), {},
                          invalidConfigBytes, invalidConfigDigest),
                      "test model config must be empty");

  const EvaluationRequest ordered = takeExpected(
      __func__,
      EvaluationRequest::get(current, {}, {second, first}, firstBinding, 0,
                             resolved, artifactStore()));
  require(__func__,
          ordered.resolve(FindingRequestOrdinal(0))->query().kind ==
              testFindingKind,
          "finding request ordinals did not follow canonical order");
  require(__func__,
          ordered.resolve(FindingRequestOrdinal(1))->query().kind ==
              secondFindingKind,
          "second finding request ordinal was unstable");
  require(__func__, ordered.resolve(FindingRequestOrdinal(2)) == nullptr,
          "out-of-range finding ordinal resolved");
  expectErrorContains(
      __func__,
      EvaluationCase::get(testSignatureRef(), bindings(__func__), std::nullopt,
                          std::nullopt, {quantile}, resolved, artifactStore()),
      "not permitted in base");

  const EvaluationCase conditioned =
      evaluationCase(__func__, {clockPeriod(__func__, 8)});
  const FindingRequest conditionedFinding =
      findingRequest(__func__, testFindingKind, conditioned, resolved);
  expectErrorContains(
      __func__, EvaluationRequest::get(conditioned, {}, {conditionedFinding},
                                       firstBinding, 0, resolved,
                                       artifactStore()),
      "does not recognize condition");

  expectErrorContains(
      __func__,
      ResolvedModelBinding::project(slottedModelDescriptor.reference(),
                                    {{ModelInputSlotRef(0), {}}},
                                    defaultResolvedConfig()),
      "declared cardinality");
  expectErrorContains(__func__,
                      ResolvedModelBinding::project(
                          slottedModelDescriptor.reference(),
                          {{ModelInputSlotRef(0), {platformArtifact()}}},
                          defaultResolvedConfig()),
                      "rejects artifact schema");
  const ResolvedModelBinding unresolvedInputBinding = modelBinding(
      __func__, slottedModelDescriptor,
      {{ModelInputSlotRef(0), {subjectArtifact()}}});
  takeExpected(__func__,
               EvaluationRequest::get(current, {}, {first},
                                      unresolvedInputBinding, 0, resolved,
                                      artifactStore()));
  TemporaryDirectory directory(__func__);
  const ArtifactStore inputStore(directory.path());
  const ArtifactIdentity inputIdentity = takeExpected(
      __func__,
      inputStore.put(subjectSchema,
                     CanonicalSemanticBytes(std::vector<std::uint8_t>{0x5a})));
  const ArtifactRootReference inputArtifact =
      root(subjectSchema, inputIdentity);
  const ResolvedModelBinding slottedBinding = modelBinding(
      __func__, slottedModelDescriptor,
      {{ModelInputSlotRef(0), {inputArtifact}}});
  takeExpected(__func__,
               EvaluationRequest::get(current, {}, {first}, slottedBinding, 0,
                                      resolved, inputStore));
  expectErrorContains(
      __func__,
      EvaluationRequest::get(current, {}, {second}, slottedBinding, 0,
                             resolved, inputStore),
      "omits a mandatory terminal finding");

  const ModelOutputSlotDescriptor *output =
      slottedModelDescriptor.outputSlotByOrdinal(0);
  require(__func__, output && output->schema == &subjectSchema,
          "output slot did not resolve its exact schema");
  require(__func__,
          output->cardinality(EvidenceOutcomeKind::Completed) ==
                  ArtifactCollectionCardinality::ExactlyOne &&
              output->cardinality(EvidenceOutcomeKind::Unsupported) ==
                  ArtifactCollectionCardinality::Forbidden &&
              output->cardinality(EvidenceOutcomeKind::ExecutionFailed) ==
                  ArtifactCollectionCardinality::Forbidden &&
              output->cardinality(EvidenceOutcomeKind::CancelledOrTimeout) ==
                  ArtifactCollectionCardinality::ZeroOrOne,
          "output slot outcome cardinalities changed");
  require(__func__, slottedModelDescriptor.outputSlotByOrdinal(1) == nullptr,
          "foreign output-slot ordinal resolved");
}

void wholeCaseMetricRequiresSignatureCycleBasis() {
  const EvaluationCase current = evaluationCase(__func__, {});
  const CaseArtifactResolution resolved = resolution(__func__);
  const MetricRequest metric = takeExpected(
      __func__, MetricRequest::get(
                    MetricQuery{MetricKind::CycleCount,
                                EvaluationScope{ScopeFormRef(0), {}}},
                    {}, current, resolved, artifactStore()));
  const ResolvedModelBinding binding =
      modelBinding(__func__, firstModelDescriptor);
  require(__func__,
          binding.resolvedModelConfig().getIf<EmptyModelConfigView>() !=
              nullptr,
          "model binding did not retain the owner-typed config view");
  const EvaluationRequest request = takeExpected(
      __func__, EvaluationRequest::get(current, {metric}, {}, binding, 0,
                                       resolved, artifactStore()));
  require(__func__, request.metricRequests().size() == 1 &&
                        request.findingRequests().empty(),
          "metric-only Request changed its requested result sets");

  const EvaluationCase foreignCase =
      evaluationCase(__func__, {}, noCycleSignatureRef());
  const FindingRequest foreignFinding =
      findingRequest(__func__, testFindingKind, foreignCase, resolved);
  expectErrorContains(
      __func__, EvaluationRequest::get(foreignCase, {}, {foreignFinding},
                                       binding, 0, resolved, artifactStore()),
      "signature does not match");

  const EvaluationModelDescriptor invalidModel{
      EvaluationModelKind{35},
      "test_model_without_cycle",
      "loom.test.model.without_cycle",
      noCycleSignatureRef(),
      {},
      metricCapabilities,
      {},
      {},
      {},
      emptyModelConfigView,
      analyticPhenomena,
      EvaluationExecutionMethod::Analytic,
      {},
      DeterminismContract::Deterministic,
      {}};
  expectErrorContains(__func__, registerEvaluationModelDescriptor(invalidModel),
                      "requires a unique whole-case reference cycle");
}

void requestVerifierDistinguishesLocalClockTargets() {
  TemporaryDirectory directory(__func__);
  const ArtifactStore store(directory.path());
  // Two anchors in distinct roles; both closures reach one shared owning
  // Artifact that carries distinct canonical local clock-domain targets.
  const ArtifactRootReference referenceSubject =
      storeSubject(__func__, store, 0x51);
  const ArtifactRootReference dependentSubject =
      storeSubject(__func__, store, 0x52);
  const ArtifactRootReference sharedOwner = storeSubject(__func__, store, 0x60);
  const EvaluationSubjectBindings twoAnchors = takeExpected(
      __func__,
      EvaluationSubjectBindings::get({{clockReferenceRole(), {referenceSubject}},
                                       {clockDependentRole(), {dependentSubject}}}));
  const CaseArtifactResolution resolved = takeExpected(
      __func__, CaseArtifactResolution::get(
                    {{referenceSubject, {sharedOwner}},
                     {dependentSubject, {sharedOwner}},
                     {sharedOwner, {}}}));
  const ResolvedModelBinding binding =
      modelBinding(__func__, clockAliasModelDescriptor);
  const SubjectTargetRef referenceTarget{
      clockReferenceRole(), referenceSubject,
      EncodedArtifactLocalReference{sharedOwner, clockLocalKind, {0x01}}};

  // Two distinct canonical local targets in one Artifact resolve to distinct
  // exact SubjectTarget identities and are accepted across the two roles.
  const EvaluationCondition distinctLocals{RelativeClockScheduleCondition{
      referenceTarget,
      SubjectTargetRef{clockDependentRole(), dependentSubject,
                       EncodedArtifactLocalReference{sharedOwner, clockLocalKind,
                                                     {0x02}}},
      ratio(__func__, 1, 1), ratio(__func__, 0, 1)}};
  const EvaluationCase distinctCase = takeExpected(
      __func__, EvaluationCase::get(clockAliasSignatureRef(), twoAnchors,
                                    std::nullopt, std::nullopt, {distinctLocals},
                                    resolved, store));
  const MetricRequest distinctMetric = takeExpected(
      __func__, MetricRequest::get(
                    MetricQuery{MetricKind::CycleCount,
                                EvaluationScope{ScopeFormRef(0), {}}},
                    {}, distinctCase, resolved, store));
  takeExpected(__func__, EvaluationRequest::get(distinctCase, {distinctMetric},
                                                {}, binding, 0, resolved, store));

  // The same exact local target across the two roles resolves to one identity
  // and is rejected, even though both the role and the anchor differ.
  const EvaluationCondition sameLocal{RelativeClockScheduleCondition{
      referenceTarget,
      SubjectTargetRef{clockDependentRole(), dependentSubject,
                       EncodedArtifactLocalReference{sharedOwner, clockLocalKind,
                                                     {0x01}}},
      ratio(__func__, 1, 1), ratio(__func__, 0, 1)}};
  const EvaluationCase sameCase = takeExpected(
      __func__, EvaluationCase::get(clockAliasSignatureRef(), twoAnchors,
                                    std::nullopt, std::nullopt, {sameLocal},
                                    resolved, store));
  const MetricRequest sameMetric = takeExpected(
      __func__, MetricRequest::get(
                    MetricQuery{MetricKind::CycleCount,
                                EvaluationScope{ScopeFormRef(0), {}}},
                    {}, sameCase, resolved, store));
  expectErrorContains(__func__,
                      EvaluationRequest::get(sameCase, {sameMetric}, {}, binding,
                                             0, resolved, store),
                      "clock domain");
}

void metricScopeAdmissibilityReturnsValidatedBasis() {
  const EvaluationCase current = evaluationCase(__func__, {});
  const CaseArtifactResolution resolved = resolution(__func__);
  // CycleCount owns a registry admission that requires the case-signature
  // reference-cycle basis. The metric registry owns admission and the
  // case-signature registry owns basis resolution; the resolved basis is
  // returned once so the caller may propagate or consume it without resolving
  // a second time.
  auto cycleAdmission = validateMetricScopeAdmissibility(
      MetricKind::CycleCount, ScopeFormRef(0), current, resolved,
      artifactStore());
  const std::optional<ReferenceCycleBasis> basis =
      takeExpected(__func__, std::move(cycleAdmission));
  require(__func__, basis.has_value(),
          "cycle-count admission did not return the validated cycle basis");
  require(__func__, std::holds_alternative<AbstractCaseCycle>(*basis),
          "cycle-count admission returned the wrong reference-cycle basis");

  // Runtime does not require a cycle basis, so admission returns no basis.
  auto runtimeAdmission = validateMetricScopeAdmissibility(
      MetricKind::Runtime, ScopeFormRef(0), current, resolved, artifactStore());
  require(__func__,
          !takeExpected(__func__, std::move(runtimeAdmission)).has_value(),
          "runtime admission returned an unexpected cycle basis");

  // The descriptor context remains error-only: a signature without a whole-case
  // cycle basis rejects cycle-count admission instead of returning a basis.
  const EvaluationCase withoutCycle =
      evaluationCase(__func__, {}, noCycleSignatureRef());
  expectErrorContains(__func__,
                      validateMetricScopeAdmissibility(
                          MetricKind::CycleCount, ScopeFormRef(0), withoutCycle,
                          resolved, artifactStore()),
                      "requires a unique whole-case reference cycle");
}

void requestVerifierOwnsExactCycleBasisResolution() {
  // The exact subject-cycle resolver fails. MetricRequest construction admits
  // CycleCount through descriptor-context static admission only and does not
  // resolve the basis, so construction succeeds without invoking the resolver.
  const EvaluationCase failingCase =
      evaluationCase(__func__, {}, failingCycleSignatureRef());
  const CaseArtifactResolution resolved = resolution(__func__);
  const MetricRequest metric = takeExpected(
      __func__, MetricRequest::get(
                    MetricQuery{MetricKind::CycleCount,
                                EvaluationScope{ScopeFormRef(0), {}}},
                    {}, failingCase, resolved, artifactStore()));
  const ResolvedModelBinding binding =
      modelBinding(__func__, failingCycleModelDescriptor);
  // RequestVerifier owns the exact request-context admission and resolves the
  // basis exactly once, so the resolver failure surfaces here and only here.
  expectErrorContains(
      __func__,
      EvaluationRequest::get(failingCase, {metric}, {}, binding, 0, resolved,
                             artifactStore()),
      "test reference-cycle resolution failure");
}

void requestCanonicalRoundTripAndStoreImport() {
  const EvaluationCase current = evaluationCase(__func__, {});
  const CaseArtifactResolution resolved = resolution(__func__);
  const FindingRequest finding = takeExpected(
      __func__,
      FindingRequest::get(
          FindingQuery{testFindingKind, EvaluationScope{ScopeFormRef(0), {}}},
          {}, current, resolved, artifactStore()));
  const ResolvedModelBinding binding =
      modelBinding(__func__, firstModelDescriptor);
  const EvaluationRequest request = takeExpected(
      __func__, EvaluationRequest::get(current, {}, {finding}, binding, 0,
                                       resolved, artifactStore()));

  const std::string canonical = serializeEvaluationRequest(request);
  require(__func__,
          !llvm::StringRef(canonical).contains("case_signature") &&
              !llvm::StringRef(canonical).contains("case_key") &&
              !llvm::StringRef(canonical).contains("model_key") &&
              !llvm::StringRef(canonical).contains("cache"),
          "request serialized a forbidden derived authority");
  require(__func__,
          llvm::StringRef(canonical).contains(
              "\"descriptor_ref\":{\"schema_major\":2,\"schema_minor\":0,") &&
              llvm::StringRef(canonical).contains(
                  "\"resolved_model_config\":{\"canonical_view_bytes\":\"\"") &&
              llvm::StringRef(canonical).contains(
                  "\"component_view_digest\":\"") &&
              !llvm::StringRef(canonical).contains(
                  "\"resolved_model_config\":[]"),
          "request did not use the descriptor-owned component-view wire");
  const std::string findingQueryJson =
      takeExpected(__func__, serializeFindingQuery(finding.query()));
  require(__func__,
          findingQueryJson ==
              "{\"schema\":\"evaluation.finding_query\",\"schema_version\":"
              "\"1.0\",\"finding\":\"test_finding\",\"scope\":{\"form\":0,"
              "\"targets\":[]}}" &&
              llvm::StringRef(canonical).contains(
                  "\"query\":{\"finding\":\"test_finding\",\"scope\":{"
                  "\"form\":0,\"targets\":[]}}"),
          "finding query payload wire diverged between codecs");
  require(__func__,
          takeExpected(__func__, parseFindingQuery(findingQueryJson)) ==
              finding.query(),
          "finding query payload did not roundtrip");
  const EvaluationRequest parsed = takeExpected(
      __func__, parseEvaluationRequest(canonical, resolved, artifactStore()));
  require(__func__, serializeEvaluationRequest(parsed) == canonical,
          "request canonical roundtrip changed bytes");
  require(__func__,
          evaluationRequestIdentity(parsed) ==
              evaluationRequestIdentity(request),
          "request canonical roundtrip changed identity");
  std::array<std::uint8_t, ComponentViewDigest::byteSize> staleDigestBytes{};
  const ComponentViewDigest staleDigest = takeExpected(
      __func__, ComponentViewDigest::fromBytes(staleDigestBytes));
  expectErrorContains(
      __func__, ResolvedModelBinding::adopt(firstModelDescriptor.reference(),
                                            {}, {}, staleDigest),
      "component_view_digest_mismatch");
  expectErrorContains(
      __func__,
      parseEvaluationRequest(canonical + "\n", resolved, artifactStore()),
      "not canonical");

  std::string legacyDescriptorVersion = canonical;
  const std::string legacyDescriptorTag =
      "\"schema_major\":2,\"schema_minor\":0";
  const std::size_t descriptorPosition =
      legacyDescriptorVersion.find(legacyDescriptorTag);
  require(__func__, descriptorPosition != std::string::npos,
          "descriptor reference wire anchor moved");
  legacyDescriptorVersion.replace(descriptorPosition,
                                 legacyDescriptorTag.size(),
                                 "\"schema_version\":\"1.0\"");
  expectErrorContains(
      __func__,
      parseEvaluationRequest(legacyDescriptorVersion, resolved, artifactStore()),
      "unknown field 'schema_version'");

  TemporaryDirectory directory(__func__);
  const ArtifactStore store(directory.path());
  auto missingDependency = publishEvaluationRequest(request, store);
  if (missingDependency)
    fail(__func__, "Request publication accepted a missing direct dependency");
  const std::string missingMessage =
      llvm::toString(missingDependency.takeError());
  require(__func__,
          llvm::StringRef(missingMessage).starts_with(
              "artifact_store_missing:"),
          "Request publication changed the ArtifactStore error class: " +
              missingMessage);

  const ArtifactIdentity storedSubjectIdentity = takeExpected(
      __func__, store.put(subjectSchema, CanonicalSemanticBytes(
                                           std::vector<std::uint8_t>{0x41})));
  const ArtifactRootReference storedSubject =
      root(subjectSchema, storedSubjectIdentity);
  const EvaluationSubjectBindings storedBindings = takeExpected(
      __func__, EvaluationSubjectBindings::get(
                    {{subjectRole(), {storedSubject}}}));
  const CaseArtifactResolution storedResolution = takeExpected(
      __func__, CaseArtifactResolution::get({{storedSubject, {}}}));
  const EvaluationCase storedCase = takeExpected(
      __func__, EvaluationCase::get(testSignatureRef(), storedBindings,
                                    std::nullopt, std::nullopt, {},
                                    storedResolution, store));
  const FindingRequest storedFinding = takeExpected(
      __func__, FindingRequest::get(
                    FindingQuery{testFindingKind,
                                 EvaluationScope{ScopeFormRef(0), {}}},
                    {}, storedCase, storedResolution, store));
  const EvaluationRequest storedRequest = takeExpected(
      __func__, EvaluationRequest::get(storedCase, {}, {storedFinding},
                                       binding, 0, storedResolution, store));
  const ArtifactRootReference published =
      takeExpected(__func__, publishEvaluationRequest(storedRequest, store));
  const EvaluationRequest imported = takeExpected(
      __func__, importEvaluationRequest(published, storedResolution, store));
  require(__func__,
          serializeEvaluationRequest(imported) ==
              serializeEvaluationRequest(storedRequest),
          "ArtifactStore import changed request semantics");

  TemporaryDirectory incompleteDirectory(__func__);
  const ArtifactStore incompleteStore(incompleteDirectory.path());
  const ArtifactIdentity incompleteIdentity = takeExpected(
      __func__, incompleteStore.put(EvaluationRequest::artifactSchema,
                                    canonicalEvaluationRequestBytes(
                                        storedRequest)));
  const ArtifactRootReference incompleteReference{
      EvaluationRequest::artifactSchema.identity.str(),
      EvaluationRequest::artifactSchema.version, incompleteIdentity};
  auto missingImport = importEvaluationRequest(
      incompleteReference, storedResolution, incompleteStore);
  if (missingImport)
    fail(__func__, "Request import accepted a missing direct dependency");
  const std::string missingImportMessage =
      llvm::toString(missingImport.takeError());
  require(__func__,
          llvm::StringRef(missingImportMessage).starts_with(
              "artifact_store_missing:"),
          "Request import changed the ArtifactStore error class: " +
              missingImportMessage);

  ArtifactRootReference foreign = published;
  foreign.schemaIdentity = "loom.test.foreign_request";
  expectErrorContains(__func__,
                      importEvaluationRequest(foreign, resolved, store),
                      "foreign EvaluationRequest");
  ArtifactRootReference stale = published;
  stale.artifact = testArtifact({0xee});
  expectError(__func__, importEvaluationRequest(stale, resolved, store));
}

void scopeChecksAnchorClosureLocalProviderAndRoleOrder() {
  const EvaluationCase current = evaluationCase(__func__, {});
  const CaseArtifactResolution resolved = resolution(__func__);
  const CaseTargetContext context =
      current.targetContext(resolved, artifactStore());

  const EvaluationScope whole{ScopeFormRef(0), {}};
  const EvaluationScope subject{ScopeFormRef(1),
                                {rootTarget(subjectArtifact())}};
  const EvaluationScope dependency{ScopeFormRef(1),
                                   {rootTarget(dependencyArtifact())}};
  for (const EvaluationScope &scope : {whole, subject, dependency})
    if (llvm::Error error =
            validateEvaluationScopeCase(scope, scopeForms, context))
      fail(__func__, llvm::toString(std::move(error)));

  expectErrorContains(
      __func__,
      validateEvaluationScopeCase(
          EvaluationScope{ScopeFormRef(1),
                          {SubjectTargetRef{subjectRole(), dependencyArtifact(),
                                            dependencyArtifact()}}},
          scopeForms, context),
      "is not bound");
  expectErrorContains(
      __func__,
      validateEvaluationScopeCase(
          EvaluationScope{ScopeFormRef(1), {rootTarget(foreignArtifact())}},
          scopeForms, context),
      "not reachable");

  const CaseArtifactResolution unresolved = takeExpected(
      __func__, CaseArtifactResolution::get(
                    {{subjectArtifact(), {dependencyArtifact()}}}));
  expectErrorContains(
      __func__,
      validateEvaluationScopeCase(
          EvaluationScope{ScopeFormRef(1), {rootTarget(dependencyArtifact())}},
          scopeForms, current.targetContext(unresolved, artifactStore())),
      "target artifact is unresolved");

  expectOwnerCodecUnavailable(
      __func__, validateEvaluationScopeCase(
                    EvaluationScope{ScopeFormRef(2), {localTarget()}},
                    scopeForms, context));

  const EvaluationScope forward{
      ScopeFormRef(3),
      {rootTarget(subjectArtifact()), rootTarget(dependencyArtifact())}};
  const EvaluationScope reverse{
      ScopeFormRef(3),
      {rootTarget(dependencyArtifact()), rootTarget(subjectArtifact())}};
  for (const EvaluationScope &scope : {forward, reverse})
    if (llvm::Error error =
            validateEvaluationScopeCase(scope, scopeForms, context))
      fail(__func__, llvm::toString(std::move(error)));
  require(__func__, canonicalScopeKey(forward) != canonicalScopeKey(reverse),
          "scope role order did not affect the canonical key");
  expectErrorContains(__func__,
                      validateEvaluationScopeCase(
                          EvaluationScope{ScopeFormRef(3),
                                          {rootTarget(subjectArtifact()),
                                           rootTarget(subjectArtifact())}},
                          scopeForms, context),
                      "must be distinct");
}

void conditionsCheckLocationApplicabilityDuplicatesAndConflicts() {
  const CaseArtifactResolution resolved = resolution(__func__);
  const EvaluationCase current = evaluationCase(__func__, {});
  const CaseTargetContext context =
      current.targetContext(resolved, artifactStore());

  expectErrorContains(
      __func__,
      EvaluationCase::get(
          testSignatureRef(), bindings(__func__), std::nullopt, std::nullopt,
          {EvaluationCondition{QuantileCondition{ratio(__func__, 1, 2)}}},
          resolved, artifactStore()),
      "not permitted in base conditions");

  const EvaluationCondition period = clockPeriod(__func__, 8);
  expectErrorContains(__func__,
                      canonicalizeEvaluationConditions(
                          {period}, ConditionLocation::MetricRequest,
                          "test metric", baseConditionPatterns, context),
                      "not permitted in metric-request conditions");

  expectErrorContains(
      __func__,
      EvaluationCase::get(
          testSignatureRef(), bindings(__func__), std::nullopt, std::nullopt,
          {EvaluationCondition{SupplyVoltageCondition{
              rootTarget(subjectArtifact()), decimal(__func__, 9, -1)}}},
          resolved, artifactStore()),
      "is not applicable");

  expectOwnerCodecUnavailable(
      __func__,
      EvaluationCase::get(
          testSignatureRef(), bindings(__func__), std::nullopt, std::nullopt,
          {EvaluationCondition{ProcessCornerCondition{
              rootTarget(subjectArtifact()),
              platform::TechnologyCornerRef{platformArtifact().artifact,
                                            platform::TechnologyCornerId(0)}}}},
          resolved, artifactStore()));

  expectSimulationExecutionOwnerUnavailable(
      __func__,
      EvaluationCase::get(
          testSignatureRef(), bindings(__func__), std::nullopt, std::nullopt,
          {EvaluationCondition{ActivityBindingCondition{
              rootTarget(subjectArtifact()),
              ExecutionActivitySource{dependencyArtifact(), 0}}}},
          resolved, artifactStore()));

  expectErrorContains(__func__,
                      EvaluationCase::get(testSignatureRef(),
                                          bindings(__func__), std::nullopt,
                                          std::nullopt, {period, period},
                                          resolved, artifactStore()),
                      "duplicate evaluation condition");
  expectErrorContains(
      __func__,
      EvaluationCase::get(testSignatureRef(), bindings(__func__), std::nullopt,
                          std::nullopt, {period, clockPeriod(__func__, 6)},
                          resolved, artifactStore()),
      "conflicting");

  const EvaluationCondition schedule{RelativeClockScheduleCondition{
      rootTarget(subjectArtifact()), rootTarget(dependencyArtifact()),
      ratio(__func__, 1, 1), ratio(__func__, 0, 1)}};
  const EvaluationCase ordered = evaluationCase(__func__, {schedule, period});
  require(__func__, ordered.baseConditions().size() == 2,
          "valid condition assignments were not preserved");
  require(__func__,
          ordered.baseConditions()[0].kind() ==
                  EvaluationConditionKind::RequiredClockPeriod &&
              ordered.baseConditions()[1].kind() ==
                  EvaluationConditionKind::RelativeClockSchedule,
          "conditions did not use registry canonical order");

  const ModelConditionCapability widenedCapabilities[] = {
      {ConditionApplicabilityPattern{
           EvaluationConditionKind::SupplyVoltage,
           {testSignatureRef(), {{subjectRole(), subjectRootType()}}}},
       ConditionDisposition::Invariant}};
  const EvaluationModelDescriptor widenedModel{
      EvaluationModelKind{33},
      "test_model_widened",
      "loom.test.model.widened",
      testSignatureRef(),
      widenedCapabilities,
      {},
      {},
      {},
      {},
      emptyModelConfigView,
      {},
      EvaluationExecutionMethod::Analytic,
      {},
      DeterminismContract::Deterministic,
      {}};
  expectErrorContains(__func__,
                      validateModelCapability(widenedModel, current, {}),
                      "widens condition applicability");
}

void modelDescriptorOwnerContractsAreClosed() {
  expectErrorContains(
      __func__, EvaluationInteractionDomainRef::get("", {1, 0}, 0),
      "nonempty canonical ASCII");

  const EvaluationCase current = evaluationCase(__func__, {});
  EvaluationModelDescriptor invalidMethod = firstModelDescriptor;
  invalidMethod.modelKind = EvaluationModelKind(36);
  invalidMethod.executionMethod =
      static_cast<EvaluationExecutionMethod>(99);
  expectErrorContains(__func__,
                      validateModelCapability(invalidMethod, current, {}),
                      "invalid execution method");

  const ModeledPhenomenon duplicatePhenomena[] = {
      ModeledPhenomenon::ClockTiming, ModeledPhenomenon::ClockTiming};
  EvaluationModelDescriptor duplicatePhenomenonModel = firstModelDescriptor;
  duplicatePhenomenonModel.modelKind = EvaluationModelKind(37);
  duplicatePhenomenonModel.modeledPhenomena = duplicatePhenomena;
  expectErrorContains(
      __func__, validateModelCapability(duplicatePhenomenonModel, current, {}),
      "modeled phenomena must be canonical");

  const EvaluationInteractionDomainRef unknownDomain = takeExpected(
      __func__, EvaluationInteractionDomainRef::get(
                    "loom.test.unknown_interaction", {1, 0}, 0));
  const EvaluationInteractionCapability unknownCapabilities[] = {
      {unknownDomain, incrementalModes}};
  EvaluationModelDescriptor unknownInteractionModel = firstModelDescriptor;
  unknownInteractionModel.modelKind = EvaluationModelKind(38);
  unknownInteractionModel.interactionCapabilities = unknownCapabilities;
  expectErrorContains(
      __func__, validateModelCapability(unknownInteractionModel, current, {}),
      "unregistered interaction domain");

  const EvaluationInteractionMode guidanceModes[] = {
      EvaluationInteractionMode::Guidance};
  const EvaluationInteractionCapability unsupportedCapabilities[] = {
      {testInteractionDomainRef(), guidanceModes}};
  EvaluationModelDescriptor unsupportedInteractionModel = firstModelDescriptor;
  unsupportedInteractionModel.modelKind = EvaluationModelKind(39);
  unsupportedInteractionModel.interactionCapabilities =
      unsupportedCapabilities;
  expectErrorContains(
      __func__,
      validateModelCapability(unsupportedInteractionModel, current, {}),
      "does not implement interaction mode");

  require(__func__,
          canonicalEvaluationModelCapabilityBytes(firstModelDescriptor) !=
              canonicalEvaluationModelCapabilityBytes(secondModelDescriptor),
          "descriptor capability projection omitted owner contract fields");
}

} // namespace

void registry20ReferencesAreRequired() {
  // Registry 2.0 rejects 1.0 case and model descriptor references outright;
  // there is no reinterpretation path.
  auto legacyCase =
      EvaluationCaseSignatureRef::get(SchemaVersion{1, 0}, testCaseKind);
  if (legacyCase)
    fail(__func__, "a 1.0 case signature reference was reinterpreted");
  llvm::consumeError(legacyCase.takeError());
  auto legacyModel = EvaluationModelDescriptorRef::get(SchemaVersion{1, 0},
                                                       EvaluationModelKind(0));
  if (legacyModel)
    fail(__func__, "a 1.0 model descriptor reference was reinterpreted");
  llvm::consumeError(legacyModel.takeError());
  if (evaluationSchemaVersion() != SchemaVersion{2, 0})
    fail(__func__, "the Evaluation registry did not move to schema 2.0");
  // The Request and Evidence artifact root records stay at their own 1.0.
  if (EvaluationRequest::artifactSchema.version != SchemaVersion{1, 0} ||
      EvaluationEvidence::artifactSchema.version != SchemaVersion{1, 0})
    fail(__func__, "Request or Evidence artifact schema identity changed");
}

int main() {
  if (llvm::Error error = registerEvaluationCaseSignature(testSignature))
    fail("registration", llvm::toString(std::move(error)));
  if (llvm::Error error = registerEvaluationCaseSignature(noCycleSignature))
    fail("registration", llvm::toString(std::move(error)));
  if (llvm::Error error = registerEvaluationCaseSignature(clockAliasSignature))
    fail("registration", llvm::toString(std::move(error)));
  if (llvm::Error error = registerEvaluationCaseSignature(failingCycleSignature))
    fail("registration", llvm::toString(std::move(error)));
  if (llvm::Error error = registerArtifactLocalReferenceKind(
          subjectSchema, clockLocalKind,
          ArtifactLocalReferenceCodec{validateClockLocalPayload,
                                      validateClockLocalTarget}))
    fail("registration", llvm::toString(std::move(error)));
  if (llvm::Error error = registerFindingDescriptor(testFindingDescriptor))
    fail("registration", llvm::toString(std::move(error)));
  if (llvm::Error error = registerFindingDescriptor(secondFindingDescriptor))
    fail("registration", llvm::toString(std::move(error)));
  if (llvm::Error error =
          registerEvaluationInteractionDomain(testInteractionDomain))
    fail("registration", llvm::toString(std::move(error)));
  if (llvm::Error error =
          registerEvaluationModelDescriptor(firstModelDescriptor))
    fail("registration", llvm::toString(std::move(error)));
  if (llvm::Error error =
          registerEvaluationModelDescriptor(secondModelDescriptor))
    fail("registration", llvm::toString(std::move(error)));
  if (llvm::Error error =
          registerEvaluationModelDescriptor(slottedModelDescriptor))
    fail("registration", llvm::toString(std::move(error)));
  if (llvm::Error error =
          registerEvaluationModelDescriptor(clockAliasModelDescriptor))
    fail("registration", llvm::toString(std::move(error)));
  if (llvm::Error error =
          registerEvaluationModelDescriptor(failingCycleModelDescriptor))
    fail("registration", llvm::toString(std::move(error)));
  sharedSignatureDerivesOneCaseKeyAcrossDescriptors();
  requestVerifierRejectsNoncanonicalAndForeignBindings();
  wholeCaseMetricRequiresSignatureCycleBasis();
  metricScopeAdmissibilityReturnsValidatedBasis();
  requestVerifierOwnsExactCycleBasisResolution();
  requestVerifierDistinguishesLocalClockTargets();
  requestCanonicalRoundTripAndStoreImport();
  scopeChecksAnchorClosureLocalProviderAndRoleOrder();
  conditionsCheckLocationApplicabilityDuplicatesAndConflicts();
  modelDescriptorOwnerContractsAreClosed();
  registry20ReferencesAreRequired();
  return 0;
}
