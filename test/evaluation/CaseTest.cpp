#include "Evaluation/Case.h"
#include "Evaluation/ModelDescriptor.h"
#include "Evaluation/Request.h"

#include "Common/Artifact.h"
#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "ImplementationPlatform/TechnologyCorner.h"

#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <string>
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
constexpr std::uint32_t testLocalKind = 9;

EvaluationCaseSignatureRef testSignatureRef() {
  return llvm::cantFail(
      EvaluationCaseSignatureRef::get(evaluationSchemaVersion(), testCaseKind));
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
    baseConditionPatterns};

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

const std::vector<OrderedTargetPattern> wholeCasePatterns = {
    {testSignatureRef(), {}},
};
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
    {ScopeFormRef(0), "the entire exact case", {}, wholeCasePatterns, nullptr},
    {ScopeFormRef(1), "one exact root", rootRoles, rootPatterns, nullptr},
    {ScopeFormRef(2), "one exact local target", rootRoles, localPatterns,
     nullptr},
    {ScopeFormRef(3), "an ordered root relation", relationRoles,
     relationPatterns, &verifyDistinctTargets},
};

void sharedSignatureDerivesOneCaseKeyAcrossDescriptors() {
  const EvaluationModelDescriptor first{
      "loom.test.model.first", testSignatureRef(), {}, {}};
  const EvaluationModelDescriptor second{
      "loom.test.model.second", testSignatureRef(), {}, {}};
  const EvaluationCase firstCase =
      evaluationCase(__func__, {}, first.caseSignature);
  const EvaluationCase secondCase =
      evaluationCase(__func__, {}, second.caseSignature);
  require(__func__, baseCaseKey(firstCase) == baseCaseKey(secondCase),
          "model descriptors changed the shared case key");
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

  expectErrorContains(
      __func__,
      EvaluationCase::get(
          testSignatureRef(), bindings(__func__), std::nullopt, std::nullopt,
          {EvaluationCondition{ActivityBindingCondition{
              rootTarget(subjectArtifact()),
              ExecutionActivitySource{dependencyArtifact(), 0}}}},
          resolved, artifactStore()),
      "SimulationExecution activity validation is unavailable");

  expectErrorContains(
      __func__,
      EvaluationCase::get(testSignatureRef(), bindings(__func__), std::nullopt,
                          std::nullopt, {period, period}, resolved,
                          artifactStore()),
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
      "loom.test.model.widened", testSignatureRef(), widenedCapabilities, {}};
  expectErrorContains(__func__,
                      validateModelCapability(widenedModel, current, {}),
                      "widens condition applicability");
}

} // namespace

int main() {
  if (llvm::Error error = registerEvaluationCaseSignature(testSignature))
    fail("registration", llvm::toString(std::move(error)));
  sharedSignatureDerivesOneCaseKeyAcrossDescriptors();
  scopeChecksAnchorClosureLocalProviderAndRoleOrder();
  conditionsCheckLocationApplicabilityDuplicatesAndConflicts();
  return 0;
}
