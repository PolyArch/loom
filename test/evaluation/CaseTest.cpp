#include "Evaluation/Case.h"
#include "Evaluation/Metric.h"
#include "Evaluation/ModelDescriptor.h"
#include "Evaluation/Request.h"

#include "Common/Artifact.h"
#include "Common/ArtifactLocalReference.h"
#include "Fabric/Identity/FabricLocalReference.h"
#include "Fabric/Identity/FabricRefImport.h"
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

void fail(const char *test, const std::string &message) {
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
  std::string message = llvm::toString(std::move(error));
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

//===----------------------------------------------------------------------===//
// Exact case artifacts and owner importer views
//===----------------------------------------------------------------------===//

ArtifactIdentity fabricArtifact() { return testArtifact({0x11}); }
ArtifactIdentity platformArtifact() { return testArtifact({0x22}); }
ArtifactIdentity otherPlatformArtifact() { return testArtifact({0x23}); }
ArtifactIdentity workloadArtifact() { return testArtifact({0x33}); }
ArtifactIdentity foreignArtifact() { return testArtifact({0x66}); }

std::uint32_t technologyCornerKind() {
  return platform::implementationPlatformLocalKind(
      platform::ImplementationPlatformLocalReferenceKind::TechnologyCorner);
}

constexpr fabric::FabricEntityId clockDomainEntity = 52;
constexpr fabric::FabricEntityId otherClockDomainEntity = 53;

/// One small finalized Fabric answering only from its own typed facts.
class TestFabricView : public fabric::FabricArtifactView {
public:
  explicit TestFabricView(ArtifactIdentity identity)
      : identity_(std::move(identity)) {}

  const ArtifactIdentity &identity() const override { return identity_; }
  fabric::FabricRootKind rootKind() const override {
    return fabric::FabricRootKind::System;
  }
  std::optional<fabric::FabricEntityKind>
  entityKind(fabric::FabricEntityId id) const override {
    if (id == clockDomainEntity || id == otherClockDomainEntity)
      return fabric::FabricEntityKind::HardwareDomain;
    return std::nullopt;
  }

  std::uint64_t transportEndpointCount(
      const fabric::FabricTransportEndpointOwnerRef &) const override {
    return 0;
  }
  std::uint64_t memoryEndpointCount(
      const fabric::FabricMemoryEndpointOwnerRef &) const override {
    return 0;
  }
  std::uint64_t
  inventorySize(const fabric::FabricInventoryOwnerRef &,
                fabric::FabricInventoryKind) const override {
    return 0;
  }
  std::optional<fabric::FabricFuNodeKind>
  fuNodeKind(const fabric::FabricInventoryOwnerRef &,
             fabric::FabricOrdinal) const override {
    return std::nullopt;
  }
  bool declaresLocalMemoryService(
      fabric::FabricMemoryOccurrenceRef) const override {
    return false;
  }
  std::optional<fabric::FabricMemoryEndpointRole>
  memoryEndpointRole(const fabric::FabricMemoryEndpointRef &) const override {
    return std::nullopt;
  }
  std::optional<fabric::FabricHardwareDomainKind>
  hardwareDomainKind(fabric::HardwareDomainRef domain) const override {
    if (domain.id() == clockDomainEntity)
      return fabric::FabricHardwareDomainKind::Clock;
    return std::nullopt;
  }
  std::optional<fabric::FabricFuTemplateRef>
  fuTemplateOf(fabric::FabricFuOccurrenceRef) const override {
    return std::nullopt;
  }
  bool hasPointConnection(
      const fabric::FabricTransportEndpointRef &,
      const fabric::FabricTransportEndpointRef &) const override {
    return false;
  }
  bool admitsTraversal(
      const fabric::FabricPhysicalTraversalRef &) const override {
    return false;
  }

private:
  ArtifactIdentity identity_;
};

/// One exact validated platform with a dense two-corner catalog.
class TestPlatformView : public platform::ImplementationPlatformView {
public:
  TestPlatformView(ArtifactIdentity identity, std::uint64_t cornerCount)
      : identity_(std::move(identity)), cornerCount_(cornerCount) {}

  const ArtifactIdentity &identity() const override { return identity_; }
  std::uint64_t technologyCornerCount() const override { return cornerCount_; }

private:
  ArtifactIdentity identity_;
  std::uint64_t cornerCount_;
};

TestFabricView fabricView(fabricArtifact());
TestPlatformView platformView(platformArtifact(), 2);
TestPlatformView otherPlatformView(otherPlatformArtifact(), 2);

ArtifactRootReference fabricSubject() {
  return ArtifactRootReference{fabric::fabricArtifactSchema, fabricArtifact()};
}
ArtifactRootReference platformSubject() {
  return ArtifactRootReference{platform::implementationPlatformSchema,
                               platformArtifact()};
}
ArtifactRootReference workloadReference() {
  return ArtifactRootReference{fabric::fabricArtifactSchema,
                               workloadArtifact()};
}

/// The exact bound Artifacts of the case, as an Artifact store resolves them.
/// The Fabric subject depends on the exact ImplementationPlatform it was built
/// for; the second platform stays outside that closure.
CaseArtifactResolution caseResolution(const char *test) {
  return takeExpected(
      test, CaseArtifactResolution::get(
                {{fabricSubject(), {platformSubject()}},
                 {platformSubject(), {}},
                 {ArtifactRootReference{platform::implementationPlatformSchema,
                                        otherPlatformArtifact()},
                  {}},
                 {workloadReference(), {}}}));
}

//===----------------------------------------------------------------------===//
// One test case signature over one exact Fabric subject
//===----------------------------------------------------------------------===//

constexpr EvaluationCaseKind testCaseKind{1};

EvaluationCaseSignatureRef testSignatureRef(const char *test) {
  return takeExpected(test, EvaluationCaseSignatureRef::get(
                                evaluationSchemaVersion(), testCaseKind));
}

CaseSubjectRoleRef fabricRole() { return CaseSubjectRoleRef(0); }

const ArtifactSchemaDescriptor *const acceptedFabricSchemas[] = {
    &fabric::fabricArtifactSchema};

const CaseSubjectRoleDescriptor testSubjectRoles[] = {
    {fabricRole(), "fabric", SubjectRoleCardinality::ExactlyOne,
     acceptedFabricSchemas, nullptr},
};

SubjectReferenceType fabricRootType() {
  return SubjectReferenceType{ArtifactRootType{fabric::fabricArtifactSchema}};
}
SubjectReferenceType platformRootType() {
  return SubjectReferenceType{
      ArtifactRootType{platform::implementationPlatformSchema}};
}
SubjectReferenceType fabricClockDomainType() {
  return SubjectReferenceType{ArtifactLocalType{
      {fabric::fabricArtifactSchema,
       fabric::fabricEntityLocalKind<fabric::FabricEntityKind::HardwareDomain>()}}};
}

const std::vector<ConditionApplicabilityPattern> testBasePatterns = {
    {EvaluationConditionKind::ProcessCorner,
     {llvm::cantFail(EvaluationCaseSignatureRef::get({1, 0}, testCaseKind)),
      {{fabricRole(), fabricRootType()}}}},
    {EvaluationConditionKind::RequiredClockPeriod,
     {llvm::cantFail(EvaluationCaseSignatureRef::get({1, 0}, testCaseKind)),
      {{fabricRole(), fabricRootType()}}}},
    {EvaluationConditionKind::RequiredClockPeriod,
     {llvm::cantFail(EvaluationCaseSignatureRef::get({1, 0}, testCaseKind)),
      {{fabricRole(), fabricClockDomainType()}}}},
    {EvaluationConditionKind::RelativeClockSchedule,
     {llvm::cantFail(EvaluationCaseSignatureRef::get({1, 0}, testCaseKind)),
      {{fabricRole(), fabricClockDomainType()},
       {fabricRole(), fabricClockDomainType()}}}},
};

const EvaluationCaseSignatureDescriptor testCaseSignature{
    testCaseKind,
    "test_fabric_case",
    "A test case signature over one exact Fabric subject.",
    testSubjectRoles,
    ArtifactRequirement::Optional,
    acceptedFabricSchemas,
    ArtifactRequirement::Forbidden,
    {},
    nullptr,
    testBasePatterns};

//===----------------------------------------------------------------------===//
// Case fixtures
//===----------------------------------------------------------------------===//

EvaluationSubjectBindings caseBindings(const char *test,
                                       ArtifactRootReference fabric) {
  return takeExpected(test, EvaluationSubjectBindings::get(
                                {{fabricRole(), {std::move(fabric)}}}));
}

EncodedArtifactLocalReference clockDomainLocal(std::uint64_t entity) {
  return fabric::encodeFabricEntityLocalReference(
      fabricArtifact(), fabric::HardwareDomainRef(entity));
}

SubjectTargetRef fabricRootTarget() {
  return SubjectTargetRef{fabricRole(), fabricSubject(), fabricSubject()};
}

SubjectTargetRef clockDomainTarget(std::uint64_t entity) {
  return SubjectTargetRef{fabricRole(), fabricSubject(),
                          clockDomainLocal(entity)};
}

EvaluationCondition clockPeriodCondition(const char *test,
                                         const SubjectTargetRef &clockDomain,
                                         std::int64_t coefficient) {
  return EvaluationCondition{RequiredClockPeriodCondition{
      clockDomain, decimal(test, coefficient, -10)}};
}

EvaluationCase testCase(const char *test,
                        llvm::ArrayRef<EvaluationCondition> baseConditions) {
  return takeExpected(
      test,
      EvaluationCase::get(testSignatureRef(test),
                          caseBindings(test, fabricSubject()),
                          workloadReference(), std::nullopt, baseConditions,
                          caseResolution(test)));
}

void expectBaseConditionRejected(const char *test,
                                 const EvaluationCondition &condition,
                                 llvm::StringRef expected) {
  expectErrorContains(test, EvaluationCase::get(testSignatureRef(test),
                                                caseBindings(test, fabricSubject()),
                                                workloadReference(), std::nullopt,
                                                {condition}, caseResolution(test)),
                      expected);
}

void expectCornerRejected(const char *test,
                          const platform::TechnologyCornerRef &corner,
                          llvm::StringRef expected) {
  expectBaseConditionRejected(
      test,
      EvaluationCondition{ProcessCornerCondition{fabricRootTarget(), corner}},
      expected);
}

void registerOwnerCodecsAndCheckLookupLifetime() {
  if (llvm::Error error =
          platform::registerImplementationPlatformLocalReferenceKinds())
    fail(__func__, llvm::toString(std::move(error)));
  if (llvm::Error error =
          platform::publishImplementationPlatformView(platformView))
    fail(__func__, llvm::toString(std::move(error)));
  if (llvm::Error error =
          platform::publishImplementationPlatformView(otherPlatformView))
    fail(__func__, llvm::toString(std::move(error)));

  const std::optional<ArtifactLocalReferenceCodec> earlyCodec =
      findArtifactLocalReferenceKind(platform::implementationPlatformSchema,
                                     technologyCornerKind());
  const std::optional<ArtifactSchemaDescriptor> earlySchema =
      findArtifactLocalReferenceSchema(
          platform::implementationPlatformSchema.identity,
          platform::implementationPlatformSchema.version);
  require(__func__, earlyCodec && earlySchema,
          "the ImplementationPlatform owner codec did not register");

  if (llvm::Error error = fabric::registerFabricLocalReferenceKinds())
    fail(__func__, llvm::toString(std::move(error)));

  const EncodedArtifactLocalReference corner =
      platform::encodeTechnologyCornerRef(platform::TechnologyCornerRef{
          platformArtifact(), platform::TechnologyCornerId(1)});
  if (llvm::Error error = earlyCodec->strictDecode(corner.payload))
    fail(__func__, llvm::toString(std::move(error)));
  if (llvm::Error error = earlyCodec->validate(corner))
    fail(__func__, llvm::toString(std::move(error)));
  require(__func__, *earlySchema == platform::implementationPlatformSchema,
          "later owner registration changed an earlier schema lookup value");
}

//===----------------------------------------------------------------------===//
// Query-owned scope forms
//===----------------------------------------------------------------------===//

const ScopeRoleDescriptor subjectFormRoles[] = {{ScopeRoleRef(0), "subject"}};
const ScopeRoleDescriptor clockFormRoles[] = {{ScopeRoleRef(0), "clock"}};
const ScopeRoleDescriptor scheduleFormRoles[] = {
    {ScopeRoleRef(0), "reference"}, {ScopeRoleRef(1), "dependent"}};

llvm::Error verifyDistinctClocks(llvm::ArrayRef<SubjectTargetRef> targets) {
  if (targets[0] == targets[1])
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "a clock schedule requires distinct clock domains");
  return llvm::Error::success();
}

const std::vector<OrderedTargetPattern> subjectFormPatterns = {
    {llvm::cantFail(EvaluationCaseSignatureRef::get({1, 0}, testCaseKind)),
     {{fabricRole(), fabricRootType()}}},
    {llvm::cantFail(EvaluationCaseSignatureRef::get({1, 0}, testCaseKind)),
     {{fabricRole(), platformRootType()}}},
};
const std::vector<OrderedTargetPattern> clockFormPatterns = {
    {llvm::cantFail(EvaluationCaseSignatureRef::get({1, 0}, testCaseKind)),
     {{fabricRole(), fabricClockDomainType()}}},
};
const std::vector<OrderedTargetPattern> scheduleFormPatterns = {
    {llvm::cantFail(EvaluationCaseSignatureRef::get({1, 0}, testCaseKind)),
     {{fabricRole(), fabricClockDomainType()},
      {fabricRole(), fabricClockDomainType()}}},
};

const ScopeFormDescriptor testScopeForms[] = {
    {ScopeFormRef(0), "the entire exact Evaluation case", {}, {}, nullptr},
    {ScopeFormRef(1), "one exact case Artifact root", subjectFormRoles,
     subjectFormPatterns, nullptr},
    {ScopeFormRef(2), "one exact clock domain", clockFormRoles,
     clockFormPatterns, nullptr},
    {ScopeFormRef(3), "an ordered reference-to-dependent clock relation",
     scheduleFormRoles, scheduleFormPatterns, &verifyDistinctClocks},
};

//===----------------------------------------------------------------------===//
// Anchors
//===----------------------------------------------------------------------===//

/// Two model descriptors that reference one exact shared case signature derive
/// one case key from identical case facts; the key never depends on the model.
void sharedSignatureDerivesOneCaseKeyAcrossModels() {
  EvaluationCaseSignatureRef signature = testSignatureRef(__func__);

  EvaluationModelDescriptor modelA{"loom.test.model_a", signature, {}, {}};
  EvaluationModelDescriptor modelB{"loom.test.model_b", signature, {}, {}};
  require(__func__, modelA.caseSignature == modelB.caseSignature,
          "the two descriptors no longer share one exact case signature");

  const std::vector<EvaluationCondition> conditions = {
      clockPeriodCondition(__func__, clockDomainTarget(clockDomainEntity), 8)};
  EvaluationCase evaluationCase = testCase(__func__, conditions);
  EvaluationCase sameFacts = testCase(__func__, conditions);
  require(__func__, baseCaseKey(evaluationCase) == baseCaseKey(sameFacts),
          "identical case facts produced different base case keys");
}

/// Scope targets: exact bound anchors, the anchor's dependency closure, typed
/// local targets through the owner codec, and significant role order.
void scopeAnchorsClosureTypedLocalTargetsAndRoleOrder() {
  EvaluationCase evaluationCase = testCase(__func__, {});
  CaseArtifactResolution resolution = caseResolution(__func__);
  CaseTargetContext context = evaluationCase.targetContext(resolution);

  // The anchor itself, an exact Artifact in its dependency closure, and a
  // family-owned local object in that closure are all reachable targets.
  EvaluationScope subjectRoot{ScopeFormRef(1), {fabricRootTarget()}};
  EvaluationScope platformRoot{
      ScopeFormRef(1),
      {SubjectTargetRef{fabricRole(), fabricSubject(), platformSubject()}}};
  EvaluationScope clock{ScopeFormRef(2),
                        {clockDomainTarget(clockDomainEntity)}};
  for (const EvaluationScope &scope : {subjectRoot, platformRoot, clock})
    if (llvm::Error error =
            validateEvaluationScopeCase(scope, testScopeForms, context))
      fail(__func__, llvm::toString(std::move(error)));

  // An unbound anchor and a target outside the anchor's exact dependency
  // closure are invalid.
  expectErrorContains(
      __func__,
      validateEvaluationScopeCase(
          EvaluationScope{ScopeFormRef(1),
                          {SubjectTargetRef{fabricRole(), workloadReference(),
                                            workloadReference()}}},
          testScopeForms, context),
      "is not bound to case subject role");
  expectErrorContains(
      __func__,
      validateEvaluationScopeCase(
          EvaluationScope{
              ScopeFormRef(1),
              {SubjectTargetRef{
                  fabricRole(), fabricSubject(),
                  ArtifactRootReference{fabric::fabricArtifactSchema,
                                        foreignArtifact()}}}},
          testScopeForms, context),
      "not reachable");

  const ArtifactRootReference unresolvedRoot{
      fabric::fabricArtifactSchema, foreignArtifact()};
  CaseArtifactResolution unresolvedResolution = takeExpected(
      __func__, CaseArtifactResolution::get(
                    {{fabricSubject(), {unresolvedRoot}}}));
  expectErrorContains(
      __func__,
      validateEvaluationScopeCase(
          EvaluationScope{ScopeFormRef(1),
                          {SubjectTargetRef{fabricRole(), fabricSubject(),
                                            unresolvedRoot}}},
          testScopeForms,
          evaluationCase.targetContext(unresolvedResolution)),
      "target artifact is unresolved");

  // The owner codec owns payload shape and kind; the owner validator owns
  // resolution inside the exact Fabric artifact.
  SubjectTargetRef wrongKindPayload{
      fabricRole(), fabricSubject(),
      EncodedArtifactLocalReference{
          fabricSubject(),
          fabric::fabricEntityLocalKind<
              fabric::FabricEntityKind::FabricPeOccurrence>(),
          clockDomainLocal(clockDomainEntity).payload}};
  expectErrorContains(
      __func__,
      validateEvaluationScopeCase(EvaluationScope{ScopeFormRef(2),
                                                  {wrongKindPayload}},
                                  testScopeForms, context),
      "wrong_entity_kind");
  // Role order is part of the scope and of its canonical key; the query form
  // owns its relation verification.
  EvaluationScope forward{ScopeFormRef(3),
                          {clockDomainTarget(clockDomainEntity),
                           clockDomainTarget(otherClockDomainEntity)}};
  EvaluationScope swapped{ScopeFormRef(3),
                          {clockDomainTarget(otherClockDomainEntity),
                           clockDomainTarget(clockDomainEntity)}};
  EvaluationScope ordered{ScopeFormRef(3),
                          {clockDomainTarget(clockDomainEntity),
                           clockDomainTarget(clockDomainEntity)}};
  require(__func__, forward != swapped, "role order is not part of the scope");
  require(__func__, canonicalScopeKey(forward) != canonicalScopeKey(swapped),
          "the canonical scope key is insensitive to role order");
  expectErrorContains(
      __func__,
      validateEvaluationScopeCase(ordered, testScopeForms, context),
      "distinct clock domains");

}

/// Conditions: kind-owned locations, exact ordered applicability patterns,
/// distinct semantic target validation, and duplicate/conflict rejection.
void conditionLocationApplicabilityDuplicatesAndConflicts() {
  EvaluationCaseSignatureRef signature = testSignatureRef(__func__);
  CaseArtifactResolution resolution = caseResolution(__func__);
  EvaluationCase evaluationCase = testCase(__func__, {});

  // The condition registry owns allowed locations.
  expectBaseConditionRejected(
      __func__,
      EvaluationCondition{QuantileCondition{ratio(__func__, 1, 2)}},
      "not permitted in base conditions");
  expectErrorContains(
      __func__,
      MetricRequest::get(
          MetricQuery{MetricKind::CycleCount, EvaluationScope{ScopeFormRef(0), {}}},
          {clockPeriodCondition(__func__, fabricRootTarget(), 8)},
          evaluationCase, resolution),
      "not permitted in metric-request conditions");

  // The case signature owns complete ordered applicability patterns: an
  // unlisted kind, and a listed kind whose exact target pattern does not
  // match, are both invalid.
  expectBaseConditionRejected(
      __func__,
      EvaluationCondition{SupplyVoltageCondition{fabricRootTarget(),
                                                 decimal(__func__, 9, -1)}},
      "is not applicable");
  expectBaseConditionRejected(
      __func__,
      EvaluationCondition{ProcessCornerCondition{
          clockDomainTarget(clockDomainEntity),
          platform::TechnologyCornerRef{platformArtifact(),
                                        platform::TechnologyCornerId(1)}}},
      "is not applicable");

  // Multiple patterns per kind: the root and the local clock patterns are
  // distinct complete alternatives of RequiredClockPeriod.
  const EvaluationCondition rootClock =
      clockPeriodCondition(__func__, fabricRootTarget(), 8);
  const EvaluationCondition localClock =
      clockPeriodCondition(__func__, clockDomainTarget(clockDomainEntity), 4);
  EvaluationCase withClocks = testCase(__func__, {rootClock, localClock});
  require(__func__, withClocks.baseConditions().size() == 2,
          "distinct condition assignments were collapsed");

  // Distinct semantic targets: a relative schedule requires two distinct
  // clock domains.
  const ExactRatio half = ratio(__func__, 1, 2);
  EvaluationCondition schedule{RelativeClockScheduleCondition{
      clockDomainTarget(clockDomainEntity),
      clockDomainTarget(otherClockDomainEntity), half, ratio(__func__, 1, 4)}};
  EvaluationCase withSchedule = testCase(__func__, {schedule});
  require(__func__, withSchedule.baseConditions().size() == 1,
          "the exact relative clock schedule was not preserved");
  expectBaseConditionRejected(
      __func__,
      EvaluationCondition{RelativeClockScheduleCondition{
          clockDomainTarget(clockDomainEntity),
          clockDomainTarget(clockDomainEntity), half, ratio(__func__, 0, 1)}},
      "distinct clock domains");
  // Exact duplicates and assignment conflicts are both invalid.
  expectErrorContains(
      __func__,
      EvaluationCase::get(signature, caseBindings(__func__, fabricSubject()),
                          workloadReference(), std::nullopt,
                          {rootClock, rootClock}, resolution),
      "duplicate evaluation condition");
  expectErrorContains(
      __func__,
      EvaluationCase::get(
          signature, caseBindings(__func__, fabricSubject()),
          workloadReference(), std::nullopt,
          {rootClock, clockPeriodCondition(__func__, fabricRootTarget(), 6)},
          resolution),
      "conflicting");
}

/// One exact TechnologyCorner import through the ImplementationPlatform owner
/// codec, plus wrong-platform, wrong-kind, malformed, and unresolved
/// rejection. The corner's platform must also be admitted by the selected
/// subject's exact dependency closure.
void technologyCornerImportIsOwnerValidated() {
  EvaluationCaseSignatureRef signature = testSignatureRef(__func__);
  CaseArtifactResolution resolution = caseResolution(__func__);

  // The fixed eight-byte known vector and typed round-trip through the owner
  // codec.
  const platform::TechnologyCornerRef cornerOne{platformArtifact(),
                                                platform::TechnologyCornerId(1)};
  const std::array<std::uint8_t, 8> knownPayload =
      platform::encodeTechnologyCornerPayload(platform::TechnologyCornerId(1));
  require(__func__,
          knownPayload == std::array<std::uint8_t, 8>{0, 0, 0, 0, 0, 0, 0, 1},
          "the technology corner payload is not exactly u64be(corner_id)");
  require(__func__,
          takeExpected(__func__, platform::decodeTechnologyCornerPayload(
                                     knownPayload)) ==
              platform::TechnologyCornerId(1),
          "the technology corner payload did not round-trip");
  EncodedArtifactLocalReference encodedCorner =
      platform::encodeTechnologyCornerRef(cornerOne);
  require(__func__,
          takeExpected(__func__,
                       platform::decodeTechnologyCornerRef(encodedCorner)) ==
              cornerOne,
          "the heterogeneous corner reference did not round-trip");

  // One exact import: the corner resolves inside the exact platform admitted
  // by the selected subject's dependency closure.
  EvaluationCondition processCorner{ProcessCornerCondition{
      fabricRootTarget(), cornerOne}};
  EvaluationCase withCorner = takeExpected(
      __func__,
      EvaluationCase::get(signature, caseBindings(__func__, fabricSubject()),
                          workloadReference(), std::nullopt, {processCorner},
                          resolution));
  require(__func__, withCorner.baseConditions().size() == 1,
          "the exact process corner condition was not preserved");

  EncodedArtifactLocalReference wrongPlatform = encodedCorner;
  wrongPlatform.artifact.schema = fabric::fabricArtifactSchema;
  expectErrorContains(__func__,
                      platform::decodeTechnologyCornerRef(wrongPlatform),
                      "loom.implementation_platform");
  EncodedArtifactLocalReference wrongKind = encodedCorner;
  wrongKind.ownerLocalKind = technologyCornerKind() + 1;
  expectErrorContains(__func__, platform::decodeTechnologyCornerRef(wrongKind),
                      "technology corner kind");
  EncodedArtifactLocalReference malformed = encodedCorner;
  malformed.payload.assign(4, 0);
  expectErrorContains(__func__, validateArtifactLocalReference(malformed),
                      "exactly eight bytes");
  expectCornerRejected(
      __func__,
      platform::TechnologyCornerRef{platformArtifact(),
                                    platform::TechnologyCornerId(5)},
      "does not resolve");

  // A platform outside the selected subject's exact closure is not admitted.
  expectCornerRejected(
      __func__,
      platform::TechnologyCornerRef{otherPlatformArtifact(),
                                    platform::TechnologyCornerId(1)},
      "not admitted");
}

} // namespace

int main() {
  registerOwnerCodecsAndCheckLookupLifetime();
  if (llvm::Error error = fabric::publishFabricImporterView(fabricView))
    fail("registration", llvm::toString(std::move(error)));
  if (llvm::Error error = registerEvaluationCaseSignature(testCaseSignature))
    fail("registration", llvm::toString(std::move(error)));

  sharedSignatureDerivesOneCaseKeyAcrossModels();
  scopeAnchorsClosureTypedLocalTargetsAndRoleOrder();
  conditionLocationApplicabilityDuplicatesAndConflicts();
  technologyCornerImportIsOwnerValidated();
  return 0;
}
