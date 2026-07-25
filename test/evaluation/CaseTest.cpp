#include "Evaluation/Case.h"
#include "Evaluation/CaseText.h"
#include "Evaluation/Metric.h"
#include "Evaluation/ModelDescriptor.h"
#include "Evaluation/Request.h"

#include "Common/Artifact.h"
#include "Fabric/ArtifactSchema.h"
#include "Mapping/ArtifactSchema.h"
#include "Simulator/ArtifactSchema.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

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
// A representative Artifact family
//===----------------------------------------------------------------------===//

// The family owns its schema descriptor, its closed local target kinds, and the
// structure and validity of their canonical payloads. A single-entity region
// reference and a two-entity channel reference show that Evaluation frames a
// family-owned payload whole rather than assuming one entity shape.
const ArtifactSchemaDescriptor representativeSchema{
    "loom.test.evaluation_owner", SchemaVersion{1, 0}};

constexpr LocalTargetKind representativeRegionKind{0};
constexpr LocalTargetKind representativeChannelKind{1};
constexpr std::uint64_t representativeRegionCount = 64;

struct RegionIdTag;
struct ChannelEndIdTag;

template <typename Tag> class RepresentativeEntityId {
public:
  explicit constexpr RepresentativeEntityId(std::uint64_t value)
      : value_(value) {}

  constexpr std::uint64_t value() const { return value_; }

private:
  std::uint64_t value_;
};

using RegionId = RepresentativeEntityId<RegionIdTag>;
using ChannelEndId = RepresentativeEntityId<ChannelEndIdTag>;

llvm::StringRef representativeLocalKindSpelling(LocalTargetKind kind) {
  if (kind == representativeRegionKind)
    return "region";
  if (kind == representativeChannelKind)
    return "channel";
  return llvm::StringRef();
}

llvm::Error representativeError(const std::string &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

llvm::Error
representativeValidateLocalTarget(LocalTargetKind kind,
                                  const LocalTargetPayload &payload) {
  llvm::ArrayRef<std::uint8_t> bytes = payload.bytes();
  if (kind == representativeRegionKind) {
    if (bytes.size() != 8)
      return representativeError("a region reference is one entity ordinal");
    std::uint64_t region = 0;
    for (std::uint8_t byte : bytes)
      region = (region << 8) | byte;
    if (region >= representativeRegionCount)
      return representativeError("region ordinal must be less than 64");
    return llvm::Error::success();
  }
  if (kind == representativeChannelKind) {
    if (bytes.size() != 16)
      return representativeError("a channel reference is two entity ordinals");
    return llvm::Error::success();
  }
  return representativeError("unknown representative local target kind");
}

const LocalTargetFamilyDescriptor representativeFamily{
    &representativeSchema, &representativeLocalKindSpelling,
    &representativeValidateLocalTarget};

//===----------------------------------------------------------------------===//
// One exact MappedWorkloadExecution case
//===----------------------------------------------------------------------===//

ArtifactIdentity fabricArtifact() { return testArtifact({0x11}); }
ArtifactIdentity mappingArtifact() { return testArtifact({0x22}); }
ArtifactIdentity workloadArtifact() { return testArtifact({0x33}); }
ArtifactIdentity runtimeInputArtifact() { return testArtifact({0x44}); }
ArtifactIdentity ownedArtifact() { return testArtifact({0x55}); }
ArtifactIdentity foreignArtifact() { return testArtifact({0x66}); }
ArtifactIdentity otherFabricArtifact() { return testArtifact({0x77}); }

CaseSubjectRoleRef fabricRole() { return CaseSubjectRoleRef(0); }
CaseSubjectRoleRef mappingRole() { return CaseSubjectRoleRef(1); }

CaseArtifactResolution::Entry resolved(ArtifactIdentity artifact,
                                       const ArtifactSchemaDescriptor &schema,
                                       std::vector<ArtifactIdentity> closure) {
  return CaseArtifactResolution::Entry{std::move(artifact), &schema,
                                       std::move(closure)};
}

/// The exact bound Artifacts of the case, as an Artifact store resolves them.
/// The Mapping subject depends on the Fabric subject and the runtime input
/// depends on the workload, exactly as those families require.
CaseArtifactResolution caseResolution(const char *test) {
  return takeExpected(
      test, CaseArtifactResolution::get(
                {resolved(fabricArtifact(), fabric::artifactSchema,
                          {ownedArtifact()}),
                 resolved(mappingArtifact(), loom::mapping::artifactSchema,
                          {fabricArtifact()}),
                 resolved(workloadArtifact(), loom::sim::workloadSchema, {}),
                 resolved(runtimeInputArtifact(), loom::sim::runtimeInputSchema,
                          {workloadArtifact()}),
                 resolved(ownedArtifact(), representativeSchema, {}),
                 resolved(otherFabricArtifact(), fabric::artifactSchema, {}),
                 resolved(foreignArtifact(), representativeSchema, {})}));
}

EvaluationSubjectBindings caseBindings(const char *test,
                                       ArtifactIdentity fabric,
                                       ArtifactIdentity mapping) {
  return takeExpected(test, EvaluationSubjectBindings::get(
                                {{fabricRole(), {std::move(fabric)}},
                                 {mappingRole(), {std::move(mapping)}}}));
}

LocalTargetRef regionTarget(const char *test, std::uint64_t region) {
  return takeExpected(
      test, LocalTargetRef::get(
                representativeFamily, representativeRegionKind, ownedArtifact(),
                LocalTargetPayload::ofEntities(RegionId(region))));
}

LocalTargetRef channelTarget(const char *test, std::uint64_t source,
                             std::uint64_t sink) {
  return takeExpected(
      test, LocalTargetRef::get(representativeFamily, representativeChannelKind,
                                ownedArtifact(),
                                LocalTargetPayload::ofEntities(
                                    ChannelEndId(source), ChannelEndId(sink))));
}

SubjectTargetRef fabricRootTarget() {
  return SubjectTargetRef{fabricRole(), fabricArtifact(),
                          ArtifactRootTarget{fabricArtifact()}};
}

SubjectTargetRef mappingRootTarget() {
  return SubjectTargetRef{mappingRole(), mappingArtifact(),
                          ArtifactRootTarget{mappingArtifact()}};
}

SubjectTargetRef regionScopeTarget(const char *test, std::uint64_t region) {
  return SubjectTargetRef{fabricRole(), fabricArtifact(),
                          regionTarget(test, region)};
}

EvaluationCondition clockPeriodCondition(const char *test,
                                         const SubjectTargetRef &clockDomain,
                                         std::int64_t coefficient) {
  return EvaluationCondition{RequiredClockPeriodCondition{
      clockDomain, decimal(test, coefficient, -10)}};
}

EvaluationCase mappedCase(const char *test,
                          llvm::ArrayRef<EvaluationCondition> baseConditions) {
  return takeExpected(
      test,
      EvaluationCase::get(
          takeExpected(test, EvaluationCaseSignatureRef::get(
                                 evaluationSchemaVersion(),
                                 EvaluationCaseKind::MappedWorkloadExecution)),
          caseBindings(test, fabricArtifact(), mappingArtifact()),
          workloadArtifact(), runtimeInputArtifact(), baseConditions,
          caseResolution(test)));
}

//===----------------------------------------------------------------------===//
// Query-owned scope forms
//===----------------------------------------------------------------------===//

const AcceptedLocalTarget regionTargets[] = {
    {&representativeFamily, representativeRegionKind}};
const AcceptedLocalTarget channelTargets[] = {
    {&representativeFamily, representativeChannelKind}};

const ScopeRoleDescriptor subjectRoles[] = {
    {ScopeRoleRef(0), "subject", true, {}}};
const ScopeRoleDescriptor regionRoles[] = {
    {ScopeRoleRef(0), "region", false, regionTargets}};
const ScopeRoleDescriptor relationRoles[] = {
    {ScopeRoleRef(0), "source", false, regionTargets},
    {ScopeRoleRef(1), "sink", false, regionTargets}};
const ScopeRoleDescriptor repeatedRoles[] = {
    {ScopeRoleRef(0), "endpoint", true, {}},
    {ScopeRoleRef(0), "endpoint", true, {}}};

llvm::Error verifyDistinctEndpoints(llvm::ArrayRef<SubjectTargetRef> targets) {
  if (targets[0] == targets[1])
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "a transfer relation requires distinct "
                                   "endpoints");
  return llvm::Error::success();
}

const ScopeFormDescriptor testScopeForms[] = {
    {ScopeFormRef(0), "the entire exact Evaluation case", {}, nullptr},
    {ScopeFormRef(1), "one exact case subject Artifact root", subjectRoles,
     nullptr},
    {ScopeFormRef(2), "one family-owned region", regionRoles, nullptr},
    {ScopeFormRef(3), "an ordered source-to-sink transfer", relationRoles,
     &verifyDistinctEndpoints},
    {ScopeFormRef(4), "a malformed repeated role tuple", repeatedRoles,
     nullptr},
};

const ScopeRoleDescriptor channelRoles[] = {
    {ScopeRoleRef(0), "channel", false, channelTargets}};

const ScopeFormDescriptor channelScopeForms[] = {
    {ScopeFormRef(0), "one family-owned channel", channelRoles, nullptr}};

std::string scopeJson(const EvaluationScope &scope) {
  llvm::SmallString<256> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  writeEvaluationScopeJson(json, scope);
  return output.str().str();
}

llvm::Expected<EvaluationScope>
parseScopeJson(llvm::StringRef text,
               llvm::ArrayRef<ScopeFormDescriptor> forms) {
  llvm::Expected<llvm::json::Value> value = llvm::json::parse(text);
  if (!value)
    return value.takeError();
  const llvm::json::Object *object = value->getAsObject();
  if (!object)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "scope must be an object");
  return parseEvaluationScopeJson(*object, forms);
}

//===----------------------------------------------------------------------===//
// Anchors
//===----------------------------------------------------------------------===//

void sharedSignatureAlignsCaseKeysAcrossModelDescriptors() {
  const EvaluationModelDescriptor &analytical =
      modelDescriptor(EvaluationModelKind::AnalyticalTimingModel);
  const EvaluationModelDescriptor &simulator =
      modelDescriptor(EvaluationModelKind::CycleAccurateSimulator);

  // Each descriptor holds one exact versioned case-signature reference.
  require(__func__, analytical.caseSignature == simulator.caseSignature,
          "the two descriptors no longer share one exact case signature");
  require(__func__,
          analytical.caseSignature.schemaVersion() == evaluationSchemaVersion(),
          "the stored case signature lost its exact schema version");
  require(__func__,
          analytical.implementationSemanticIdentity !=
              simulator.implementationSemanticIdentity,
          "distinct model descriptors lost their distinct model identity");

  const std::vector<EvaluationCondition> conditions = {
      clockPeriodCondition(__func__, fabricRootTarget(), 8)};
  EvaluationCase evaluationCase = mappedCase(__func__, conditions);
  EvaluationCase sameFacts = mappedCase(__func__, conditions);
  require(__func__, baseCaseKey(evaluationCase) == baseCaseKey(sameFacts),
          "identical case facts produced different base case keys");

  MetricQuery cycles{MetricKind::CycleCount,
                     EvaluationScope{ScopeFormRef(1), {fabricRootTarget()}}};
  const std::vector<EvaluationCondition> quantile = {
      EvaluationCondition{QuantileCondition{ratio(__func__, 95, 100)}}};
  CaseArtifactResolution resolution = caseResolution(__func__);
  MetricRequest request =
      takeExpected(__func__, MetricRequest::get(cycles, quantile,
                                                evaluationCase, resolution));
  MetricRequest sameRequest = takeExpected(
      __func__, MetricRequest::get(cycles, quantile, sameFacts, resolution));
  require(__func__,
          metricCaseKey(evaluationCase, request) ==
              metricCaseKey(sameFacts, sameRequest),
          "identical metric case facts produced different keys");
  require(__func__,
          baseCaseKey(evaluationCase) != metricCaseKey(evaluationCase, request),
          "base and metric case keys share one domain");

  MetricRequest medianRequest = takeExpected(
      __func__,
      MetricRequest::get(
          cycles,
          {EvaluationCondition{QuantileCondition{ratio(__func__, 1, 2)}}},
          evaluationCase, resolution));
  require(__func__,
          metricCaseKey(evaluationCase, request) !=
              metricCaseKey(evaluationCase, medianRequest),
          "request-specific conditions do not reach the metric case key");

  EvaluationCase withoutRuntimeInput = takeExpected(
      __func__, EvaluationCase::get(
                    analytical.caseSignature,
                    caseBindings(__func__, fabricArtifact(), mappingArtifact()),
                    workloadArtifact(), std::nullopt, conditions, resolution));
  require(__func__,
          baseCaseKey(evaluationCase) != baseCaseKey(withoutRuntimeInput),
          "distinct case facts produced one base case key");

  // Model capability covers the exact requested outputs, not just the case.
  if (llvm::Error error =
          validateModelCapability(analytical, evaluationCase, request))
    fail(__func__, llvm::toString(std::move(error)));
  expectErrorContains(
      __func__, validateModelCapability(simulator, evaluationCase, request),
      "does not recognize condition");
  expectErrorContains(
      __func__,
      validateModelCapability(
          simulator, evaluationCase,
          takeExpected(__func__,
                       MetricRequest::get(
                           MetricQuery{MetricKind::ClockPeriod,
                                       EvaluationScope{ScopeFormRef(0), {}}},
                           {}, evaluationCase, resolution))),
      "does not support metric");
  expectErrorContains(__func__,
                      validateModelCapability(simulator,
                                              mappedCase(__func__, {}),
                                              llvm::ArrayRef<MetricRequest>{}),
                      "requires condition");
}

void caseSignatureOwnsSchemasAndCompatibility() {
  EvaluationCaseSignatureRef signature =
      takeExpected(__func__, EvaluationCaseSignatureRef::get(
                                 evaluationSchemaVersion(),
                                 EvaluationCaseKind::MappedWorkloadExecution));
  CaseArtifactResolution resolution = caseResolution(__func__);

  // The signature owns accepted schemas per subject role.
  expectErrorContains(
      __func__,
      EvaluationCase::get(
          signature,
          caseBindings(__func__, mappingArtifact(), mappingArtifact()),
          workloadArtifact(), std::nullopt, {}, resolution),
      "does not accept schema");
  expectErrorContains(
      __func__,
      EvaluationCase::get(
          signature,
          caseBindings(__func__, fabricArtifact(), foreignArtifact()),
          workloadArtifact(), std::nullopt, {}, resolution),
      "does not accept schema");

  // An unresolved subject is never silently accepted.
  expectErrorContains(
      __func__,
      EvaluationCase::get(
          signature,
          caseBindings(__func__, fabricArtifact(), testArtifact({0xfe})),
          workloadArtifact(), std::nullopt, {}, resolution),
      "unresolved");

  // The signature owns the workload and runtime-input requirements, their
  // accepted schemas, and their compatibility.
  expectErrorContains(
      __func__,
      EvaluationCase::get(
          signature,
          caseBindings(__func__, fabricArtifact(), mappingArtifact()),
          std::nullopt, std::nullopt, {}, resolution),
      "requires a workload");
  expectErrorContains(
      __func__,
      EvaluationCase::get(
          signature,
          caseBindings(__func__, fabricArtifact(), mappingArtifact()),
          fabricArtifact(), std::nullopt, {}, resolution),
      "does not accept schema");
  expectErrorContains(
      __func__,
      EvaluationCase::get(
          signature,
          caseBindings(__func__, fabricArtifact(), mappingArtifact()),
          workloadArtifact(), foreignArtifact(), {}, resolution),
      "does not accept schema");

  // Cross-role compatibility: a Mapping subject must bind the Fabric subject.
  expectErrorContains(
      __func__,
      EvaluationCase::get(
          signature,
          caseBindings(__func__, otherFabricArtifact(), mappingArtifact()),
          workloadArtifact(), std::nullopt, {}, resolution),
      "does not depend on");

  // Workload compatibility: a runtime input is for its exact workload.
  CaseArtifactResolution strayRuntimeInput = takeExpected(
      __func__,
      CaseArtifactResolution::get(
          {resolved(fabricArtifact(), fabric::artifactSchema, {}),
           resolved(mappingArtifact(), loom::mapping::artifactSchema,
                    {fabricArtifact()}),
           resolved(workloadArtifact(), loom::sim::workloadSchema, {}),
           resolved(runtimeInputArtifact(), loom::sim::runtimeInputSchema,
                    {})}));
  expectErrorContains(
      __func__,
      EvaluationCase::get(
          signature,
          caseBindings(__func__, fabricArtifact(), mappingArtifact()),
          workloadArtifact(), runtimeInputArtifact(), {}, strayRuntimeInput),
      "runtime input");

  // Role totality and cardinality stay with the signature.
  expectErrorContains(
      __func__,
      EvaluationCase::get(
          signature,
          takeExpected(__func__, EvaluationSubjectBindings::get(
                                     {{fabricRole(), {fabricArtifact()}}})),
          workloadArtifact(), std::nullopt, {}, resolution),
      "requires a binding for subject role");
  expectErrorContains(
      __func__,
      EvaluationCase::get(
          signature,
          takeExpected(
              __func__,
              EvaluationSubjectBindings::get(
                  {{fabricRole(), {fabricArtifact(), otherFabricArtifact()}},
                   {mappingRole(), {mappingArtifact()}}})),
          workloadArtifact(), std::nullopt, {}, resolution),
      "requires exactly one subject");
}

void scopeFormsOwnRolesTargetsAndRelations() {
  EvaluationScope wholeCase{ScopeFormRef(0), {}};
  EvaluationScope subjectRoot{ScopeFormRef(1), {fabricRootTarget()}};
  EvaluationScope region{ScopeFormRef(2), {regionScopeTarget(__func__, 3)}};
  for (const EvaluationScope &scope : {wholeCase, subjectRoot, region})
    if (llvm::Error error = validateEvaluationScopeForm(testScopeForms, scope))
      fail(__func__, llvm::toString(std::move(error)));

  expectErrorContains(__func__,
                      validateEvaluationScopeForm(
                          testScopeForms, EvaluationScope{ScopeFormRef(9), {}}),
                      "unknown scope form");
  expectErrorContains(__func__,
                      validateEvaluationScopeForm(
                          testScopeForms, EvaluationScope{ScopeFormRef(1), {}}),
                      "requires exactly 1 target");
  expectErrorContains(
      __func__,
      validateEvaluationScopeForm(
          testScopeForms,
          EvaluationScope{ScopeFormRef(0), {fabricRootTarget()}}),
      "requires exactly 0 targets");

  // A role accepts exactly the target kinds its descriptor declares.
  expectErrorContains(
      __func__,
      validateEvaluationScopeForm(
          testScopeForms,
          EvaluationScope{ScopeFormRef(1), {regionScopeTarget(__func__, 3)}}),
      "does not accept");
  expectErrorContains(
      __func__,
      validateEvaluationScopeForm(
          testScopeForms,
          EvaluationScope{ScopeFormRef(2), {fabricRootTarget()}}),
      "does not accept");
  expectErrorContains(
      __func__,
      validateEvaluationScopeForm(
          testScopeForms,
          EvaluationScope{ScopeFormRef(2),
                          {SubjectTargetRef{fabricRole(), fabricArtifact(),
                                            channelTarget(__func__, 1, 2)}}}),
      "does not accept");

  // The role tuple of a form is ordered and nonrepeating.
  expectErrorContains(
      __func__,
      validateEvaluationScopeForm(
          testScopeForms,
          EvaluationScope{ScopeFormRef(4),
                          {fabricRootTarget(), mappingRootTarget()}}),
      "ordered nonrepeating roles");

  // The query form owns its own relation verification.
  EvaluationScope transfer{
      ScopeFormRef(3),
      {regionScopeTarget(__func__, 1), regionScopeTarget(__func__, 2)}};
  if (llvm::Error error = validateEvaluationScopeForm(testScopeForms, transfer))
    fail(__func__, llvm::toString(std::move(error)));
  expectErrorContains(
      __func__,
      validateEvaluationScopeForm(
          testScopeForms, EvaluationScope{ScopeFormRef(3),
                                          {regionScopeTarget(__func__, 1),
                                           regionScopeTarget(__func__, 1)}}),
      "distinct endpoints");

  // Role order is part of the scope and of its canonical key.
  EvaluationScope swapped{
      ScopeFormRef(3),
      {regionScopeTarget(__func__, 2), regionScopeTarget(__func__, 1)}};
  require(__func__, transfer != swapped, "role order is not part of the scope");
  require(__func__, canonicalScopeKey(transfer) != canonicalScopeKey(swapped),
          "the canonical scope key is insensitive to role order");

  // The complete family-owned payload reaches the canonical key.
  EvaluationScope firstChannel{
      ScopeFormRef(0),
      {SubjectTargetRef{fabricRole(), fabricArtifact(),
                        channelTarget(__func__, 1, 2)}}};
  EvaluationScope secondChannel{
      ScopeFormRef(0),
      {SubjectTargetRef{fabricRole(), fabricArtifact(),
                        channelTarget(__func__, 1, 3)}}};
  require(__func__,
          canonicalScopeKey(firstChannel) != canonicalScopeKey(secondChannel),
          "the canonical key drops part of the family payload");

  // The family owns payload validity, so an invalid local reference never
  // reaches a scope.
  expectErrorContains(
      __func__,
      LocalTargetRef::get(representativeFamily, representativeRegionKind,
                          ownedArtifact(),
                          LocalTargetPayload::ofEntities(RegionId(64))),
      "less than 64");
  expectErrorContains(
      __func__,
      LocalTargetRef::get(representativeFamily, representativeChannelKind,
                          ownedArtifact(),
                          LocalTargetPayload::ofEntities(ChannelEndId(1))),
      "two entity ordinals");

  // Canonical text frames the complete payload and decodes through the family.
  const std::string channelText =
      R"({"form":0,"targets":[{"case_subject_role":0,"anchor":"1100000000000000000000000000000000000000000000000000000000000000","target":{"kind":"artifact_local","family":"loom.test.evaluation_owner","family_version":"1.0","local_kind":"channel","artifact":"5500000000000000000000000000000000000000000000000000000000000000","payload":"00000000000000010000000000000002"}}]})";
  require(__func__, scopeJson(firstChannel) == channelText,
          "canonical local-target text changed:\n" + scopeJson(firstChannel));
  require(
      __func__,
      takeExpected(__func__, parseScopeJson(channelText, channelScopeForms)) ==
          firstChannel,
      "the local-target scope did not round-trip");
  expectErrorContains(
      __func__,
      parseScopeJson(
          R"({"form":0,"targets":[{"case_subject_role":0,"anchor":"1100000000000000000000000000000000000000000000000000000000000000","target":{"kind":"artifact_local","family":"loom.test.evaluation_owner","family_version":"1.0","local_kind":"channel","artifact":"5500000000000000000000000000000000000000000000000000000000000000","payload":"0000000000000001"}}]})",
          channelScopeForms),
      "two entity ordinals");
}

void scopeCaseValidationOwnsAnchorsAndClosure() {
  EvaluationCase evaluationCase = mappedCase(__func__, {});
  CaseArtifactResolution resolution = caseResolution(__func__);
  CaseTargetContext context = evaluationCase.targetContext(resolution);

  // The anchor itself, its exact closure, and a family-owned local object in
  // that closure are all reachable.
  for (const SubjectTargetRef &target :
       {fabricRootTarget(),
        SubjectTargetRef{fabricRole(), fabricArtifact(),
                         ArtifactRootTarget{ownedArtifact()}},
        regionScopeTarget(__func__, 3)})
    if (llvm::Error error = validateSubjectTargetRef(target, context))
      fail(__func__, llvm::toString(std::move(error)));

  expectErrorContains(
      __func__,
      validateSubjectTargetRef(
          SubjectTargetRef{CaseSubjectRoleRef(7), fabricArtifact(),
                           ArtifactRootTarget{fabricArtifact()}},
          context),
      "is not a role of");
  expectErrorContains(
      __func__,
      validateSubjectTargetRef(
          SubjectTargetRef{fabricRole(), mappingArtifact(),
                           ArtifactRootTarget{mappingArtifact()}},
          context),
      "is not bound to case subject role");
  expectErrorContains(
      __func__,
      validateSubjectTargetRef(
          SubjectTargetRef{fabricRole(), fabricArtifact(),
                           ArtifactRootTarget{foreignArtifact()}},
          context),
      "not reachable");

  // A local target must name an Artifact of its own family.
  expectErrorContains(
      __func__,
      validateSubjectTargetRef(
          SubjectTargetRef{
              fabricRole(), fabricArtifact(),
              takeExpected(__func__,
                           LocalTargetRef::get(
                               representativeFamily, representativeRegionKind,
                               fabricArtifact(),
                               LocalTargetPayload::ofEntities(RegionId(1))))},
          context),
      "does not belong to family");

  // A scope validates every one of its targets.
  expectErrorContains(
      __func__,
      validateEvaluationScopeCase(
          EvaluationScope{
              ScopeFormRef(1),
              {SubjectTargetRef{fabricRole(), fabricArtifact(),
                                ArtifactRootTarget{foreignArtifact()}}}},
          context),
      "not reachable");
}

void conditionOwnershipDuplicatesAndConflicts() {
  CaseArtifactResolution resolution = caseResolution(__func__);
  EvaluationCase evaluationCase = mappedCase(__func__, {});
  EvaluationCaseSignatureRef signature = evaluationCase.signature();

  // The condition registry owns allowed locations.
  expectErrorContains(
      __func__,
      EvaluationCase::get(
          signature,
          caseBindings(__func__, fabricArtifact(), mappingArtifact()),
          workloadArtifact(), std::nullopt,
          {EvaluationCondition{QuantileCondition{ratio(__func__, 1, 2)}}},
          resolution),
      "not permitted in base conditions");
  expectErrorContains(
      __func__,
      MetricRequest::get(
          MetricQuery{MetricKind::CycleCount,
                      EvaluationScope{ScopeFormRef(0), {}}},
          {clockPeriodCondition(__func__, fabricRootTarget(), 8)},
          evaluationCase, resolution),
      "not permitted in metric-request conditions");

  // The case signature owns which base-condition kinds and target roles apply.
  expectErrorContains(
      __func__,
      EvaluationCase::get(
          signature,
          caseBindings(__func__, fabricArtifact(), mappingArtifact()),
          workloadArtifact(), std::nullopt,
          {EvaluationCondition{SupplyVoltageCondition{
              fabricRootTarget(), decimal(__func__, 9, -1)}}},
          resolution),
      "is not applicable");
  expectErrorContains(
      __func__,
      EvaluationCase::get(
          signature,
          caseBindings(__func__, fabricArtifact(), mappingArtifact()),
          workloadArtifact(), std::nullopt,
          {clockPeriodCondition(__func__, mappingRootTarget(), 8)}, resolution),
      "does not permit case subject role");

  // The Metric descriptor owns request-specific kinds.
  const std::vector<EvaluationCondition> quantile = {
      EvaluationCondition{QuantileCondition{ratio(__func__, 1, 2)}}};
  expectErrorContains(
      __func__,
      MetricRequest::get(MetricQuery{MetricKind::ClockPeriod,
                                     EvaluationScope{ScopeFormRef(0), {}}},
                         quantile, evaluationCase, resolution),
      "is not applicable");
  MetricRequest sampled = takeExpected(
      __func__,
      MetricRequest::get(MetricQuery{MetricKind::CycleCount,
                                     EvaluationScope{ScopeFormRef(0), {}}},
                         quantile, evaluationCase, resolution));
  require(__func__, sampled.conditions().size() == 1,
          "the metric request lost its request-specific condition");

  // Exact duplicates and assignment conflicts are both invalid, while distinct
  // assignments of one kind are legal and canonically ordered.
  const EvaluationCondition firstDomain =
      clockPeriodCondition(__func__, fabricRootTarget(), 8);
  const EvaluationCondition secondDomain =
      clockPeriodCondition(__func__, regionScopeTarget(__func__, 1), 4);
  const EvaluationCondition conflicting =
      clockPeriodCondition(__func__, fabricRootTarget(), 6);

  expectErrorContains(
      __func__,
      EvaluationCase::get(
          signature,
          caseBindings(__func__, fabricArtifact(), mappingArtifact()),
          workloadArtifact(), std::nullopt, {firstDomain, firstDomain},
          resolution),
      "duplicate evaluation condition");
  expectErrorContains(
      __func__,
      EvaluationCase::get(
          signature,
          caseBindings(__func__, fabricArtifact(), mappingArtifact()),
          workloadArtifact(), std::nullopt, {firstDomain, conflicting},
          resolution),
      "conflicting");

  EvaluationCase forward = mappedCase(__func__, {firstDomain, secondDomain});
  EvaluationCase reverse = mappedCase(__func__, {secondDomain, firstDomain});
  require(__func__, forward.baseConditions().size() == 2,
          "distinct condition assignments were collapsed");
  require(__func__, baseCaseKey(forward) == baseCaseKey(reverse),
          "authoring order reached the canonical condition set");
}

} // namespace

int main() {
  sharedSignatureAlignsCaseKeysAcrossModelDescriptors();
  caseSignatureOwnsSchemasAndCompatibility();
  scopeFormsOwnRolesTargetsAndRelations();
  scopeCaseValidationOwnsAnchorsAndClosure();
  conditionOwnershipDuplicatesAndConflicts();
  return 0;
}
