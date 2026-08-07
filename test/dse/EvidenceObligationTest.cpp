#include "DSE/EvidenceObligation.h"

#include "Common/ArtifactFinalizer.h"
#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace loom;
using namespace loom::dse;
using namespace loom::evaluation;

[[noreturn]] void fail(const std::string &message) {
  std::cerr << "DSE Evidence obligation test failure: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void requireErrorContains(llvm::Expected<T> value, llvm::StringRef needle) {
  if (value)
    fail("expected rejection containing '" + needle.str() + "'");
  const std::string message = llvm::toString(value.takeError());
  if (!llvm::StringRef(message).contains(needle))
    fail("unexpected rejection: " + message);
}

constexpr ArtifactSchemaDescriptor candidateSchema{
    "loom.test.dse_obligation.candidate", {1, 0}};
constexpr ArtifactSchemaDescriptor contextSchema{
    "loom.test.dse_obligation.context", {1, 0}};
constexpr EvaluationCaseKind caseKind(0x7ffe1000);
constexpr EvaluationModelKind modelKind(0x7ffe1000);
constexpr CaseSubjectRoleRef candidateRole(0);
constexpr CaseSubjectRoleRef contextRole(1);

const ArtifactSchemaDescriptor *const candidateSchemas[] = {&candidateSchema};
const ArtifactSchemaDescriptor *const contextSchemas[] = {&contextSchema};
const CaseSubjectRoleDescriptor subjectRoles[] = {
    {candidateRole, "candidate", SubjectRoleCardinality::ExactlyOne,
     candidateSchemas, nullptr},
    {contextRole, "context", SubjectRoleCardinality::OneOrMore, contextSchemas,
     nullptr},
};

EvaluationCaseSignatureRef signatureRef() {
  return llvm::cantFail(
      EvaluationCaseSignatureRef::get(evaluationSchemaVersion(), caseKind));
}

const EvaluationCaseSignatureDescriptor caseDescriptor{
    caseKind,
    "dse_obligation_test_case",
    "One dynamic candidate and one input-bound context collection.",
    subjectRoles,
    ArtifactRequirement::Forbidden,
    {},
    ArtifactRequirement::Forbidden,
    {},
    nullptr,
    AbstractCaseCycle{},
    {}};

struct EmptyConfig {};

llvm::ArrayRef<std::uint8_t> configSchemaBytes() {
  static constexpr llvm::StringLiteral descriptor =
      "loom.test.dse_obligation.model_config.1.0";
  return {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
          descriptor.size()};
}

llvm::Expected<OwnerValue> projectConfig(const ResolvedConfig &) {
  return OwnerValue::get(EmptyConfig{});
}

llvm::Expected<std::vector<std::uint8_t>> encodeConfig(const OwnerValue &view) {
  if (!view.getIf<EmptyConfig>())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unexpected model config type");
  return std::vector<std::uint8_t>{};
}

llvm::Expected<OwnerValue> adoptConfig(llvm::ArrayRef<std::uint8_t> bytes,
                                       const ComponentViewDigest &) {
  if (!bytes.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "model config is not empty");
  return OwnerValue::get(EmptyConfig{});
}

const ScopeFormRef metricForms[] = {ScopeFormRef(0)};
const MetricCapability metrics[] = {
    {MetricKind::Runtime, metricForms, allObservationFormsMask()},
};
const EvaluationModelDescriptor modelDescriptor{
    modelKind,
    "dse_obligation_test_model",
    "loom.test.dse_obligation.model.v1",
    signatureRef(),
    {},
    metrics,
    {},
    {},
    {},
    {configSchemaBytes(), &projectConfig, &encodeConfig, &adoptConfig},
    {},
    EvaluationExecutionMethod::Analytic,
    {},
    DeterminismContract::Deterministic,
    {},
    ProviderForm::InProcess};

ArtifactRootReference storeArtifact(const ArtifactStore &store,
                                    const ArtifactSchemaDescriptor &schema,
                                    std::uint8_t value) {
  CanonicalSemanticBytes bytes(std::vector<std::uint8_t>{value});
  const ArtifactIdentity identity = take(store.put(schema, bytes));
  return ArtifactRootReference{schema.identity.str(), schema.version, identity};
}

EvaluationRequest makePrototype(const ArtifactRootReference &candidate,
                                const ArtifactRootReference &context,
                                const CaseArtifactResolution &resolution,
                                const ArtifactStore &store) {
  EvaluationSubjectBindings bindings = take(EvaluationSubjectBindings::get(
      {{candidateRole, {candidate}}, {contextRole, {context}}}));
  EvaluationCase evaluationCase = take(
      EvaluationCase::get(signatureRef(), std::move(bindings), std::nullopt,
                          std::nullopt, {}, resolution, store));
  MetricRequest metric = take(MetricRequest::get(
      MetricQuery{MetricKind::Runtime, EvaluationScope{ScopeFormRef(0), {}}},
      {}, evaluationCase, resolution, store));
  ResolvedModelBinding model = take(ResolvedModelBinding::project(
      modelDescriptor.reference(), {}, defaultResolvedConfig()));
  return take(EvaluationRequest::get(evaluationCase, {metric}, {},
                                     std::move(model), 0, resolution, store));
}

void exerciseCanonicalTemplateAndInstantiation() {
  if (llvm::Error error = registerEvaluationCaseSignature(caseDescriptor))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = registerEvaluationModelDescriptor(modelDescriptor))
    fail(llvm::toString(std::move(error)));

  llvm::SmallString<128> directory;
  if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
          "loom-dse-obligation", directory))
    fail(error.message());
  ArtifactStore store(directory);
  const ArtifactRootReference candidateA =
      storeArtifact(store, candidateSchema, 0x11);
  const ArtifactRootReference candidateB =
      storeArtifact(store, candidateSchema, 0x22);
  const ArtifactRootReference contextA =
      storeArtifact(store, contextSchema, 0x33);
  const ArtifactRootReference contextB =
      storeArtifact(store, contextSchema, 0x44);
  const CaseArtifactResolution resolution = take(CaseArtifactResolution::get({
      {candidateA, {}},
      {candidateB, {}},
      {contextA, {}},
      {contextB, {}},
  }));

  const InputSubjectBinding contextInput{contextRole,
                                         EvidenceAcquisitionInputSlotRef(1)};
  EvidenceObligationTemplate first = take(EvidenceObligationTemplate::get(
      makePrototype(candidateA, contextA, resolution, store), candidateRole,
      {contextInput}));
  EvidenceObligationTemplate second = take(EvidenceObligationTemplate::get(
      makePrototype(candidateB, contextB, resolution, store), candidateRole,
      {contextInput}));
  if (first.canonicalBytes() != second.canonicalBytes())
    fail("dynamic subjects changed template identity");

  EvidenceObligationTemplate adopted =
      take(adoptEvidenceObligationTemplate(first.canonicalBytes()));
  EvaluationRequest request = take(instantiateEvidenceObligation(
      adopted, candidateB, {{EvidenceAcquisitionInputSlotRef(1), {contextB}}},
      0, resolution, store));
  if (request.subjectBindings().subjects(candidateRole).size() != 1 ||
      request.subjectBindings().subjects(candidateRole).front() != candidateB ||
      request.subjectBindings().subjects(contextRole).size() != 1 ||
      request.subjectBindings().subjects(contextRole).front() != contextB ||
      request.metricRequests().size() != 1)
    fail("template instantiation did not bind the exact candidate and input");

  requireErrorContains(instantiateEvidenceObligation(adopted, candidateB, {}, 0,
                                                     resolution, store),
                       "input slot 1");
  requireErrorContains(instantiateEvidenceObligation(
                           adopted, candidateB,
                           {{EvidenceAcquisitionInputSlotRef(1), {candidateA}}},
                           0, resolution, store),
                       "does not accept schema");
  requireErrorContains(instantiateEvidenceObligation(
                           adopted, candidateB,
                           {{EvidenceAcquisitionInputSlotRef(1), {contextB}},
                            {EvidenceAcquisitionInputSlotRef(2), {contextA}}},
                           0, resolution, store),
                       "unreferenced slot 2");
  requireErrorContains(
      EvidenceObligationTemplate::get(
          makePrototype(candidateA, contextA, resolution, store), candidateRole,
          {{candidateRole, EvidenceAcquisitionInputSlotRef(1)}}),
      "candidate role");

  std::vector<std::uint8_t> noncanonical(first.canonicalBytes().begin(),
                                         first.canonicalBytes().end());
  noncanonical.push_back(0);
  requireErrorContains(adoptEvidenceObligationTemplate(noncanonical),
                       "trailing bytes");
  if (std::error_code error = llvm::sys::fs::remove_directories(directory))
    fail(error.message());
}

} // namespace

int main() {
  exerciseCanonicalTemplateAndInstantiation();
  return 0;
}
