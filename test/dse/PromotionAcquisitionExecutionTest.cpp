#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/ResolvedConfigView.h"
#include "Evaluation/ModelProvider.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"

#include <algorithm>
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
using namespace loom::evaluation;

[[noreturn]] void fail(const std::string &message) {
  std::cerr << "promotion acquisition execution test failure: " << message
            << '\n';
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

void requireErrorContains(llvm::Error error, llvm::StringRef needle) {
  const std::string message = llvm::toString(std::move(error));
  if (!llvm::StringRef(message).contains(needle))
    fail("expected error containing '" + needle.str() + "', got: " + message);
}

constexpr ArtifactSchemaDescriptor candidateSchema{
    "loom.test.acquisition_candidate", SchemaVersion{1, 0}};
constexpr ArtifactSchemaDescriptor contextSchema{
    "loom.test.acquisition_context", SchemaVersion{1, 0}};
constexpr EvaluationCaseKind caseKind(0x7fff4100);
constexpr EvaluationModelKind modelKind(0x7fff4100);
constexpr CaseSubjectRoleRef candidateRole(0);
constexpr CaseSubjectRoleRef contextRole(1);
constexpr PromotionAcquisitionKind acquisitionKind(0x7fff4100);

EvaluationCaseSignatureRef signatureRef() {
  return llvm::cantFail(
      EvaluationCaseSignatureRef::get(evaluationSchemaVersion(), caseKind));
}

const ArtifactSchemaDescriptor *const candidateSchemas[] = {&candidateSchema};
const ArtifactSchemaDescriptor *const contextSchemas[] = {&contextSchema};
const CaseSubjectRoleDescriptor subjectRoles[] = {
    {candidateRole, "candidate", SubjectRoleCardinality::ExactlyOne,
     candidateSchemas, nullptr},
    {contextRole, "context", SubjectRoleCardinality::ExactlyOne, contextSchemas,
     nullptr},
};
const EvaluationCaseSignatureDescriptor caseDescriptor{
    caseKind,
    "promotion_acquisition_execution",
    "One exact candidate evaluated with one exact context.",
    subjectRoles,
    ArtifactRequirement::Forbidden,
    {},
    ArtifactRequirement::Forbidden,
    {},
    nullptr,
    AbsentReferenceCycle{},
    {}};

struct EmptyModelConfig final {};

llvm::ArrayRef<std::uint8_t> modelConfigSchema() {
  static constexpr llvm::StringLiteral descriptor =
      "loom.test.acquisition.model_config.1.0";
  return {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
          descriptor.size()};
}

llvm::Expected<OwnerValue> projectModelConfig(const ResolvedConfig &) {
  return OwnerValue::get(EmptyModelConfig{});
}

llvm::Expected<std::vector<std::uint8_t>>
encodeModelConfig(const OwnerValue &value) {
  if (!value.getIf<EmptyModelConfig>())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "model config has the wrong owner type");
  return std::vector<std::uint8_t>{};
}

llvm::Expected<OwnerValue> adoptModelConfig(llvm::ArrayRef<std::uint8_t> bytes,
                                            const ComponentViewDigest &) {
  if (!bytes.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "model config is not empty");
  return OwnerValue::get(EmptyModelConfig{});
}

const ScopeFormRef scopeForms[] = {ScopeFormRef(0)};
const MetricCapability metricCapabilities[] = {
    {MetricKind::Runtime, scopeForms,
     observationFormMask(ObservationForm::Point)}};
const EvaluationModelDescriptor modelDescriptor{
    modelKind,
    "promotion_acquisition_execution_model",
    "loom.test.acquisition.model.v1",
    signatureRef(),
    {},
    metricCapabilities,
    {},
    {},
    {},
    {modelConfigSchema(), &projectModelConfig, &encodeModelConfig,
     &adoptModelConfig},
    {},
    EvaluationExecutionMethod::Analytic,
    {},
    DeterminismContract::Deterministic,
    {}};

llvm::Expected<EvaluationModelResult> evaluate(const EvaluationRequest &request,
                                               const CaseArtifactResolution &,
                                               const ArtifactStore &) {
  if (request.subjectBindings().subjects(candidateRole).size() != 1 ||
      request.subjectBindings().subjects(contextRole).size() != 1)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "model received incomplete subjects");
  return EvaluationModelResult{
      {},
      CompletedEvidence{
          {MetricResult{UncertaintyKind::ExactWithinModel,
                        PointObservation{take(DecimalValue::get(17, 0))},
                        {}}},
          {}}};
}

const EvaluationModelProvider modelProvider{modelDescriptor.reference(),
                                            &evaluate};

constexpr std::array<std::uint8_t, 4> acquisitionConfigSchema = {0x41, 0x43,
                                                                 0x51, 0x31};
constexpr std::array<PromotionAcquisitionInputSlotDescriptor, 2> inputSlots = {{
    {PromotionAcquisitionInputSlotRef(0), "candidates",
     PlanValueRole::CandidateSet, &candidateSchema,
     PlanValueCardinality::NonEmptySet},
    {PromotionAcquisitionInputSlotRef(1), "context",
     PlanValueRole::CandidateSet, &contextSchema,
     PlanValueCardinality::ExactlyOne},
}};

llvm::Error validateAcquisitionConfig(llvm::ArrayRef<std::uint8_t> bytes,
                                      const ComponentViewDigest &digest) {
  if (bytes.size() != 1 || (bytes.front() != 0x00 && bytes.front() != 0x01))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "acquisition config is not canonical");
  return validateComponentViewDigest(acquisitionConfigSchema, bytes, digest);
}

llvm::Expected<std::vector<EvidenceObligationTemplateRef>>
resolveEvidenceObligations(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != 1 || (bytes.front() != 0x00 && bytes.front() != 0x01))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "acquisition config is not canonical");
  if (bytes.front() == 0x01)
    return std::vector<EvidenceObligationTemplateRef>{};
  return std::vector<EvidenceObligationTemplateRef>{
      EvidenceObligationTemplateRef(0)};
}

const PromotionAcquisitionDescriptor acquisitionDescriptor{
    acquisitionKind,
    "test.exact_evaluation",
    "loom.test.acquisition.v1",
    inputSlots,
    PromotionAcquisitionInputSlotRef(0),
    candidateRole,
    ResolvedDseConfigViewContract{acquisitionConfigSchema,
                                  validateAcquisitionConfig},
    &resolveEvidenceObligations,
};

llvm::Expected<PromotionAcquisitionResolutionOutcome>
resolveCases(const ResolvedPromotionAcquisitionBinding &,
             llvm::ArrayRef<PromotionAcquisitionInputBinding>,
             llvm::ArrayRef<PromotionEvidenceAcquisitionTask> tasks,
             const ArtifactStore &store) {
  std::vector<ResolvedPromotionEvidenceAcquisitionTask> resolved;
  resolved.reserve(tasks.size());
  for (const PromotionEvidenceAcquisitionTask &task : tasks) {
    if (!task.obligation || task.obligationTemplate.ordinal() != 0 ||
        task.inputBindings.size() != 1 ||
        task.inputBindings.front().slot != EvidenceAcquisitionInputSlotRef(1) ||
        task.inputBindings.front().artifacts.size() != 1)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "central task projection is malformed");
    const ArtifactRootReference &context =
        task.inputBindings.front().artifacts.front();
    auto candidateBytes = store.get(task.candidate);
    if (!candidateBytes)
      return candidateBytes.takeError();
    auto contextBytes = store.get(context);
    if (!contextBytes)
      return contextBytes.takeError();
    resolved.push_back({0, std::make_shared<const CaseArtifactResolution>(
                               take(CaseArtifactResolution::get(
                                   {{task.candidate, {}}, {context, {}}})))});
  }
  return PromotionAcquisitionResolutionOutcome{
      CompletedPromotionAcquisitionResolution{std::move(resolved)}};
}

const PromotionAcquisitionProvider acquisitionProvider{
    acquisitionDescriptor.reference(), &resolveCases};

ArtifactRootReference storeArtifact(const ArtifactStore &store,
                                    const ArtifactSchemaDescriptor &schema,
                                    std::uint8_t value) {
  const ArtifactIdentity identity = take(store.put(
      schema, CanonicalSemanticBytes(std::vector<std::uint8_t>{value})));
  return {schema.identity.str(), schema.version, identity};
}

EvaluationRequest makePrototype(const ArtifactRootReference &candidate,
                                const ArtifactRootReference &context,
                                const CaseArtifactResolution &resolution,
                                const ArtifactStore &store) {
  EvaluationSubjectBindings subjects = take(EvaluationSubjectBindings::get(
      {{candidateRole, {candidate}}, {contextRole, {context}}}));
  EvaluationCase evaluationCase = take(
      EvaluationCase::get(signatureRef(), std::move(subjects), std::nullopt,
                          std::nullopt, {}, resolution, store));
  MetricRequest metric = take(MetricRequest::get(
      {MetricKind::Runtime, EvaluationScope{ScopeFormRef(0), {}}}, {},
      evaluationCase, resolution, store));
  ResolvedModelBinding model = take(ResolvedModelBinding::project(
      modelDescriptor.reference(), {}, defaultResolvedConfig()));
  return take(EvaluationRequest::get(evaluationCase, {metric}, {},
                                     std::move(model), 0, resolution, store));
}

void exactTemplateDrivesProductionAcquisition() {
  requireSuccess(registerEvaluationCaseSignature(caseDescriptor));
  requireSuccess(registerEvaluationModelDescriptor(modelDescriptor));
  requireSuccess(registerEvaluationModelProvider(modelProvider));
  requireSuccess(registerPromotionAcquisitionDescriptor(acquisitionDescriptor));
  requireSuccess(registerPromotionAcquisitionProvider(acquisitionProvider));

  llvm::SmallString<128> directory;
  if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
          "loom-promotion-acquisition", directory))
    fail(error.message());
  ArtifactStore store(directory);
  const ArtifactRootReference first =
      storeArtifact(store, candidateSchema, 0x11);
  const ArtifactRootReference second =
      storeArtifact(store, candidateSchema, 0x22);
  const ArtifactRootReference context =
      storeArtifact(store, contextSchema, 0x33);
  std::vector<ArtifactRootReference> candidates{first, second};
  std::sort(candidates.begin(), candidates.end(), artifactRootReferenceLess);
  const CaseArtifactResolution prototypeResolution = take(
      CaseArtifactResolution::get({{first, {}}, {second, {}}, {context, {}}}));
  EvidenceObligationTemplate obligation = take(EvidenceObligationTemplate::get(
      makePrototype(first, context, prototypeResolution, store), candidateRole,
      {{contextRole, EvidenceAcquisitionInputSlotRef(1)}}));

  QualityGatePolicy requiredGate = take(QualityGatePolicy::get({{
      {MetricGate{0, MetricRequestOrdinal(0), MetricGateComparator::LE,
                  take(DecimalValue::get(20, 0))}},
  }}));
  const ComponentViewDigest missingDigest = take(computeComponentViewDigest(
      acquisitionConfigSchema, llvm::ArrayRef<std::uint8_t>({0x01})));
  const std::vector<DsePlanNodeDefinition> missingNodes = {
      PromotePlanNodeDefinition{
          acquisitionDescriptor.reference(),
          {ExactPlanArtifacts{candidates}, ExactPlanArtifacts{{context}}},
          {0x01},
          missingDigest,
          QualityGatePolicyRef(0),
          AllPassingSelection{},
          PromotePurpose::CandidateSelection}};
  auto missing =
      ResolvedDsePlan::get(missingNodes, {obligation}, {}, {requiredGate});
  if (missing)
    fail("plan accepted an acquisition policy missing required Evidence");
  requireErrorContains(missing.takeError(), "omits a required Evidence");

  const ComponentViewDigest configDigest = take(computeComponentViewDigest(
      acquisitionConfigSchema, llvm::ArrayRef<std::uint8_t>({0x00})));
  ResolvedConfig config = defaultResolvedConfig();
  config.dse.modelAuthorizations = {{modelDescriptor.reference()}};
  config.dse.evidenceObligationTemplates = {std::move(obligation)};
  config.dse.qualityGatePolicies = {take(QualityGatePolicy::get({}))};
  config.dse.planNodes = {PromotePlanNodeDefinition{
      acquisitionDescriptor.reference(),
      {ExactPlanArtifacts{candidates}, ExactPlanArtifacts{{context}}},
      {0x00},
      configDigest,
      QualityGatePolicyRef(0),
      AllPassingSelection{},
      PromotePurpose::CandidateSelection}};

  ResolvedDseConfigView view = take(projectResolvedDseConfigView(config));
  DsePlanExecutionOutcome outcome = take(executeDsePlan(view, store));
  const auto *completed = std::get_if<CompletedDsePlanExecution>(&outcome);
  if (!completed || completed->resolve({0, 0}).size() != 2 ||
      completed->resolve({0, 1}).size() != 2)
    fail("production acquisition did not select candidates with Evidence");

  for (const ArtifactRootReference &evidenceRef : completed->resolve({0, 1})) {
    EvaluationEvidence evidence =
        take(importEvaluationEvidence(evidenceRef, prototypeResolution, store));
    EvaluationRequest request = take(importEvaluationRequest(
        evidence.requestRef(), prototypeResolution, store));
    const llvm::ArrayRef<ArtifactRootReference> contexts =
        request.subjectBindings().subjects(contextRole);
    if (request.subjectBindings().subjects(candidateRole).size() != 1 ||
        contexts.size() != 1 || contexts.front() != context ||
        request.metricRequests().size() != 1 || request.replicateIndex() != 0)
      fail("production acquisition changed the exact template binding");
  }
  if (std::error_code error = llvm::sys::fs::remove_directories(directory))
    fail(error.message());
}

} // namespace

int main() {
  exactTemplateDrivesProductionAcquisition();
  return 0;
}
