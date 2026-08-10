#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/ResolvedConfigView.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/StandardFindings.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

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
constexpr PromotionAcquisitionKind emptyContextAcquisitionKind(0x7fff4101);

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
const FindingCapability findingCapabilities[] = {
    {standard_findings::FunctionalMismatch, scopeForms,
     findingResultFormMask(FindingResultForm::Absent) |
         findingResultFormMask(FindingResultForm::Present)}};
const EvaluationModelDescriptor modelDescriptor{
    modelKind,
    "promotion_acquisition_execution_model",
    "loom.test.acquisition.model.v1",
    signatureRef(),
    {},
    metricCapabilities,
    findingCapabilities,
    {},
    {},
    {modelConfigSchema(), &projectModelConfig, &encodeModelConfig,
     &adoptModelConfig},
    {},
    EvaluationExecutionMethod::Analytic,
    {},
    DeterminismContract::Deterministic,
    {},
    ProviderForm::InProcess};

llvm::Expected<EvaluationModelResult> evaluate(const EvaluationRequest &request,
                                               const CaseArtifactResolution &,
                                               const ArtifactStore &store,
                                               const BlobStore &) {
  llvm::ArrayRef<ArtifactRootReference> candidates =
      request.subjectBindings().subjects(candidateRole);
  if (candidates.size() != 1 ||
      request.subjectBindings().subjects(contextRole).size() != 1)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "model received incomplete subjects");
  auto candidateBytes = store.get(candidates.front());
  if (!candidateBytes)
    return candidateBytes.takeError();
  if (candidateBytes->bytes().size() != 1)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "candidate bytes are malformed");
  const std::uint8_t value = candidateBytes->bytes().front();

  std::vector<MetricResult> metrics;
  metrics.reserve(request.metricRequests().size());
  for (const MetricRequest &metric : request.metricRequests()) {
    if (metric.query().metric != MetricKind::Runtime)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "model received an unknown metric");
    metrics.push_back(
        MetricResult{UncertaintyKind::ExactWithinModel,
                     PointObservation{take(DecimalValue::get(value, 0))},
                     {}});
  }
  std::vector<FindingResult> findings;
  findings.reserve(request.findingRequests().size());
  for (const FindingRequest &finding : request.findingRequests()) {
    if (finding.query().kind != standard_findings::FunctionalMismatch)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "model received an unknown finding");
    if (value == 0x11) {
      findings.push_back(FindingResult{PresentFinding{{FindingOccurrence::get(
          standard_findings::FunctionalMismatchOccurrence{})}}});
    } else {
      findings.push_back(FindingResult{AbsentFinding{}});
    }
  }
  return EvaluationModelResult{
      {}, CompletedEvidence{std::move(metrics), std::move(findings)}};
}

const EvaluationModelProvider modelProvider{
    modelDescriptor.reference(), EvaluationModelInProcessProvider{&evaluate}};

constexpr std::array<std::uint8_t, 4> acquisitionConfigSchema = {0x41, 0x43,
                                                                 0x51, 0x31};
constexpr std::array<PromotionAcquisitionInputSlotDescriptor, 2> inputSlots = {{
    {PromotionAcquisitionInputSlotRef(0), "candidates",
     PlanValueRole::CandidateSet, &candidateSchema,
     PlanValueCardinality::NonEmptySet},
    {PromotionAcquisitionInputSlotRef(1), "context",
     PlanValueRole::CandidateSet, &contextSchema,
     PlanValueCardinality::NonEmptySet},
}};

llvm::Error validateAcquisitionConfig(llvm::ArrayRef<std::uint8_t> bytes,
                                      const ComponentViewDigest &digest) {
  if (bytes.size() != 1 || bytes.front() > 0x08)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "acquisition config is not canonical");
  return validateComponentViewDigest(acquisitionConfigSchema, bytes, digest);
}

llvm::Expected<std::vector<EvidenceObligationTemplateRef>>
resolveEvidenceObligations(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != 1 || bytes.front() > 0x08)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "acquisition config is not canonical");
  if (bytes.front() == 0x01)
    return std::vector<EvidenceObligationTemplateRef>{};
  if (bytes.front() == 0x02)
    return std::vector<EvidenceObligationTemplateRef>{
        EvidenceObligationTemplateRef(0), EvidenceObligationTemplateRef(1)};
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

constexpr std::array<PromotionAcquisitionInputSlotDescriptor, 2>
    emptyContextInputSlots = {{
        {PromotionAcquisitionInputSlotRef(0), "candidates",
         PlanValueRole::CandidateSet, &candidateSchema,
         PlanValueCardinality::NonEmptySet},
        {PromotionAcquisitionInputSlotRef(1), "context",
         PlanValueRole::CandidateSet, &contextSchema,
         PlanValueCardinality::FiniteSet},
    }};

const PromotionAcquisitionDescriptor emptyContextAcquisitionDescriptor{
    emptyContextAcquisitionKind,
    "test.empty_context_evaluation",
    "loom.test.empty_context_acquisition.v1",
    emptyContextInputSlots,
    PromotionAcquisitionInputSlotRef(0),
    candidateRole,
    ResolvedDseConfigViewContract{acquisitionConfigSchema,
                                  validateAcquisitionConfig},
    &resolveEvidenceObligations,
};

std::array<std::uint64_t, 2> resolvedTaskCounts{};

llvm::Expected<PromotionAcquisitionResolutionOutcome>
resolveCases(const ResolvedPromotionAcquisitionBinding &binding,
             llvm::ArrayRef<PromotionAcquisitionInputBinding>,
             llvm::ArrayRef<PromotionEvidenceAcquisitionTask> tasks,
             const ArtifactStore &store) {
  std::vector<ResolvedPromotionEvidenceAcquisitionTask> resolved;
  resolved.reserve(tasks.size());
  for (const PromotionEvidenceAcquisitionTask &task : tasks) {
    if (!task.obligation || task.obligationTemplate.ordinal() > 1 ||
        task.inputBindings.size() != 1 ||
        task.inputBindings.front().slot != EvidenceAcquisitionInputSlotRef(1) ||
        task.inputBindings.front().artifacts.empty())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "central task projection is malformed");
    ++resolvedTaskCounts[task.obligationTemplate.ordinal()];
    auto candidateBytes = store.get(task.candidate);
    if (!candidateBytes)
      return candidateBytes.takeError();
    const bool taskLocalSelection =
        task.inputBindings.front().artifacts.size() > 1;
    const ArtifactRootReference *context = nullptr;
    if (taskLocalSelection) {
      const std::uint8_t expected =
          candidateBytes->bytes().front() == 0x11 ? 0x33 : 0x44;
      for (const ArtifactRootReference &candidateContext :
           task.inputBindings.front().artifacts) {
        auto contextBytes = store.get(candidateContext);
        if (!contextBytes)
          return contextBytes.takeError();
        if (contextBytes->bytes().front() == expected) {
          context = &candidateContext;
          break;
        }
      }
    } else {
      context = &task.inputBindings.front().artifacts.front();
    }
    if (!context)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "candidate context is unavailable");
    auto contextBytes = store.get(*context);
    if (!contextBytes)
      return contextBytes.takeError();
    std::optional<std::vector<EvidenceAcquisitionInputBinding>> selectedInputs;
    if (taskLocalSelection) {
      switch (binding.canonicalConfigBytes().front()) {
      case 0x04:
        selectedInputs = std::vector<EvidenceAcquisitionInputBinding>{
            {EvidenceAcquisitionInputSlotRef(1), {task.candidate}}};
        break;
      case 0x05:
        selectedInputs = std::vector<EvidenceAcquisitionInputBinding>{};
        break;
      case 0x06:
        selectedInputs = std::vector<EvidenceAcquisitionInputBinding>{
            {EvidenceAcquisitionInputSlotRef(0), {*context}}};
        break;
      case 0x07: {
        std::vector<ArtifactRootReference> reversed =
            task.inputBindings.front().artifacts;
        std::reverse(reversed.begin(), reversed.end());
        selectedInputs = std::vector<EvidenceAcquisitionInputBinding>{
            {EvidenceAcquisitionInputSlotRef(1), std::move(reversed)}};
        break;
      }
      case 0x08:
        selectedInputs = std::vector<EvidenceAcquisitionInputBinding>{
            {EvidenceAcquisitionInputSlotRef(1), {*context, *context}}};
        break;
      default:
        selectedInputs = std::vector<EvidenceAcquisitionInputBinding>{
            {EvidenceAcquisitionInputSlotRef(1), {*context}}};
        break;
      }
    }
    resolved.push_back({0,
                        std::make_shared<const CaseArtifactResolution>(
                            take(CaseArtifactResolution::get(
                                {{task.candidate, {}}, {*context, {}}}))),
                        std::move(selectedInputs)});
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
                                const ArtifactStore &store,
                                const BlobStore &blobs) {
  EvaluationSubjectBindings subjects = take(EvaluationSubjectBindings::get(
      {{candidateRole, {candidate}}, {contextRole, {context}}}));
  EvaluationCase evaluationCase = take(
      EvaluationCase::get(signatureRef(), std::move(subjects), std::nullopt,
                          std::nullopt, {}, resolution, store, blobs));
  MetricRequest metric = take(MetricRequest::get(
      {MetricKind::Runtime, EvaluationScope{ScopeFormRef(0), {}}}, {},
      evaluationCase, resolution, store));
  ResolvedModelBinding model = take(ResolvedModelBinding::project(
      modelDescriptor.reference(), {}, defaultResolvedConfig()));
  return take(EvaluationRequest::get(evaluationCase, {metric}, {},
                                     std::move(model), 0, resolution, store,
                                     blobs));
}

EvaluationRequest makeFindingPrototype(const ArtifactRootReference &candidate,
                                       const ArtifactRootReference &context,
                                       const CaseArtifactResolution &resolution,
                                       const ArtifactStore &store,
                                       const BlobStore &blobs) {
  EvaluationSubjectBindings subjects = take(EvaluationSubjectBindings::get(
      {{candidateRole, {candidate}}, {contextRole, {context}}}));
  EvaluationCase evaluationCase = take(
      EvaluationCase::get(signatureRef(), std::move(subjects), std::nullopt,
                          std::nullopt, {}, resolution, store, blobs));
  FindingRequest finding =
      take(FindingRequest::get({standard_findings::FunctionalMismatch,
                                EvaluationScope{ScopeFormRef(0), {}}},
                               {}, evaluationCase, resolution, store));
  ResolvedModelBinding model = take(ResolvedModelBinding::project(
      modelDescriptor.reference(), {}, defaultResolvedConfig()));
  return take(EvaluationRequest::get(evaluationCase, {}, {finding},
                                     std::move(model), 0, resolution, store,
                                     blobs));
}

void exactTemplateDrivesProductionAcquisition() {
  requireSuccess(standard_findings::registerStandardFindings());
  requireSuccess(registerEvaluationCaseSignature(caseDescriptor));
  requireSuccess(registerEvaluationModelDescriptor(modelDescriptor));
  requireSuccess(registerEvaluationModelProvider(modelProvider));
  requireSuccess(registerPromotionAcquisitionDescriptor(acquisitionDescriptor));
  requireSuccess(registerPromotionAcquisitionDescriptor(
      emptyContextAcquisitionDescriptor));
  requireSuccess(registerPromotionAcquisitionProvider(acquisitionProvider));

  llvm::SmallString<128> directory;
  if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
          "loom-promotion-acquisition", directory))
    fail(error.message());
  ArtifactStore store(directory);
  llvm::SmallString<128> blobPath(directory);
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const BlobStore blobs(blobPath);
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
      makePrototype(first, context, prototypeResolution, store, blobs),
      candidateRole, {{contextRole, EvidenceAcquisitionInputSlotRef(1)}}));

  const ComponentViewDigest cardinalityDigest = take(computeComponentViewDigest(
      acquisitionConfigSchema, llvm::ArrayRef<std::uint8_t>({0x00})));
  const std::vector<DsePlanNodeDefinition> emptyContextNodes = {
      PromotePlanNodeDefinition{
          emptyContextAcquisitionDescriptor.reference(),
          {ExactPlanArtifacts{candidates}, ExactPlanArtifacts{{context}}},
          {0x00},
          cardinalityDigest,
          QualityGatePolicyRef(0),
          AllPassingSelection{},
          PromotePurpose::CandidateSelection}};
  auto emptyContextPlan = ResolvedDsePlan::get(
      emptyContextNodes, {obligation}, {}, {take(QualityGatePolicy::get({}))});
  if (emptyContextPlan)
    fail("plan accepted a case role from a possibly empty input slot");
  requireErrorContains(emptyContextPlan.takeError(),
                       "slot cardinality do not match");

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
  DsePlanExecutionOutcome outcome = take(executeDsePlan(view, store, blobs));
  const auto *completed = std::get_if<CompletedDsePlanExecution>(&outcome);
  if (!completed || completed->resolve({0, 0}).size() != 2 ||
      completed->resolve({0, 1}).size() != 2)
    fail("production acquisition did not select candidates with Evidence");

  for (const ArtifactRootReference &evidenceRef : completed->resolve({0, 1})) {
    EvaluationEvidence evidence = take(importEvaluationEvidence(
        evidenceRef, prototypeResolution, store, blobs));
    EvaluationRequest request = take(importEvaluationRequest(
        evidence.requestRef(), prototypeResolution, store, blobs));
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

void topKAcquiresOnlyTheRequiredFunctionalPrefix() {
  llvm::SmallString<128> directory;
  if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
          "loom-staged-promotion-acquisition", directory))
    fail(error.message());
  ArtifactStore store(directory);
  llvm::SmallString<128> blobPath(directory);
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const BlobStore blobs(blobPath);
  const ArtifactRootReference first =
      storeArtifact(store, candidateSchema, 0x11);
  const ArtifactRootReference second =
      storeArtifact(store, candidateSchema, 0x22);
  const ArtifactRootReference third =
      storeArtifact(store, candidateSchema, 0x33);
  const ArtifactRootReference fourth =
      storeArtifact(store, candidateSchema, 0x44);
  const ArtifactRootReference context =
      storeArtifact(store, contextSchema, 0x55);
  std::vector<ArtifactRootReference> candidates{first, second, third, fourth};
  std::sort(candidates.begin(), candidates.end(), artifactRootReferenceLess);
  const CaseArtifactResolution resolution = take(CaseArtifactResolution::get(
      {{first, {}}, {second, {}}, {third, {}}, {fourth, {}}, {context, {}}}));
  EvidenceObligationTemplate runtime = take(EvidenceObligationTemplate::get(
      makePrototype(first, context, resolution, store, blobs), candidateRole,
      {{contextRole, EvidenceAcquisitionInputSlotRef(1)}}));
  EvidenceObligationTemplate functional = take(EvidenceObligationTemplate::get(
      makeFindingPrototype(first, context, resolution, store, blobs),
      candidateRole, {{contextRole, EvidenceAcquisitionInputSlotRef(1)}}));
  std::vector<EvidenceObligationTemplate> obligations;
  obligations.push_back(std::move(runtime));
  obligations.push_back(std::move(functional));
  llvm::sort(obligations, [](const EvidenceObligationTemplate &lhs,
                             const EvidenceObligationTemplate &rhs) {
    return std::lexicographical_compare(
        lhs.canonicalBytes().begin(), lhs.canonicalBytes().end(),
        rhs.canonicalBytes().begin(), rhs.canonicalBytes().end());
  });
  std::optional<std::uint32_t> runtimeOrdinal;
  std::optional<std::uint32_t> functionalOrdinal;
  for (std::uint32_t ordinal = 0; ordinal != obligations.size(); ++ordinal) {
    if (!obligations[ordinal].metricRequests().empty())
      runtimeOrdinal = ordinal;
    if (!obligations[ordinal].findingRequests().empty())
      functionalOrdinal = ordinal;
  }
  if (!runtimeOrdinal || !functionalOrdinal)
    fail("canonical obligation roles were not recovered");

  ResolvedObjectiveCatalogs objectives;
  objectives.dimensions = {
      {ResolvedEvaluationMetricObjectiveSource{*runtimeOrdinal, 0},
       ResolvedObjectiveDirection::Minimize, resolvedObjectiveDecimal(0, 0),
       resolvedObjectiveDecimal(1, 0), 0, 255}};
  objectives.weightedLevels = {{{{0, 1}}}};
  objectives.totalOrderings = {{{0}}};
  QualityGatePolicy gate = take(QualityGatePolicy::get(
      {{{FindingGate{*functionalOrdinal, FindingRequestOrdinal(0),
                     RequiredFindingState::Absent}}}}));
  const std::array<std::uint8_t, 1> configBytes = {0x02};
  const ComponentViewDigest configDigest =
      take(computeComponentViewDigest(acquisitionConfigSchema, configBytes));
  ResolvedConfig config = defaultResolvedConfig();
  config.dse.modelAuthorizations = {{modelDescriptor.reference()}};
  config.dse.evidenceObligationTemplates = std::move(obligations);
  config.dse.objectiveCatalogs = std::move(objectives);
  config.dse.qualityGatePolicies = {std::move(gate)};
  config.dse.planNodes = {PromotePlanNodeDefinition{
      acquisitionDescriptor.reference(),
      {ExactPlanArtifacts{candidates}, ExactPlanArtifacts{{context}}},
      {configBytes.begin(), configBytes.end()},
      configDigest,
      QualityGatePolicyRef(0),
      TopKSelection{0, 2},
      PromotePurpose::CandidateSelection}};

  resolvedTaskCounts = {};
  ResolvedDseConfigView view = take(projectResolvedDseConfigView(config));
  DsePlanExecutionOutcome outcome = take(executeDsePlan(view, store, blobs));
  const auto *completed = std::get_if<CompletedDsePlanExecution>(&outcome);
  if (!completed)
    fail("TopK execution did not complete");
  std::array<ArtifactRootReference, 2> expected = {second, third};
  std::sort(expected.begin(), expected.end(), artifactRootReferenceLess);
  if (completed->resolve({0, 0}) !=
      llvm::ArrayRef<ArtifactRootReference>(expected))
    fail("TopK did not refill its semantically valid candidate set");
  if (completed->resolve({0, 1}).size() != 7)
    fail("TopK retained evidence outside the required functional prefix");
  if (resolvedTaskCounts[*runtimeOrdinal] != 4 ||
      resolvedTaskCounts[*functionalOrdinal] != 3)
    fail("TopK did not acquire the deterministic minimal gate prefix");
  if (std::error_code error = llvm::sys::fs::remove_directories(directory))
    fail(error.message());
}

void taskLocalInputSelectionIsVerified() {
  llvm::SmallString<128> directory;
  if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
          "loom-task-local-acquisition-input", directory))
    fail(error.message());
  ArtifactStore store(directory);
  llvm::SmallString<128> blobPath(directory);
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const BlobStore blobs(blobPath);
  const ArtifactRootReference first =
      storeArtifact(store, candidateSchema, 0x11);
  const ArtifactRootReference second =
      storeArtifact(store, candidateSchema, 0x22);
  const ArtifactRootReference firstContext =
      storeArtifact(store, contextSchema, 0x33);
  const ArtifactRootReference secondContext =
      storeArtifact(store, contextSchema, 0x44);
  std::vector<ArtifactRootReference> candidates{first, second};
  std::vector<ArtifactRootReference> contexts{firstContext, secondContext};
  llvm::sort(candidates, artifactRootReferenceLess);
  llvm::sort(contexts, artifactRootReferenceLess);
  const CaseArtifactResolution resolution = take(CaseArtifactResolution::get(
      {{first, {}}, {second, {}}, {firstContext, {}}, {secondContext, {}}}));
  EvidenceObligationTemplate obligation = take(EvidenceObligationTemplate::get(
      makePrototype(first, firstContext, resolution, store, blobs),
      candidateRole, {{contextRole, EvidenceAcquisitionInputSlotRef(1)}}));

  const std::array<std::uint8_t, 1> configBytes = {0x03};
  const ComponentViewDigest configDigest =
      take(computeComponentViewDigest(acquisitionConfigSchema, configBytes));
  auto binding = take(ResolvedPromotionAcquisitionBinding::get(
      acquisitionDescriptor.reference(), configBytes, configDigest));
  const std::vector<PromotionAcquisitionInputBinding> inputs = {
      {PromotionAcquisitionInputSlotRef(0), candidates},
      {PromotionAcquisitionInputSlotRef(1), contexts},
  };
  const std::array<EvidenceObligationTemplateRef, 1> obligations = {
      EvidenceObligationTemplateRef(0)};
  auto outcome = take(invokePromotionAcquisition(
      inputs, binding, {obligation}, {candidates, obligations}, store, blobs));
  const auto *completed = std::get_if<CompletedPromotionAcquisition>(&outcome);
  if (!completed || completed->evidence.size() != 2)
    fail("task-local input selection did not acquire every candidate");
  for (const PromotionEvidence &evidence : completed->evidence) {
    const auto candidate =
        evidence.request.subjectBindings().subjects(candidateRole);
    const auto context =
        evidence.request.subjectBindings().subjects(contextRole);
    if (candidate.size() != 1 || context.size() != 1)
      fail("task-local input selection changed case cardinality");
    auto candidateBytes = take(store.get(candidate.front()));
    auto contextBytes = take(store.get(context.front()));
    const std::uint8_t expected =
        candidateBytes.bytes().front() == 0x11 ? 0x33 : 0x44;
    if (contextBytes.bytes().front() != expected)
      fail("task-local input selection crossed candidate lineage");
  }

  auto requireInvalidSelection = [&](std::uint8_t mode, llvm::StringRef error) {
    const std::array<std::uint8_t, 1> invalidConfigBytes = {mode};
    const ComponentViewDigest invalidDigest = take(computeComponentViewDigest(
        acquisitionConfigSchema, invalidConfigBytes));
    auto invalidBinding = take(ResolvedPromotionAcquisitionBinding::get(
        acquisitionDescriptor.reference(), invalidConfigBytes, invalidDigest));
    auto invalid =
        invokePromotionAcquisition(inputs, invalidBinding, {obligation},
                                   {candidates, obligations}, store, blobs);
    if (invalid)
      fail("provider malformed task input selection was accepted");
    requireErrorContains(invalid.takeError(), error);
  };
  requireInvalidSelection(0x04, "outside the bound task input");
  requireInvalidSelection(0x05, "did not select every bound task input slot");
  requireInvalidSelection(0x06, "changed a bound task input slot");
  requireInvalidSelection(0x07, "not canonical and unique");
  requireInvalidSelection(0x08, "not canonical and unique");
  if (std::error_code error = llvm::sys::fs::remove_directories(directory))
    fail(error.message());
}

} // namespace

int main() {
  exactTemplateDrivesProductionAcquisition();
  topKAcquiresOnlyTheRequiredFunctionalPrefix();
  taskLocalInputSelectionIsVerified();
  return 0;
}
