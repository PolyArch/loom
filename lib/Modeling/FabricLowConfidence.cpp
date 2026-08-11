#include "Evaluation/Models/FabricLowConfidence.h"

#include "AnalyticModelSupport.h"

#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/ProductionRegistry.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "ImplementationPlatform/TechnologyCorner.h"

#include "llvm/ADT/STLExtras.h"

#include <array>
#include <optional>
#include <utility>
#include <vector>

namespace loom::evaluation::models {
namespace {

constexpr BuiltinEvaluationCase kCase =
    BuiltinEvaluationCase::FabricHardwareAnalysis;
constexpr BuiltinEvaluationModel kModel =
    BuiltinEvaluationModel::FabricLowConfidence;
constexpr CaseSubjectRoleRef kFabricRole(0);
constexpr ScopeFormRef kWholeCaseScope(0);

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_low_confidence_invalid: " + message);
}

EvaluationCaseSignatureRef caseSignatureRef() {
  return llvm::cantFail(EvaluationCaseSignatureRef::get(
      evaluationSchemaVersion(), builtinEvaluationCaseKind(kCase)));
}

const ArtifactSchemaDescriptor *const kFabricSchemas[] = {
    &fabric::fabricArtifactSchema};

const CaseSubjectRoleDescriptor kSubjectRoles[] = {
    {kFabricRole, "fabric", SubjectRoleCardinality::ExactlyOne, kFabricSchemas,
     nullptr},
};

SubjectTargetPattern fabricRootPattern() {
  return SubjectTargetPattern{
      kFabricRole,
      SubjectReferenceType{ArtifactRootType{fabric::fabricArtifactSchema}}};
}

const std::vector<ConditionApplicabilityPattern> kBaseConditionPatterns = {
    {EvaluationConditionKind::ProcessCorner,
     {caseSignatureRef(), {fabricRootPattern()}}},
    {EvaluationConditionKind::SupplyVoltage,
     {caseSignatureRef(), {fabricRootPattern()}}},
    {EvaluationConditionKind::Temperature,
     {caseSignatureRef(), {fabricRootPattern()}}},
    {EvaluationConditionKind::RequiredClockPeriod,
     {caseSignatureRef(), {fabricRootPattern()}}},
    {EvaluationConditionKind::RelativeClockSchedule,
     {caseSignatureRef(), {fabricRootPattern(), fabricRootPattern()}}},
    {EvaluationConditionKind::ActivityBinding,
     {caseSignatureRef(), {fabricRootPattern()}}},
    {EvaluationConditionKind::ActivityBinding,
     {caseSignatureRef(), {fabricRootPattern(), fabricRootPattern()}}},
};

const EvaluationCaseSignatureDescriptor kCaseSignature{
    builtinEvaluationCaseKind(kCase),
    "fabric_hardware_analysis",
    "One exact finalized Fabric analyzed without a software workload.",
    kSubjectRoles,
    ArtifactRequirement::Forbidden,
    {},
    ArtifactRequirement::Forbidden,
    {},
    nullptr,
    AbsentReferenceCycle{},
    kBaseConditionPatterns,
};

const std::vector<ModelConditionCapability> &conditionCapabilities() {
  static const std::vector<ModelConditionCapability> capabilities = [] {
    std::vector<ModelConditionCapability> result;
    result.reserve(kBaseConditionPatterns.size());
    for (const ConditionApplicabilityPattern &pattern :
         kBaseConditionPatterns) {
      const bool activity =
          pattern.kind == EvaluationConditionKind::ActivityBinding;
      result.push_back({pattern, activity ? ConditionDisposition::Consumed
                                          : ConditionDisposition::Invariant});
    }
    return result;
  }();
  return capabilities;
}

const ScopeFormRef kWholeCaseScopes[] = {kWholeCaseScope};
constexpr std::uint8_t kPoint = observationFormMask(ObservationForm::Point);
const MetricCapability kMetricCapabilities[] = {
    {MetricKind::LimitingClockFrequency, kWholeCaseScopes, kPoint},
    {MetricKind::TotalArea, kWholeCaseScopes, kPoint},
    {MetricKind::DynamicPower, kWholeCaseScopes, kPoint},
    {MetricKind::LeakagePower, kWholeCaseScopes, kPoint},
};
const ModeledPhenomenon kModeledPhenomena[] = {
    ModeledPhenomenon::SpatialResources,
    ModeledPhenomenon::PhysicalImplementation};

const EvaluationModelDescriptor &descriptor() {
  static const EvaluationModelDescriptor value{
      builtinEvaluationModelKind(kModel),
      "fabric_low_confidence",
      "loom.fabric.low_confidence.v2",
      caseSignatureRef(),
      conditionCapabilities(),
      kMetricCapabilities,
      {},
      {},
      {},
      detail::emptyLowConfidenceConfigView(),
      kModeledPhenomena,
      EvaluationExecutionMethod::Analytic,
      {},
      DeterminismContract::Deterministic,
      {},
      ProviderForm::InProcess};
  return value;
}

bool requestsDynamicPower(const EvaluationRequest &request) {
  return llvm::any_of(
      request.metricRequests(), [](const MetricRequest &metric) {
        return metric.query().metric == MetricKind::DynamicPower;
      });
}

llvm::Expected<EvaluationModelResult>
evaluate(const EvaluationRequest &request,
         const CaseArtifactResolution &resolution,
         const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  RequestVerifier verifier(resolution, artifactStore, blobStore);
  if (llvm::Error error = verifier.verify(request))
    return std::move(error);
  if (request.modelBinding().descriptorRef() != descriptor().reference())
    return invalid("request selects a foreign model descriptor");

  const auto fabrics = request.subjectBindings().subjects(kFabricRole);
  if (fabrics.size() != 1)
    return invalid("request does not bind exactly one Fabric");
  auto fabricRoot =
      fabric::importEntireFabricRoot(fabrics.front(), artifactStore);
  if (!fabricRoot)
    return fabricRoot.takeError();

  std::optional<detail::LowConfidencePhysicalActivity> activity;
  for (const EvaluationCondition &condition : request.baseConditions()) {
    const auto *binding =
        std::get_if<ActivityBindingCondition>(&condition.payload);
    if (!binding)
      continue;
    const auto *assumption =
        std::get_if<ExplicitAssumptionSource>(&binding->source);
    if (!assumption || activity)
      return EvaluationModelResult{
          {}, UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
    activity = detail::LowConfidencePhysicalActivity{
        assumption->staticProbability, assumption->transitionsPerClock};
  }
  if (requestsDynamicPower(request) && !activity)
    return EvaluationModelResult{
        {}, UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};

  auto metrics =
      detail::estimateLowConfidencePhysicalMetrics(*fabricRoot, activity);
  if (!metrics)
    return metrics.takeError();
  std::vector<MetricResult> results;
  results.reserve(request.metricRequests().size());
  for (const MetricRequest &metric : request.metricRequests()) {
    auto result = metrics->result(metric.query().metric);
    if (!result)
      return result.takeError();
    results.push_back(std::move(*result));
  }
  return EvaluationModelResult{{}, CompletedEvidence{std::move(results), {}}};
}

const EvaluationModelProvider &provider() {
  static const EvaluationModelProvider value{
      descriptor().reference(), EvaluationModelInProcessProvider{&evaluate}};
  return value;
}

llvm::Expected<CaseArtifactResolution>
resolveCase(const ArtifactRootReference &fabricReference,
            llvm::ArrayRef<EvaluationCondition> conditions,
            const ArtifactStore &artifactStore) {
  auto base = detail::resolveSingleSubjectFabricCase(
      fabricReference, fabricReference, artifactStore);
  if (!base)
    return base.takeError();
  std::vector<CaseArtifactResolution::Entry> entries(base->entries().begin(),
                                                     base->entries().end());
  auto fabricEntry = llvm::find_if(entries, [&](const auto &entry) {
    return entry.artifact == fabricReference;
  });
  if (fabricEntry == entries.end())
    return invalid("resolved Fabric closure omits its root");
  for (const EvaluationCondition &condition : conditions) {
    const auto *corner =
        std::get_if<ProcessCornerCondition>(&condition.payload);
    if (!corner)
      continue;
    const ArtifactRootReference platformReference{
        platform::implementationPlatformSchema.identity.str(),
        platform::implementationPlatformSchema.version,
        corner->corner.artifact};
    if (auto stored = artifactStore.get(platformReference); !stored)
      return stored.takeError();
    fabricEntry->dependencyClosure.push_back(platformReference);
    if (llvm::none_of(entries, [&](const auto &entry) {
          return entry.artifact == platformReference;
        }))
      entries.push_back({platformReference, {}});
  }
  llvm::sort(fabricEntry->dependencyClosure, artifactRootReferenceLess);
  fabricEntry->dependencyClosure.erase(
      std::unique(fabricEntry->dependencyClosure.begin(),
                  fabricEntry->dependencyClosure.end()),
      fabricEntry->dependencyClosure.end());
  return CaseArtifactResolution::get(std::move(entries));
}

} // namespace

llvm::Error registerFabricLowConfidenceModel() {
  if (llvm::Error error =
          platform::registerImplementationPlatformLocalReferenceKinds())
    return error;
  if (llvm::Error error = registerEvaluationCaseSignature(kCaseSignature))
    return error;
  if (llvm::Error error = registerEvaluationModelDescriptor(descriptor()))
    return error;
  return registerEvaluationModelProvider(provider());
}

EvaluationModelDescriptorRef fabricLowConfidenceModelDescriptorRef() {
  return descriptor().reference();
}

CaseSubjectRoleRef fabricHardwareAnalysisSubjectRole() { return kFabricRole; }

llvm::Expected<PreparedFabricLowConfidenceEvaluation>
prepareFabricLowConfidenceEvaluation(
    const ArtifactRootReference &fabricReference,
    llvm::ArrayRef<EvaluationCondition> conditions,
    llvm::ArrayRef<MetricKind> metrics, const ResolvedConfig &config,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  if (llvm::Error error = registerFabricLowConfidenceModel())
    return std::move(error);
  if (metrics.empty())
    return invalid("metric set is empty");
  auto resolution = resolveCase(fabricReference, conditions, artifactStore);
  if (!resolution)
    return resolution.takeError();
  auto subjects =
      EvaluationSubjectBindings::get({{kFabricRole, {fabricReference}}});
  if (!subjects)
    return subjects.takeError();
  auto evaluationCase = EvaluationCase::get(
      caseSignatureRef(), std::move(*subjects), std::nullopt, std::nullopt,
      conditions, *resolution, artifactStore, blobStore);
  if (!evaluationCase)
    return evaluationCase.takeError();

  std::vector<MetricRequest> requests;
  requests.reserve(metrics.size());
  for (MetricKind metricKind : metrics) {
    auto metric =
        MetricRequest::get({metricKind, EvaluationScope{kWholeCaseScope, {}}},
                           {}, *evaluationCase, *resolution, artifactStore);
    if (!metric)
      return metric.takeError();
    requests.push_back(std::move(*metric));
  }
  auto binding =
      ResolvedModelBinding::project(descriptor().reference(), {}, config);
  if (!binding)
    return binding.takeError();
  auto request =
      EvaluationRequest::get(*evaluationCase, requests, {}, std::move(*binding),
                             0, *resolution, artifactStore, blobStore);
  if (!request)
    return request.takeError();
  auto published = publishEvaluationRequest(*request, artifactStore);
  if (!published)
    return published.takeError();
  return PreparedFabricLowConfidenceEvaluation{
      std::move(*request), std::move(*resolution), kFabricRole};
}

} // namespace loom::evaluation::models
