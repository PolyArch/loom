#include "Evaluation/Models/CanonicalDataflowFabricAnalytic.h"
#include "Evaluation/ProductionRegistry.h"

#include "AnalyticModelSupport.h"
#include "StructuredEvaluationInvocationCacheInternal.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Evaluation/ModelProvider.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"

#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace loom::evaluation::models {
namespace {

constexpr BuiltinEvaluationCase kCase =
    BuiltinEvaluationCase::CanonicalDataflowWithFabric;
constexpr BuiltinEvaluationModel kModel =
    BuiltinEvaluationModel::CanonicalDataflowFabricLowConfidence;
constexpr CaseSubjectRoleRef kCanonicalDataflowRole(0);
constexpr CaseSubjectRoleRef kFabricRole(1);

EvaluationCaseSignatureRef caseSignatureRef() {
  return llvm::cantFail(EvaluationCaseSignatureRef::get(
      evaluationSchemaVersion(), builtinEvaluationCaseKind(kCase)));
}

const ArtifactSchemaDescriptor *const kDataflowSchemas[] = {
    &dataflow::canonicalDataflowSchema};
const ArtifactSchemaDescriptor *const kFabricSchemas[] = {
    &fabric::fabricArtifactSchema};
const CaseSubjectRoleDescriptor kSubjectRoles[] = {
    {kCanonicalDataflowRole, "canonical_dataflow",
     SubjectRoleCardinality::ExactlyOne, kDataflowSchemas, nullptr},
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
    "canonical_dataflow_with_fabric",
    "One exact Canonical Dataflow Program evaluated against one exact Fabric.",
    kSubjectRoles,
    ArtifactRequirement::Forbidden,
    {},
    ArtifactRequirement::Forbidden,
    {},
    nullptr,
    AbsentReferenceCycle{},
    kBaseConditionPatterns};

const ScopeFormRef kWholeCaseScopeForms[] = {ScopeFormRef(0)};
const MetricCapability kMetricCapabilities[] = {
    {MetricKind::Runtime, kWholeCaseScopeForms,
     observationFormMask(ObservationForm::Point)},
    {MetricKind::LimitingClockFrequency, kWholeCaseScopeForms,
     observationFormMask(ObservationForm::Point)},
    {MetricKind::TotalArea, kWholeCaseScopeForms,
     observationFormMask(ObservationForm::Point)},
    {MetricKind::DynamicPower, kWholeCaseScopeForms,
     observationFormMask(ObservationForm::Point)},
    {MetricKind::LeakagePower, kWholeCaseScopeForms,
     observationFormMask(ObservationForm::Point)}};
const ModeledPhenomenon kModeledPhenomena[] = {
    ModeledPhenomenon::CanonicalDataflow, ModeledPhenomenon::SpatialResources};
const EvaluationModelDescriptor kModelDescriptor{
    builtinEvaluationModelKind(kModel),
    "canonical_dataflow_fabric_low_confidence",
    "loom.canonical_dataflow_fabric.low_confidence.v3",
    caseSignatureRef(),
    {},
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

llvm::Expected<std::optional<detail::LowConfidenceMetricSet>>
estimateMetrics(const dataflow::CanonicalDataflowArtifact &program,
                const fabric::FinalizedFabricRoot &fabricRoot) {
  auto view = program.view();
  if (!view)
    return view.takeError();

  auto pressure = detail::projectCanonicalDataflowWorkload(*view, fabricRoot);
  if (!pressure)
    return pressure.takeError();
  if (!*pressure)
    return std::optional<detail::LowConfidenceMetricSet>{};

  auto metrics =
      detail::estimateLowConfidenceMetrics(0, **pressure, fabricRoot);
  if (!metrics)
    return metrics.takeError();
  return std::optional<detail::LowConfidenceMetricSet>(std::move(*metrics));
}

llvm::Expected<EvaluationModelResult>
evaluate(const EvaluationRequest &request, const CaseArtifactResolution &,
         const ArtifactStore &artifactStore, const BlobStore &) {
  llvm::ArrayRef<ArtifactRootReference> dataflowPrograms =
      request.subjectBindings().subjects(kCanonicalDataflowRole);
  llvm::ArrayRef<ArtifactRootReference> fabrics =
      request.subjectBindings().subjects(kFabricRole);
  if (dataflowPrograms.size() != 1 || fabrics.size() != 1)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "canonical_dataflow_fabric_model_invalid: exact subjects are not "
        "total");

  auto program = dataflow::importCanonicalDataflow(dataflowPrograms.front(),
                                                   artifactStore);
  if (!program)
    return program.takeError();
  auto fabricRoot =
      detail::importCachedFabricRoot(fabrics.front(), artifactStore);
  if (!fabricRoot)
    return fabricRoot.takeError();
  auto metrics = estimateMetrics(*program, **fabricRoot);
  if (!metrics)
    return metrics.takeError();
  if (!*metrics)
    return EvaluationModelResult{
        {}, UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};

  std::vector<MetricResult> metricResults;
  metricResults.reserve(request.metricRequests().size());
  for (const MetricRequest &metric : request.metricRequests()) {
    auto result = (**metrics).result(metric.query().metric);
    if (!result)
      return result.takeError();
    metricResults.push_back(std::move(*result));
  }
  return EvaluationModelResult{{},
                               CompletedEvidence{std::move(metricResults), {}}};
}

const EvaluationModelProvider kProvider{
    kModelDescriptor.reference(), EvaluationModelInProcessProvider{&evaluate}};

} // namespace

llvm::Error registerCanonicalDataflowFabricAnalyticModel() {
  if (llvm::Error error = registerEvaluationCaseSignature(kCaseSignature))
    return error;
  if (llvm::Error error = registerEvaluationModelDescriptor(kModelDescriptor))
    return error;
  return registerEvaluationModelProvider(kProvider);
}

EvaluationModelDescriptorRef
canonicalDataflowFabricAnalyticModelDescriptorRef() {
  return kModelDescriptor.reference();
}

CaseSubjectRoleRef canonicalDataflowFabricAnalyticCandidateRole() {
  return kCanonicalDataflowRole;
}

CaseSubjectRoleRef canonicalDataflowFabricAnalyticFabricRole() {
  return kFabricRole;
}

llvm::Expected<std::int64_t>
canonicalDataflowFabricAnalyticMetricQuantumBase10Exponent(MetricKind metric) {
  return detail::lowConfidenceMetricQuantumBase10Exponent(metric);
}

llvm::Expected<CaseArtifactResolution>
resolveCanonicalDataflowFabricEvaluationCase(
    const ArtifactRootReference &canonicalDataflow,
    const ArtifactRootReference &fabricReference,
    const ArtifactStore &artifactStore) {
  return resolveCanonicalDataflowFabricEvaluationCases(
      {canonicalDataflow}, fabricReference, artifactStore);
}

llvm::Expected<CaseArtifactResolution>
resolveCanonicalDataflowFabricEvaluationCases(
    llvm::ArrayRef<ArtifactRootReference> canonicalDataflowPrograms,
    const ArtifactRootReference &fabricReference,
    const ArtifactStore &artifactStore) {
  if (canonicalDataflowPrograms.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "canonical_dataflow_fabric_model_invalid: invocation has no "
        "Canonical Dataflow candidates");
  std::vector<ArtifactRootReference> candidates(
      canonicalDataflowPrograms.begin(), canonicalDataflowPrograms.end());
  llvm::sort(candidates, artifactRootReferenceLess);
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());
  std::vector<CaseArtifactResolution::Entry> additionalEntries;
  additionalEntries.reserve(candidates.size());
  for (const ArtifactRootReference &candidate : candidates) {
    if (candidate.schemaIdentity !=
            dataflow::canonicalDataflowSchema.identity ||
        candidate.schemaVersion != dataflow::canonicalDataflowSchema.version)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "canonical_dataflow_fabric_model_invalid: invocation contains a "
          "foreign candidate");
    auto stored = artifactStore.get(candidate);
    if (!stored)
      return stored.takeError();
    additionalEntries.push_back({candidate, {}});
  }
  return detail::resolveSingleSubjectFabricCase(
      candidates.front(), fabricReference, artifactStore, additionalEntries);
}

llvm::Expected<PreparedCanonicalDataflowFabricEvaluation>
prepareCanonicalDataflowFabricEvaluation(
    const ArtifactRootReference &canonicalDataflow,
    const ArtifactRootReference &fabricReference, const ResolvedConfig &config,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  if (llvm::Error error = registerCanonicalDataflowFabricAnalyticModel())
    return std::move(error);

  auto resolution = resolveCanonicalDataflowFabricEvaluationCase(
      canonicalDataflow, fabricReference, artifactStore);
  if (!resolution)
    return resolution.takeError();

  auto bindings = EvaluationSubjectBindings::get(
      {{kCanonicalDataflowRole, {canonicalDataflow}},
       {kFabricRole, {fabricReference}}});
  if (!bindings)
    return bindings.takeError();
  auto evaluationCase = EvaluationCase::get(
      caseSignatureRef(), std::move(*bindings), std::nullopt, std::nullopt, {},
      *resolution, artifactStore, blobStore);
  if (!evaluationCase)
    return evaluationCase.takeError();
  std::vector<MetricRequest> metrics;
  metrics.reserve(std::size(kMetricCapabilities));
  for (const MetricCapability &capability : kMetricCapabilities) {
    auto metric = MetricRequest::get(
        MetricQuery{capability.kind, EvaluationScope{ScopeFormRef(0), {}}}, {},
        *evaluationCase, *resolution, artifactStore);
    if (!metric)
      return metric.takeError();
    metrics.push_back(std::move(*metric));
  }
  auto modelBinding =
      ResolvedModelBinding::project(kModelDescriptor.reference(), {}, config);
  if (!modelBinding)
    return modelBinding.takeError();
  auto request = EvaluationRequest::get(*evaluationCase, metrics, {},
                                        std::move(*modelBinding), 0,
                                        *resolution, artifactStore, blobStore);
  if (!request)
    return request.takeError();
  auto published = publishEvaluationRequest(*request, artifactStore);
  if (!published)
    return published.takeError();
  return PreparedCanonicalDataflowFabricEvaluation{
      std::move(*request), std::move(*resolution), kCanonicalDataflowRole};
}

} // namespace loom::evaluation::models
