#include "Evaluation/Models/CanonicalDataflowFabricAnalytic.h"

#include "AnalyticModelSupport.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Evaluation/ModelProvider.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace loom::evaluation::models {
namespace {

constexpr EvaluationCaseKind kCaseKind(1);
constexpr EvaluationModelKind kModelKind(3);
constexpr CaseSubjectRoleRef kCanonicalDataflowRole(0);
constexpr CaseSubjectRoleRef kFabricRole(1);

EvaluationCaseSignatureRef caseSignatureRef() {
  return llvm::cantFail(
      EvaluationCaseSignatureRef::get(evaluationSchemaVersion(), kCaseKind));
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
const EvaluationCaseSignatureDescriptor kCaseSignature{
    kCaseKind,
    "canonical_dataflow_with_fabric",
    "One exact Canonical Dataflow Program evaluated against one exact Fabric.",
    kSubjectRoles,
    ArtifactRequirement::Forbidden,
    {},
    ArtifactRequirement::Forbidden,
    {},
    nullptr,
    AbsentReferenceCycle{},
    {}};

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
    kModelKind,
    "canonical_dataflow_fabric_low_confidence",
    "loom.canonical_dataflow_fabric.low_confidence.v1",
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
    {}};

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
         const ArtifactStore &artifactStore) {
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
      fabric::importEntireFabricRoot(fabrics.front(), artifactStore);
  if (!fabricRoot)
    return fabricRoot.takeError();
  auto metrics = estimateMetrics(*program, *fabricRoot);
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

const EvaluationModelProvider kProvider{kModelDescriptor.reference(),
                                        &evaluate};

} // namespace

llvm::Error registerCanonicalDataflowFabricAnalyticModel() {
  if (llvm::Error error = registerEvaluationCaseSignature(kCaseSignature))
    return error;
  if (llvm::Error error = registerEvaluationModelDescriptor(kModelDescriptor))
    return error;
  return registerEvaluationModelProvider(kProvider);
}

llvm::Expected<PreparedCanonicalDataflowFabricEvaluation>
prepareCanonicalDataflowFabricEvaluation(
    const ArtifactRootReference &canonicalDataflow,
    const ArtifactRootReference &fabricReference, const ResolvedConfig &config,
    const ArtifactStore &artifactStore) {
  if (llvm::Error error = registerCanonicalDataflowFabricAnalyticModel())
    return std::move(error);

  auto program =
      dataflow::importCanonicalDataflow(canonicalDataflow, artifactStore);
  if (!program)
    return program.takeError();
  auto fabricRoot =
      fabric::importEntireFabricRoot(fabricReference, artifactStore);
  if (!fabricRoot)
    return fabricRoot.takeError();
  auto resolution = detail::resolveSingleSubjectFabricCase(
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
      *resolution, artifactStore);
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
                                        *resolution, artifactStore);
  if (!request)
    return request.takeError();
  auto published = publishEvaluationRequest(*request, artifactStore);
  if (!published)
    return published.takeError();
  return PreparedCanonicalDataflowFabricEvaluation{std::move(*request),
                                                   std::move(*resolution)};
}

} // namespace loom::evaluation::models
