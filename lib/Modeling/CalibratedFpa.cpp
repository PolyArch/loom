#include "Evaluation/Models/CalibratedFpa.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Evaluation/ModelParameter.h"
#include "Evaluation/ModelParameterBundle.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/Models/FpaParameterContract.h"
#include "Evaluation/ProductionRegistry.h"
#include "Evaluation/Request.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <system_error>
#include <vector>

namespace loom::evaluation::models {
namespace {

constexpr ModelInputSlotRef kParameterInput(0);

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      llvm::Twine("calibrated_fpa_invalid: ") + message);
}

llvm::Expected<EvaluationCase> reconstructCase(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  const EvaluationModelDescriptor *descriptor =
      request.modelBinding().descriptorRef().descriptor();
  if (!descriptor)
    return invalid("model descriptor is unavailable");
  return EvaluationCase::get(descriptor->caseSignature,
                             request.subjectBindings(), request.workload(),
                             request.runtimeInput(), request.baseConditions(),
                             resolution, artifactStore, blobStore);
}

llvm::Expected<EvaluationModelResult>
evaluate(const EvaluationRequest &request,
         const CaseArtifactResolution &resolution,
         const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  const ModelInputBinding *input =
      request.modelBinding().findInputBinding(kParameterInput);
  if (!input || input->artifacts.size() != 1)
    return invalid("parameter input is not total");
  auto bundle = importModelParameterBundle(input->artifacts.front(),
                                           artifactStore, blobStore);
  if (!bundle)
    return bundle.takeError();
  if (bundle->bundle().parameterContract() != fpaModelParameterContractRef())
    return invalid("parameter input has a foreign contract");

  const bool hardwareOnly =
      request.modelBinding().descriptorRef().modelKind() ==
      builtinEvaluationModelKind(BuiltinEvaluationModel::FabricCalibratedFpa);
  if (hardwareOnly &&
      llvm::any_of(request.metricRequests(),
                   [](const MetricRequest &metric) {
                     return metric.query().metric == MetricKind::DynamicPower;
                   }) &&
      llvm::none_of(
          request.baseConditions(), [](const EvaluationCondition &condition) {
            return condition.kind() == EvaluationConditionKind::ActivityBinding;
          }))
    return EvaluationModelResult{
        {}, UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};

  auto evaluationCase =
      reconstructCase(request, resolution, artifactStore, blobStore);
  if (!evaluationCase)
    return evaluationCase.takeError();
  auto features =
      projectModelFeatures(fpaModelParameterContractRef(), *evaluationCase,
                           resolution, artifactStore, blobStore);
  if (!features)
    return features.takeError();
  auto inference = inferModelParameters(*bundle, *features);
  if (!inference)
    return inference.takeError();
  if (std::holds_alternative<OutOfDomainModelParameterInference>(*inference))
    return EvaluationModelResult{
        {}, UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
  const auto *prediction = std::get<ModelParameterPrediction>(*inference)
                               .view.getIf<FpaMetricPredictionView>();
  if (!prediction)
    return invalid("parameter contract returned a foreign prediction view");

  std::vector<MetricResult> results;
  results.reserve(request.metricRequests().size());
  for (const MetricRequest &metric : request.metricRequests()) {
    const DecimalValue *value = nullptr;
    switch (metric.query().metric) {
    case MetricKind::LimitingClockFrequency:
      value = &prediction->limitingClockFrequency;
      break;
    case MetricKind::TotalArea:
      value = &prediction->totalArea;
      break;
    case MetricKind::DynamicPower:
      value = &prediction->dynamicPower;
      break;
    case MetricKind::LeakagePower:
      value = &prediction->leakagePower;
      break;
    default:
      return invalid("request contains a foreign metric");
    }
    results.push_back({UncertaintyKind::Unquantified,
                       PointObservation{MetricValue{*value}},
                       {kParameterInput}});
  }
  return EvaluationModelResult{{}, CompletedEvidence{std::move(results), {}}};
}

EvaluationModelDescriptorRef modelRef(BuiltinEvaluationModel model) {
  return llvm::cantFail(builtinEvaluationModelDescriptorRef(model));
}

const EvaluationModelProvider kProviders[] = {
    {modelRef(BuiltinEvaluationModel::StructuredFabricCalibratedFpa),
     EvaluationModelInProcessProvider{&evaluate}},
    {modelRef(BuiltinEvaluationModel::CanonicalDataflowFabricCalibratedFpa),
     EvaluationModelInProcessProvider{&evaluate}},
    {modelRef(BuiltinEvaluationModel::FabricCalibratedFpa),
     EvaluationModelInProcessProvider{&evaluate}},
};

} // namespace

llvm::Error registerCalibratedFpaProviders() {
  for (const EvaluationModelProvider &provider : kProviders)
    if (llvm::Error error = registerEvaluationModelProvider(provider))
      return error;
  return llvm::Error::success();
}

} // namespace loom::evaluation::models
