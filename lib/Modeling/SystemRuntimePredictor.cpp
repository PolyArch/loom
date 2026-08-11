#include "Evaluation/Models/SystemRuntimePredictor.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Evaluation/ModelParameter.h"
#include "Evaluation/ModelParameterBundle.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/Models/SystemRuntimeParameterContract.h"
#include "Evaluation/ProductionRegistry.h"
#include "Evaluation/Request.h"

#include "llvm/Support/Error.h"

#include <system_error>

namespace loom::evaluation::models {
namespace {

constexpr ModelInputSlotRef kParameterInput(0);

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      llvm::Twine("system_runtime_predictor_invalid: ") + message);
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
  if (bundle->bundle().parameterContract() !=
      systemRuntimeModelParameterContractRef())
    return invalid("parameter input has a foreign contract");

  const EvaluationModelDescriptor *descriptor =
      request.modelBinding().descriptorRef().descriptor();
  if (!descriptor)
    return invalid("model descriptor is unavailable");
  auto evaluationCase = EvaluationCase::get(
      descriptor->caseSignature, request.subjectBindings(), request.workload(),
      request.runtimeInput(), request.baseConditions(), resolution,
      artifactStore, blobStore);
  if (!evaluationCase)
    return evaluationCase.takeError();
  auto features = projectModelFeatures(systemRuntimeModelParameterContractRef(),
                                       *evaluationCase, resolution,
                                       artifactStore, blobStore);
  if (!features)
    return features.takeError();
  auto inference = inferModelParameters(*bundle, *features);
  if (!inference)
    return inference.takeError();
  if (std::holds_alternative<UnsupportedModelParameterInference>(*inference))
    return EvaluationModelResult{
        {}, UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
  const auto *prediction = std::get<ModelParameterPrediction>(*inference)
                               .view.getIf<SystemRuntimePredictionView>();
  if (!prediction)
    return invalid("parameter contract returned a foreign prediction view");
  if (request.metricRequests().size() != 1 ||
      request.metricRequests().front().query().metric != MetricKind::Runtime)
    return invalid("request does not contain exactly one Runtime metric");
  MetricResult result{UncertaintyKind::Unquantified,
                      PointObservation{MetricValue{prediction->runtime}},
                      {kParameterInput}};
  return EvaluationModelResult{{}, CompletedEvidence{{std::move(result)}, {}}};
}

const EvaluationModelProvider kProvider{
    systemRuntimePredictorModelDescriptorRef(),
    EvaluationModelInProcessProvider{&evaluate}};

} // namespace

EvaluationModelDescriptorRef systemRuntimePredictorModelDescriptorRef() {
  return llvm::cantFail(builtinEvaluationModelDescriptorRef(
      BuiltinEvaluationModel::Gem5CgraSystemRuntimePredictor));
}

llvm::Error registerSystemRuntimePredictorProvider() {
  return registerEvaluationModelProvider(kProvider);
}

} // namespace loom::evaluation::models
