#include "Evaluation/Models/CalibratedFpa.h"

#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Evaluation/ModelParameter.h"
#include "Evaluation/ModelParameterBundle.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/Models/FpaParameterContract.h"
#include "Evaluation/ProductionRegistry.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "llvm/ADT/STLExtras.h"

#include <array>
#include <optional>
#include <utility>
#include <vector>

namespace loom::evaluation::models {
namespace {

constexpr ModelInputSlotRef kParameterInput(0);
constexpr ModelInputSlotRef kPlatformInput(1);
constexpr ScopeFormRef kWholeCaseScope(0);

struct EmptyCalibratedFpaConfig final {};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "calibrated_fpa_invalid: " + message);
}

llvm::ArrayRef<std::uint8_t> configSchemaBytes() {
  static constexpr llvm::StringLiteral schema =
      "loom.fpa.calibrated_predictor.config.1.0";
  return {reinterpret_cast<const std::uint8_t *>(schema.data()), schema.size()};
}

llvm::Expected<OwnerValue> projectConfig(const ResolvedConfig &) {
  return OwnerValue::get(EmptyCalibratedFpaConfig{});
}

llvm::Expected<std::vector<std::uint8_t>>
encodeConfig(const OwnerValue &value) {
  if (!value.getIf<EmptyCalibratedFpaConfig>())
    return invalid("config has a foreign owner type");
  return std::vector<std::uint8_t>{};
}

llvm::Expected<OwnerValue> adoptConfig(llvm::ArrayRef<std::uint8_t> bytes,
                                       const ComponentViewDigest &) {
  if (!bytes.empty())
    return invalid("fixed calibrated predictor config is not empty");
  return OwnerValue::get(EmptyCalibratedFpaConfig{});
}

const ResolvedModelConfigViewContract kConfigView{
    configSchemaBytes(), &projectConfig, &encodeConfig, &adoptConfig};

const ArtifactSchemaDescriptor *const kParameterSchemas[] = {
    &modelParameterBundleSchema};
const ArtifactSchemaDescriptor *const kPlatformSchemas[] = {
    &platform::implementationPlatformSchema};

llvm::Error verifyPlatformCompatibility(
    llvm::ArrayRef<ArtifactRootReference> artifacts,
    const EvaluationCase &evaluationCase, const CaseArtifactResolution &,
    const ArtifactStore &artifactStore, const BlobStore &) {
  if (artifacts.size() != 1)
    return invalid(
        "predictor does not bind exactly one ImplementationPlatform");
  auto implementationPlatform =
      platform::importImplementationPlatform(artifacts.front(), artifactStore);
  if (!implementationPlatform)
    return implementationPlatform.takeError();
  for (const EvaluationCondition &condition : evaluationCase.baseConditions()) {
    const auto *corner =
        std::get_if<ProcessCornerCondition>(&condition.payload);
    if (!corner)
      continue;
    if (corner->corner.artifact != artifacts.front().artifact)
      return invalid(
          "process corner belongs to a foreign ImplementationPlatform");
    if (!implementationPlatform->platform().findTechnologyCorner(
            corner->corner.entity))
      return invalid("process corner is absent from the bound platform");
  }
  return llvm::Error::success();
}

const ModelInputSlotDescriptor kInputSlots[] = {
    {kParameterInput, "model_parameter_bundle", kParameterSchemas,
     ArtifactCollectionCardinality::ExactlyOne, nullptr,
     fpaModelParameterContractRef()},
    {kPlatformInput, "implementation_platform", kPlatformSchemas,
     ArtifactCollectionCardinality::ExactlyOne, &verifyPlatformCompatibility},
};

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

const std::vector<ModelConditionCapability> &
conditionCapabilities(BuiltinEvaluationCase evaluationCase) {
  static const std::vector<ModelConditionCapability> structured = [] {
    std::vector<ModelConditionCapability> result;
    const auto reference = builtinEvaluationCaseSignatureRef(
        BuiltinEvaluationCase::StructuredProgramWithFabric);
    for (const ConditionApplicabilityPattern &pattern :
         reference.descriptor()->permittedBaseConditions)
      result.push_back({pattern, ConditionDisposition::Consumed});
    return result;
  }();
  static const std::vector<ModelConditionCapability> dataflow = [] {
    std::vector<ModelConditionCapability> result;
    const auto reference = builtinEvaluationCaseSignatureRef(
        BuiltinEvaluationCase::CanonicalDataflowWithFabric);
    for (const ConditionApplicabilityPattern &pattern :
         reference.descriptor()->permittedBaseConditions)
      result.push_back({pattern, ConditionDisposition::Consumed});
    return result;
  }();
  static const std::vector<ModelConditionCapability> fabricOnly = [] {
    std::vector<ModelConditionCapability> result;
    const auto reference = fabricHardwareAnalysisCaseSignatureRef();
    for (const ConditionApplicabilityPattern &pattern :
         reference.descriptor()->permittedBaseConditions)
      result.push_back({pattern, ConditionDisposition::Consumed});
    return result;
  }();
  switch (evaluationCase) {
  case BuiltinEvaluationCase::StructuredProgramWithFabric:
    return structured;
  case BuiltinEvaluationCase::CanonicalDataflowWithFabric:
    return dataflow;
  case BuiltinEvaluationCase::FabricHardwareAnalysis:
    return fabricOnly;
  default:
    llvm_unreachable("foreign calibrated FPA prediction case");
  }
}

const EvaluationModelDescriptor &structuredDescriptor() {
  static const EvaluationModelDescriptor value{
      builtinEvaluationModelKind(
          BuiltinEvaluationModel::StructuredFabricCalibratedFpa),
      "structured_fabric_calibrated_fpa",
      "loom.structured_fabric.calibrated_fpa.v1",
      builtinEvaluationCaseSignatureRef(
          BuiltinEvaluationCase::StructuredProgramWithFabric),
      conditionCapabilities(BuiltinEvaluationCase::StructuredProgramWithFabric),
      kMetricCapabilities,
      {},
      kInputSlots,
      {},
      kConfigView,
      kModeledPhenomena,
      EvaluationExecutionMethod::Analytic,
      {},
      DeterminismContract::Deterministic,
      {},
      ProviderForm::InProcess};
  return value;
}

const EvaluationModelDescriptor &dataflowDescriptor() {
  static const EvaluationModelDescriptor value{
      builtinEvaluationModelKind(
          BuiltinEvaluationModel::CanonicalDataflowFabricCalibratedFpa),
      "canonical_dataflow_fabric_calibrated_fpa",
      "loom.canonical_dataflow_fabric.calibrated_fpa.v1",
      builtinEvaluationCaseSignatureRef(
          BuiltinEvaluationCase::CanonicalDataflowWithFabric),
      conditionCapabilities(BuiltinEvaluationCase::CanonicalDataflowWithFabric),
      kMetricCapabilities,
      {},
      kInputSlots,
      {},
      kConfigView,
      kModeledPhenomena,
      EvaluationExecutionMethod::Analytic,
      {},
      DeterminismContract::Deterministic,
      {},
      ProviderForm::InProcess};
  return value;
}

const EvaluationModelDescriptor &fabricDescriptor() {
  static const EvaluationModelDescriptor value{
      builtinEvaluationModelKind(BuiltinEvaluationModel::FabricCalibratedFpa),
      "fabric_calibrated_fpa",
      "loom.fabric.calibrated_fpa.v2",
      fabricHardwareAnalysisCaseSignatureRef(),
      conditionCapabilities(BuiltinEvaluationCase::FabricHardwareAnalysis),
      kMetricCapabilities,
      {},
      kInputSlots,
      {},
      kConfigView,
      kModeledPhenomena,
      EvaluationExecutionMethod::Analytic,
      {},
      DeterminismContract::Deterministic,
      {},
      ProviderForm::InProcess};
  return value;
}

bool supportedDescriptor(EvaluationModelDescriptorRef reference) {
  return reference == structuredDescriptor().reference() ||
         reference == dataflowDescriptor().reference() ||
         reference == fabricDescriptor().reference();
}

bool requestsDynamicPower(const EvaluationRequest &request) {
  return llvm::any_of(
      request.metricRequests(), [](const MetricRequest &metric) {
        return metric.query().metric == MetricKind::DynamicPower;
      });
}

llvm::Expected<DecimalValue>
predictionForMetric(const FpaMetricPredictionView &prediction,
                    MetricKind metric) {
  switch (metric) {
  case MetricKind::LimitingClockFrequency:
    return prediction.limitingClockFrequency;
  case MetricKind::TotalArea:
    return prediction.totalArea;
  case MetricKind::DynamicPower:
    return prediction.dynamicPower;
  case MetricKind::LeakagePower:
    return prediction.leakagePower;
  default:
    return invalid("request contains a metric outside the FPA contract");
  }
}

llvm::Expected<EvaluationModelResult>
evaluate(const EvaluationRequest &request,
         const CaseArtifactResolution &resolution,
         const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  RequestVerifier verifier(resolution, artifactStore, blobStore);
  if (llvm::Error error = verifier.verify(request))
    return std::move(error);
  if (!supportedDescriptor(request.modelBinding().descriptorRef()))
    return invalid("request selects a foreign model descriptor");

  bool hasExplicitActivity = false;
  for (const EvaluationCondition &condition : request.baseConditions()) {
    const auto *binding =
        std::get_if<ActivityBindingCondition>(&condition.payload);
    if (!binding)
      continue;
    if (!std::holds_alternative<ExplicitAssumptionSource>(binding->source))
      return EvaluationModelResult{
          {}, UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
    hasExplicitActivity = true;
  }
  if (requestsDynamicPower(request) && !hasExplicitActivity)
    return EvaluationModelResult{
        {}, UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};

  const ModelInputBinding *parameterBinding =
      request.modelBinding().findInputBinding(kParameterInput);
  if (!parameterBinding || parameterBinding->artifacts.size() != 1)
    return invalid("request does not bind exactly one parameter bundle");
  auto bundle = importModelParameterBundle(parameterBinding->artifacts.front(),
                                           artifactStore, blobStore);
  if (!bundle)
    return bundle.takeError();

  const EvaluationModelDescriptor *model =
      request.modelBinding().descriptorRef().descriptor();
  auto evaluationCase = EvaluationCase::get(
      model->caseSignature, request.subjectBindings(), request.workload(),
      request.runtimeInput(), request.baseConditions(), resolution,
      artifactStore, blobStore);
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
  const auto *predicted = std::get_if<ModelParameterPrediction>(&*inference);
  if (!predicted)
    return EvaluationModelResult{
        {}, UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
  const auto *view = predicted->view.getIf<FpaMetricPredictionView>();
  if (!view)
    return invalid("parameter contract returned a foreign prediction view");

  std::vector<MetricResult> results;
  results.reserve(request.metricRequests().size());
  for (const MetricRequest &metric : request.metricRequests()) {
    auto value = predictionForMetric(*view, metric.query().metric);
    if (!value)
      return value.takeError();
    results.push_back({UncertaintyKind::Unquantified,
                       PointObservation{*value},
                       {kParameterInput}});
  }
  return EvaluationModelResult{{}, CompletedEvidence{std::move(results), {}}};
}

const EvaluationModelProvider &structuredProvider() {
  static const EvaluationModelProvider value{
      structuredDescriptor().reference(),
      EvaluationModelInProcessProvider{&evaluate}};
  return value;
}

const EvaluationModelProvider &dataflowProvider() {
  static const EvaluationModelProvider value{
      dataflowDescriptor().reference(),
      EvaluationModelInProcessProvider{&evaluate}};
  return value;
}

const EvaluationModelProvider &fabricProvider() {
  static const EvaluationModelProvider value{
      fabricDescriptor().reference(),
      EvaluationModelInProcessProvider{&evaluate}};
  return value;
}

} // namespace

llvm::Error registerCalibratedFpaModels() {
  for (const EvaluationModelDescriptor *model :
       {&structuredDescriptor(), &dataflowDescriptor(), &fabricDescriptor()})
    if (llvm::Error error = registerEvaluationModelDescriptor(*model))
      return error;
  for (const EvaluationModelProvider *modelProvider :
       {&structuredProvider(), &dataflowProvider(), &fabricProvider()})
    if (llvm::Error error = registerEvaluationModelProvider(*modelProvider))
      return error;
  return llvm::Error::success();
}

EvaluationModelDescriptorRef structuredFabricCalibratedFpaModelDescriptorRef() {
  return structuredDescriptor().reference();
}

EvaluationModelDescriptorRef
canonicalDataflowFabricCalibratedFpaModelDescriptorRef() {
  return dataflowDescriptor().reference();
}

EvaluationModelDescriptorRef fabricCalibratedFpaModelDescriptorRef() {
  return fabricDescriptor().reference();
}

} // namespace loom::evaluation::models
